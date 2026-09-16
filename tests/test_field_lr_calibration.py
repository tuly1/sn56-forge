"""Synthetic CPU tests of the actual experimental Trainer integration."""
import ast
import copy
import hashlib
import inspect
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig, Trainer, TrainingArguments

from forge.tuning import field_lr_calibration as cal


class TinyLM(torch.nn.Module):
    loss_type = 'ForCausalLM'
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.embedding = torch.nn.Embedding(13, 5)
        self.output = torch.nn.Linear(5, 13, bias=False)
        self.output.weight = self.embedding.weight
        self.dropout = torch.nn.Dropout(.1)
        self.register_buffer('calls', torch.zeros((), dtype=torch.int64))
        self.register_buffer('scratch', torch.zeros(1), persistent=False)
        self.starts = []
        self.fail = False

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, labels=None, num_items_in_batch=None, **kwargs):
        if self.calls.item() == 0:
            self.starts.append((self.embedding.weight.detach().clone(), torch.get_rng_state().clone(),
                                random.getstate(), copy.deepcopy(np.random.get_state())))
        self.calls.add_(1)
        self.scratch = self.scratch + 1
        random.random(); np.random.rand()
        if self.fail:
            raise RuntimeError('synthetic forward failure')
        logits = self.output(self.dropout(self.embedding(input_ids)))
        target = labels[:, 1:].reshape(-1)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 13), target, reduction='sum', ignore_index=-100)
        count = target.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
        return {'loss': loss / count, 'logits': logits}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    for key in ('FORGE_V2_FIELD_SWEEP', 'FORGE_V2_ROW_LOSS', 'FORGE_V2_LR', 'FORGE_V2_FIELD_LR_MULT'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('FORGE_V2_FIELD_LR_CALIBRATION', '1')
    model = TinyLM()
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False,
        per_device_train_batch_size=2, gradient_accumulation_steps=2, learning_rate=5e-5,
        optim='adamw_torch', weight_decay=0, max_grad_norm=1, report_to=[],
        neftune_noise_alpha=1.0)
    trainer = Trainer(model=model, args=args)
    batches = []
    for i in range(50):
        ids = (torch.arange(10).reshape(2, 5) + i) % 13
        labels = ids.clone(); labels[:, :1 + (i % 3)] = -100
        batches.append({'input_ids': ids, 'labels': labels})
    deadline = SimpleNamespace(remaining_hard=lambda: 5000., export_reserve_s=180.)
    return model, trainer, batches, deadline


def run(setup, **kwargs):
    model, trainer, batches, deadline = setup
    return cal.run_field_calibration(trainer, batches, prior_lr=5e-5, deadline=deadline,
                                    finish_reserve_s=390, **kwargs)


def assert_rng_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])


def rng():
    return (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone())


def test_real_eight_warmup_complete_panel_and_exact_recovery(setup):
    model, trainer, batches, _ = setup
    trainer.create_optimizer()
    original_optimizer = trainer.optimizer
    p = next(model.parameters())
    p.grad = torch.ones_like(p); trainer.optimizer.step()
    p.grad = torch.ones_like(p) * .2
    optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
    before = cal.InitialState(trainer)
    initial_rng = rng()
    model.eval(); model.dropout.train()
    original_modes = [m.training for m in model.modules()]
    original_buffers = dict(model.named_buffers())
    seen_updates = []
    original_create = trainer.create_optimizer
    def create():
        result = original_create()
        assert not result.state  # every arm starts with a fresh optimizer
        old_step = result.step
        def step(*a, **kw):
            seen_updates.append(result.param_groups[0]['lr'])
            return old_step(*a, **kw)
        result.step = step
        return result
    trainer.create_optimizer = create
    lr, timing, diag = run(setup)
    assert diag['status'] == 'selected', diag
    assert timing > 0 and 2e-5 <= lr <= 1e-4
    assert diag['warmup_completed_steps'] == 8
    assert len(diag['probes']) in (2, 3, 4)
    assert len(seen_updates) == 8 + 25 * len(diag['probes'])
    assert seen_updates[:8] == [5e-5] * 8
    assert all(p['completed_steps'] == 25 and p['state_restored'] for p in diag['probes'])
    assert diag['restoration_verified'] and len(diag['initial_state_sha256']) == 64
    assert cal.InitialState(trainer).sha256 == before.sha256
    assert model.output.weight is model.embedding.weight
    assert all(dict(model.named_buffers())[k] is b for k, b in original_buffers.items())
    assert torch.equal(p.grad, torch.ones_like(p) * .2)
    assert [m.training for m in model.modules()] == original_modes
    assert_rng_equal(rng(), initial_rng)
    assert trainer.optimizer is original_optimizer
    new_state = trainer.optimizer.state_dict()
    for key, value in optimizer_state['state'].items():
        for name, tensor in value.items():
            assert torch.equal(new_state['state'][key][name], tensor)
    assert trainer.args.learning_rate == 5e-5
    assert not hasattr(trainer, 'current_gradient_accumulation_steps')
    assert not model.embedding._forward_hooks and not hasattr(model.embedding, 'neftune_noise_alpha')
    # First forward of every arm observes the same parameters and stochastic state.
    assert len(model.starts) == 1 + len(diag['probes'])
    for start in model.starts[1:]:
        assert torch.equal(start[0], model.starts[0][0])
        assert torch.equal(start[1], model.starts[0][1])
        assert start[2] == model.starts[0][2]
        assert np.array_equal(start[3][1], model.starts[0][3][1])


def test_actual_trainer_gradient_accumulation_matches_one_token_weighted_batch(setup):
    model, trainer, batches, _ = setup
    trainer.neftune_noise_alpha = None
    model.dropout.p = 0
    trainer.create_optimizer()
    initial = copy.deepcopy(model.state_dict())
    a, b = batches[:2]
    assert a['labels'][:, 1:].ne(-100).sum() != b['labels'][:, 1:].ne(-100).sum()
    observed = cal._optimizer_step(trainer, [a,b], learning_rate=5e-5, before_micro=lambda: None)
    result = copy.deepcopy(model.state_dict())
    model.load_state_dict(initial)
    trainer.optimizer = None; trainer.create_optimizer()
    all_ids = torch.cat([a['input_ids'], b['input_ids']]); all_labels = torch.cat([a['labels'], b['labels']])
    trainer.optimizer.zero_grad()
    expected = model(all_ids, labels=all_labels)['loss']; expected.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    trainer.optimizer.step()
    assert observed == pytest.approx(float(expected.detach()), rel=1e-6)
    for key, value in model.named_parameters():
        assert torch.allclose(value, result[key], atol=1e-8, rtol=1e-6), key


@pytest.mark.parametrize('name,value,reason', [
    ('FORGE_V2_FIELD_LR_CALIBRATION','0','flag_off'),
    ('FORGE_V2_FIELD_SWEEP','1','conflicting_field_sweep'),
    ('FORGE_V2_ROW_LOSS','1','unsupported_row_objective'),
    ('FORGE_V2_LR','0.0001','conflicting_fixed_lr'),
    ('FORGE_V2_FIELD_LR_MULT','1.1','conflicting_fixed_lr'),
])
def test_experiment_fork_guards_do_not_run_probes(setup, monkeypatch, name, value, reason):
    monkeypatch.setenv(name, value)
    lr, timing, diag = run(setup)
    assert lr == 5e-5 and timing is None and diag['reason'] == reason
    assert not setup[0].starts


def test_forward_failure_recovers_prior_and_cleans_neftune(setup):
    model, trainer, _, _ = setup
    model.fail = True
    initial = cal.InitialState(trainer)
    before_rng = rng()
    lr, timing, diag = run(setup)
    assert lr == 5e-5 and diag['reason'] == 'RuntimeError'
    assert cal.InitialState(trainer).sha256 == initial.sha256 and diag['restoration_verified']
    assert_rng_equal(rng(), before_rng)
    assert not model.embedding._forward_hooks


def test_nonfinite_loss_never_selects(setup, monkeypatch):
    monkeypatch.setattr(setup[1], 'training_step', lambda *a, **k: torch.tensor(float('nan')))
    initial = cal.InitialState(setup[1])
    lr, _, diag = run(setup)
    assert lr == 5e-5 and diag['reason'] == 'Diverged'
    assert cal.InitialState(setup[1]).sha256 == initial.sha256


def test_no_time_or_incomplete_batch_panel_returns_prior_without_mutation(setup):
    model, trainer, batches, deadline = setup
    deadline.remaining_hard = lambda: 400.
    lr, _, diag = run(setup)
    assert lr == 5e-5 and diag['reason'] == 'insufficient_time_before_snapshot'
    assert not model.starts
    deadline.remaining_hard = lambda: 5000.
    batches.pop()
    lr, _, diag = run(setup)
    assert diag['reason'] == 'insufficient_distinct_microbatches' and not model.starts


def test_cooperative_timeout_restores_after_actual_optimizer_step(setup):
    model, trainer, _, _ = setup
    initial = cal.InitialState(trainer)
    now = [0.]
    original_step = trainer.training_step
    def step(*a, **kw):
        out = original_step(*a, **kw)
        now[0] += 100.
        return out
    trainer.training_step = step
    lr, _, diag = run(setup, clock=lambda: now[0])
    assert lr == 5e-5 and diag['reason'] == 'ProbeTimeout'
    assert model.starts and diag['restoration_verified']
    assert cal.InitialState(trainer).sha256 == initial.sha256


def test_snapshot_copy_time_is_inside_admission(setup, monkeypatch):
    now = [0.]
    original = cal.InitialState.__init__
    def slow(self, trainer):
        original(self, trainer); now[0] += 335.
    monkeypatch.setattr(cal.InitialState, '__init__', slow)
    _, _, diag = run(setup, clock=lambda: now[0])
    assert diag['reason'] == 'ProbeTimeout' and diag['elapsed_s'] >= 335.
    assert not setup[0].starts and diag['restoration_verified']


def test_restore_failure_must_escape_caller(setup, monkeypatch):
    def failure(self):
        raise cal.StateRestoreError('synthetic copy corruption')
    monkeypatch.setattr(cal.InitialState, 'restore', failure)
    with pytest.raises(cal.StateRestoreError):
        run(setup)


def test_mutated_parameter_topology_fail_closed(setup):
    model, trainer, _, _ = setup
    original = model.forward
    def forward(*a, **kw):
        out = original(*a, **kw)
        model.register_parameter('injected', torch.nn.Parameter(torch.ones(1)))
        return out
    model.forward = forward
    with pytest.raises(cal.StateRestoreError, match='topology'):
        run(setup)


def test_paged_optimizer_explicitly_keeps_prior(setup):
    setup[1].args.optim = 'paged_adamw_8bit'
    _, _, diag = run(setup)
    assert diag['reason'] == 'unsupported_optimizer' and not setup[0].starts


def test_field_hook_default_off_nonfield_short_circuit_and_failure_propagation(monkeypatch):
    from forge.tasks import sft_v2
    monkeypatch.delenv('FORGE_V2_FIELD_LR_CALIBRATION', raising=False)
    assert not sft_v2._field_lr_calibration()
    source = inspect.getsource(sft_v2.run)
    tree = ast.parse(source)
    gate = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'field_calibration' for t in n.targets))
    expr = compile(ast.Expression(gate.value), '<actual source gate>', 'eval')
    assert eval(expr, {'field': False, '_field_lr_calibration': lambda: pytest.fail('nonfield evaluated experiment')}) is False
    monkeypatch.setenv('FORGE_V2_FIELD_LR_CALIBRATION', '1')
    assert eval(expr, {'field': True, '_field_lr_calibration': sft_v2._field_lr_calibration}) is True
    handler = next(n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.body and isinstance(n.body[0], ast.If)
                   and isinstance(n.body[0].test, ast.Name) and n.body[0].test.id == 'field_calibration')
    # Execute actual exception-handler statements: an enabled unexpected failure
    # must raise, not take the old generic "log and continue" fallback.
    body = ast.Module(body=[ast.Try(body=[ast.Raise(exc=ast.Call(func=ast.Name(id='RuntimeError',ctx=ast.Load()),args=[],keywords=[]))],handlers=[handler],orelse=[],finalbody=[])], type_ignores=[])
    ast.fix_missing_locations(body)
    with pytest.raises(RuntimeError):
        exec(compile(body, '<actual source failure policy>', 'exec'), {'field_calibration': True})


def test_actual_gpu_field_geometry_optimizer_is_supported(setup, monkeypatch):
    from forge.tasks import sft_v2
    monkeypatch.delenv('FORGE_V2_FIELD_BF16', raising=False)
    geo = sft_v2.choose_geometry(params_b=1.24, max_len=1024, per_gpu_gb=80., vocab=128256, liger=False, bnb_ok=True, eff_default=100)
    assert geo.fp32_master and geo.optim == 'adamw_torch_fused'
    setup[1].args.optim = geo.optim
    assert cal._conflict(setup[1], 5e-5) is None
    # Torch's fused AdamW also supports this tiny CPU fixture. This validates
    # optimizer creation/updates, not the unmeasured CUDA kernel behavior.
    lr, _, diag = run(setup)
    assert diag['status'] == 'selected', diag
    assert diag['optim'] == 'adamw_torch_fused' and diag['restoration_verified']


def test_late_timeout_never_selects_completed_subset(setup):
    trainer = setup[1]
    now, micro_calls = [0.], [0]
    original = trainer.training_step
    def step(*a, **kw):
        result = original(*a, **kw)
        micro_calls[0] += 1
        now[0] += 1. if micro_calls[0] <= 66 else 20.
        return result
    trainer.training_step = step
    lr, _, diag = run(setup, clock=lambda: now[0])
    assert len(diag['probes']) == 1, diag
    assert diag['probes'][0]['status'] == 'complete'
    assert lr == 5e-5 and diag['status'] == 'prior' and diag['reason'] == 'ProbeTimeout'
    assert diag['restoration_verified']


def test_neftune_hook_is_active_during_probe_and_prior_attribute_restored(setup):
    model = setup[0]
    model.embedding.neftune_noise_alpha = 99.
    seen = []
    def observe(module, inputs, outputs):
        seen.append((module.neftune_noise_alpha, len(module._forward_hooks)))
    handle = model.embedding.register_forward_hook(observe)
    _, _, diag = run(setup)
    assert diag['status'] == 'selected'
    assert seen and all(alpha == 1. and hooks == 2 for alpha, hooks in seen)
    assert model.embedding.neftune_noise_alpha == 99.
    assert len(model.embedding._forward_hooks) == 1
    handle.remove()


def test_default_off_source_erases_to_exact_parent_ast():
    """Not just a gate assertion: erase only the inactive added branches and
    compare every remaining statement against the frozen parent handler."""
    from forge.tasks import sft_v2
    candidate = ast.parse(Path(sft_v2.__file__).read_text())
    baseline_path = Path(__file__).resolve().parents[2] / 'baseline/forge/tasks/sft_v2.py'
    if not baseline_path.exists():
        pytest.skip('archive-level parent comparison; run from supplied isolated package')
    baseline = ast.parse(baseline_path.read_text())
    class EraseInactive(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == '_field_lr_calibration':
                return None
            return self.generic_visit(node)
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'field_calibration' for t in node.targets):
                return None
            return self.generic_visit(node)
        def visit_If(self, node):
            if isinstance(node.test, ast.Name) and node.test.id == 'field_calibration':
                return [self.visit(x) for x in node.orelse]
            return self.generic_visit(node)
    assert ast.dump(EraseInactive().visit(candidate), include_attributes=False) == ast.dump(baseline, include_attributes=False)


def test_frozen_parameters_and_no_target_update_recover(setup):
    model, trainer, batches, _ = setup
    model.register_parameter('frozen', torch.nn.Parameter(torch.tensor([3.]), requires_grad=False))
    initial = cal.InitialState(trainer)
    model.frozen.add_(2.)
    initial.restore()
    assert model.frozen.item() == 3. and not model.frozen.requires_grad
    for batch in batches:
        batch['labels'].fill_(-100)
    lr, _, diag = run(setup)
    assert lr == 5e-5 and diag['reason'] == 'ValueError'
    assert diag['restoration_verified'] and not model.starts
