"""2026-09-17: family-specific lower field gates for Llama (10k rows / 5M tokens), validated on GossipCop-10k
(the exact Sept 14 dataset below the generic 15k / 8M gates). Other families keep the generic gates."""
from forge.tasks import sft_v2


def test_llama_gates_admit_gossipcop_10k_and_keep_codefeedback_8k_out(monkeypatch):
    for k in ("FORGE_V2_FIELD_TYPES", "FORGE_V2_FIELD_MIN_ROWS", "FORGE_V2_FIELD_MIN_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FORGE_V2_FIELD", "1")
    assert sft_v2.field_full_min_rows("llama") == 10000 and sft_v2.field_full_min_tokens("llama") == 5_000_000
    assert sft_v2.field_full_min_rows("lfm2") == 15000 and sft_v2.field_full_min_tokens("lfm2") == 8_000_000
    assert sft_v2.field_full_min_rows() == 15000 and sft_v2.field_full_min_tokens() == 8_000_000
    # GossipCop subsampled to 10k rows / 6.0M tokens: field route
    assert sft_v2.field_full_route(params_b=1.24, n_gpus=1, is_kl=False, model_type="llama", n_rows=10000,
                                   total_tokens=6_004_004, vocab_size=128256) is True
    # CodeFeedback 8k rows / 3.9M tokens: adapter (measured loss for full weights)
    assert sft_v2.field_full_route(params_b=1.24, n_gpus=1, is_kl=False, model_type="llama", n_rows=8000,
                                   total_tokens=3_900_000, vocab_size=128256) is False
    # LFM keeps the generic gates: a 12k-row / 6M-token lfm2 task stays pinned
    assert sft_v2.field_full_route(params_b=2.697, n_gpus=1, is_kl=False, model_type="lfm2", n_rows=12000,
                                   total_tokens=6_000_000, vocab_size=128000) is False
    # env override still wins
    monkeypatch.setenv("FORGE_V2_FIELD_MIN_ROWS", "20000")
    assert sft_v2.field_full_min_rows("llama") == 20000
