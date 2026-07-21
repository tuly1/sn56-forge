from dataclasses import dataclass
import inspect

import pytest

from forge.tasks.common import compatible_dataclass_kwargs
from forge.tasks import fallback


@dataclass
class _CurrentConfig:
    output_dir: str
    beta: float = 0.1


def test_compat_kwargs_drops_only_explicitly_audited_removals():
    result = compatible_dataclass_kwargs(
        _CurrentConfig,
        {
            "output_dir": "/tmp/work",
            "beta": 0.5,
            "overwrite_output_dir": True,
            "max_prompt_length": 512,
        },
        allow_removed={"overwrite_output_dir", "max_prompt_length"},
    )

    assert result == {"output_dir": "/tmp/work", "beta": 0.5}


def test_compat_kwargs_rejects_unreviewed_unknown_fields():
    with pytest.raises(TypeError, match="typo_field"):
        compatible_dataclass_kwargs(
            _CurrentConfig,
            {"output_dir": "/tmp/work", "typo_field": True},
            allow_removed={"overwrite_output_dir"},
        )


def test_fallback_uses_transformers5_dtype_keyword():
    source = inspect.getsource(fallback._emit_lora_adapter)
    assert '"dtype": torch.float32' in source
    assert '"torch_dtype":' not in source
