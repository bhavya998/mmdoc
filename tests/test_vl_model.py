"""Tests for mmdoc.vl_model - VLModel wrapper around Qwen3.5-VL.

Mocks the model loading so tests run without downloading the 2GB model.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from mmdoc.vl_model import VLModel


def _make_loaded_model() -> VLModel:
    """Create a VLModel with mocked _model and _processor (skips real loading)."""
    model = VLModel()

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 0

    mock_tensor = MagicMock()
    mock_tensor.to.return_value = mock_tensor
    mock_tensor.shape = [1, 3]  # so inputs["input_ids"].shape[1] works

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "<mocked_template>"
    mock_processor.return_value = {"input_ids": mock_tensor, "attention_mask": mock_tensor}
    mock_processor.tokenizer = mock_tokenizer
    mock_processor.decode.return_value = "mocked output"

    mock_param = MagicMock()
    mock_param.device = "cpu"

    mock_nn = MagicMock()
    mock_nn.parameters.return_value = iter([mock_param])
    mock_nn.generate.return_value = [[1, 2, 3, 4, 5]]

    model._processor = mock_processor
    model._model = mock_nn
    return model


class TestVLModelInit:
    def test_init_defaults(self) -> None:
        model = VLModel()
        assert model._model_id == "Qwen/Qwen3.5-0.8B"
        assert model._quantize is True
        assert model._max_tokens == 2048

    def test_init_custom(self) -> None:
        model = VLModel(model_id="test/model", quantize_4bit=False, max_tokens=512)
        assert model._model_id == "test/model"
        assert model._quantize is False
        assert model._max_tokens == 512

    def test_model_not_loaded_until_query(self) -> None:
        model = VLModel()
        assert model._model is None
        assert model._processor is None


class TestVLModelQuery:
    def test_query_returns_string(self) -> None:
        model = _make_loaded_model()
        img = Image.new("RGB", (10, 10))
        result = model.query(img, "What is this?")
        assert isinstance(result, str)
        assert result == "mocked output"

    def test_query_with_system_prompt(self) -> None:
        model = _make_loaded_model()
        img = Image.new("RGB", (10, 10))
        result = model.query(img, "Extract data", system="Be precise")
        assert isinstance(result, str)
        assert model._processor.apply_chat_template.called

    def test_query_calls_generate(self) -> None:
        model = _make_loaded_model()
        img = Image.new("RGB", (10, 10))
        model.query(img, "test prompt")
        assert model._model.generate.called

    def test_describe(self) -> None:
        model = _make_loaded_model()
        img = Image.new("RGB", (10, 10))
        result = model.describe(img)
        assert isinstance(result, str)

    def test_extract_json_parsed(self) -> None:
        model = _make_loaded_model()
        model._processor.decode.return_value = '{"key": "value"}'
        img = Image.new("RGB", (10, 10))
        result = model.extract_json(img, "Get key-value pairs")
        assert isinstance(result, dict)
        assert result == {"key": "value"}

    def test_extract_json_strips_fences(self) -> None:
        model = _make_loaded_model()
        model._processor.decode.return_value = '```json\n{"a": 1}\n```'
        img = Image.new("RGB", (10, 10))
        result = model.extract_json(img, "Get data")
        assert result == {"a": 1}

    def test_extract_json_invalid_raises(self) -> None:
        model = _make_loaded_model()
        model._processor.decode.return_value = "not json at all"
        img = Image.new("RGB", (10, 10))
        with pytest.raises(Exception):
            model.extract_json(img, "Get data")
