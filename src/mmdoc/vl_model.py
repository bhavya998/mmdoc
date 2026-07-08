"""Vision-Language Model — Qwen3.5-0.8B (unified multimodal).

0.8B params, reads images + text, outputs text/JSON. Fits in ~1GB VRAM in 4-bit.
Handles: structured extraction, QA, image description/summarization.
"""

from __future__ import annotations

import json
from typing import Any


class VLModel:
    """Qwen3.5-0.8B — unified vision-language, one model for all modes."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-0.8B",
        quantize_4bit: bool = True,
        max_tokens: int = 2048,
    ) -> None:
        self._model_id = model_id
        self._quantize = quantize_4bit
        self._max_tokens = max_tokens
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        has_cuda = torch.cuda.is_available()
        kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}

        if self._quantize and has_cuda:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print(f"[mmdoc] Loading {self._model_id} in 4-bit on GPU")
        elif has_cuda:
            kwargs["torch_dtype"] = torch.float16
            print(f"[mmdoc] Loading {self._model_id} fp16 on GPU")
        else:
            kwargs["torch_dtype"] = torch.float16
            print(f"[mmdoc] No CUDA — loading {self._model_id} fp16 on CPU")
            kwargs["device_map"] = "cpu"

        self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(self._model_id, **kwargs)
        if not has_cuda:
            self._model.to("cpu")
        self._model.eval()

    def query(
        self, image: Any, prompt: str, *, system: str | None = None, temperature: float = 0.0
    ) -> str:
        """Send an image + text prompt. Returns the model's text response."""
        import torch

        self._load()
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[image], return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self._processor.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        ).strip()

    def describe(self, image: Any, temperature: float = 0.0) -> str:
        """Describe what the image depicts in 1-2 sentences."""
        return self.query(
            image,
            "Describe what this image depicts or conveys in 1-2 concise sentences.",
            system="You are an image analyst. Be concise and factual.",
            temperature=temperature,
        )

    def extract_json(
        self, image: Any, schema_description: str, temperature: float = 0.0
    ) -> dict[str, Any]:
        """Extract structured data matching the described schema. Returns parsed JSON."""
        system = (
            "You are a document data extractor. Respond ONLY with valid JSON."
            " No markdown fences, no explanation."
        )
        prompt = f"Extract the following information as JSON:\n{schema_description}"
        raw = self.query(image, prompt, system=system, temperature=temperature)
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
