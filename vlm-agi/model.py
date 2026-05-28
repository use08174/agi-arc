from __future__ import annotations

import gc
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from config import AppConfig
from prompts import POLICY_SYSTEM_PROMPT, SCENE_UNDERSTANDING_SYSTEM_PROMPT


def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if (path / "config.json").exists():
        return path

    hits = [candidate.parent for candidate in path.rglob("config.json")] if path.exists() else []
    if len(hits) == 1:
        return hits[0]
    if not path.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {path}")
    raise FileNotFoundError(
        f"Could not resolve model dir from {path}. Found {len(hits)} config.json files: {hits[:10]}"
    )


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def make_symlink_tree(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        target = dst_dir / item.name
        if target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(item, target_is_directory=item.is_dir())
        except Exception:
            if item.is_file() and item.stat().st_size < 20_000_000:
                shutil.copy2(item, target)


def patched_qwen3vl_processor_dir(model_path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="qwen3vl_processor_patch_"))
    make_symlink_tree(model_path, tmp)

    processor_config = _read_json(model_path / "processor_config.json", {})
    processor_config["processor_class"] = "Qwen3VLProcessor"
    processor_config.setdefault("image_processor_type", "Qwen3VLImageProcessor")
    processor_config["video_processor_type"] = "Qwen3VLVideoProcessor"
    _write_json(tmp / "processor_config.json", processor_config)

    preprocessor_config = _read_json(model_path / "preprocessor_config.json", {})
    preprocessor_config.setdefault("image_processor_type", "Qwen3VLImageProcessor")
    preprocessor_config["video_processor_type"] = "Qwen3VLVideoProcessor"
    _write_json(tmp / "preprocessor_config.json", preprocessor_config)

    video_config = _read_json(model_path / "video_preprocessor_config.json", None)
    if not video_config:
        video_config = dict(preprocessor_config)
    video_config["video_processor_type"] = "Qwen3VLVideoProcessor"
    video_config.pop("image_processor_type", None)
    _write_json(tmp / "video_preprocessor_config.json", video_config)

    print("patched processor dir:", tmp)
    return tmp


def extract_json_object(text: str) -> dict[str, Any]:
    raw_text = (text or "").strip()
    for candidate in (
        raw_text,
        re.sub(r"^```(?:json)?\s*", "", raw_text),
    ):
        candidate = re.sub(r"\s*```$", "", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            continue

    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON object found in VLM output:\n{raw_text}")


class VLMManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.processor: Any | None = None
        self.model: Any | None = None
        self.model_path: str | None = None
        self.processor_path: str | None = None

    def load(self) -> tuple[Any, Any]:
        from transformers import AutoConfig, AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration
        except Exception:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLForConditionalGeneration,
            )

        model_path = resolve_model_path(self.config.model_path)
        local_only = not self.config.allow_download
        print("resolved model_path =", model_path)

        config = AutoConfig.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=local_only,
        )
        print(
            "model_type:",
            getattr(config, "model_type", None),
            "architectures:",
            getattr(config, "architectures", None),
        )

        processor_path: Path = model_path
        try:
            processor = AutoProcessor.from_pretrained(
                str(processor_path),
                trust_remote_code=True,
                local_files_only=local_only,
                use_fast=False,
            )
            print("AutoProcessor loaded from original model path")
        except Exception as exc:
            print("AutoProcessor original load failed:", repr(exc))
            print("Trying patched processor config directory...")
            processor_path = patched_qwen3vl_processor_dir(model_path)
            for module_name in list(sys.modules):
                if module_name.startswith(
                    "transformers.models.qwen3_vl.processing_qwen3_vl"
                ):
                    sys.modules.pop(module_name, None)
            processor = AutoProcessor.from_pretrained(
                str(processor_path),
                trust_remote_code=True,
                local_files_only=local_only,
                use_fast=False,
            )
            print("AutoProcessor loaded from patched processor path")

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": local_only,
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"

        model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_path), **kwargs)
        model.eval()

        self.processor = processor
        self.model = model
        self.model_path = str(model_path)
        self.processor_path = str(processor_path)
        print("loaded processor:", type(processor).__name__)
        print("loaded model:", type(model).__name__)
        return processor, model

    def get(self, *, force_reload: bool | None = None) -> tuple[Any, Any]:
        if force_reload is None:
            force_reload = self.config.force_reload_model
        if not force_reload and self.processor is not None and self.model is not None:
            print("using cached VLM:", self.model_path)
            return self.processor, self.model
        return self.load()

    @staticmethod
    def model_device(model: Any) -> torch.device:
        try:
            return next(model.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def move_inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
        out = {}
        for key, value in inputs.items():
            out[key] = value.to(device) if hasattr(value, "to") else value
        return out

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        if isinstance(exc, torch.OutOfMemoryError):
            return True
        return "out of memory" in str(exc).lower()

    @staticmethod
    def _release_cuda_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    @staticmethod
    def _downscale_image(image: Any, factor: int) -> Any:
        if factor <= 1:
            return image
        if hasattr(image, "copy") and hasattr(image, "shape"):
            return image[::factor, ::factor].copy()
        return image

    def _fallback_generation_attempts(
        self,
        images: list[Any],
        max_tokens: int,
    ) -> list[tuple[str, list[Any], int]]:
        attempts: list[tuple[str, list[Any], int]] = [("default", images, max_tokens)]
        if images:
            attempts.append(
                (
                    "downscaled_x2",
                    [self._downscale_image(image, 2) for image in images],
                    min(max_tokens, 256),
                )
            )
        if len(images) >= 2:
            attempts.append(
                (
                    "last_image_only",
                    [self._downscale_image(images[-1], 2)],
                    min(max_tokens, 192),
                )
            )
        elif images:
            attempts.append(
                (
                    "single_image_smaller",
                    [self._downscale_image(images[0], 3)],
                    min(max_tokens, 192),
                )
            )
        return attempts

    def prepare_inputs(self, images: list[Any], prompt: str, *, processor: Any | None = None) -> dict[str, Any]:
        if processor is None:
            processor, _ = self.get()
        full_text = POLICY_SYSTEM_PROMPT + "\n\n" + prompt
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images]
                + [{"type": "text", "text": full_text}],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return processor(text=[text], images=images, return_tensors="pt")

    def generate_local_vlm(
        self,
        *,
        images: list[Any],
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str:
        processor, model = self.get()
        max_tokens = int(max_new_tokens or self.config.max_new_tokens)
        attempts = self._fallback_generation_attempts(images, max_tokens)
        last_exc: Exception | None = None

        for label, attempt_images, attempt_tokens in attempts:
            try:
                inputs = self.prepare_inputs(attempt_images, prompt, processor=processor)
                inputs = self.move_inputs_to_device(inputs, self.model_device(model))
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=attempt_tokens,
                        do_sample=False,
                    )
                input_len = inputs["input_ids"].shape[-1]
                generated_ids = generated_ids[:, input_len:]
                if label != "default":
                    print(
                        f"VLM retry succeeded with {label}: "
                        f"images={len(attempt_images)} max_new_tokens={attempt_tokens}"
                    )
                return processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
            except Exception as exc:
                last_exc = exc
                if not self._is_oom_error(exc):
                    raise
                self._release_cuda_memory()
                print(
                    f"VLM OOM on {label}; retrying with smaller inputs "
                    f"(images={len(attempt_images)} max_new_tokens={attempt_tokens})"
                )
                continue

        assert last_exc is not None
        raise last_exc

    def generate_scene_understanding(
        self,
        *,
        image: Any,
        prompt: str,
        max_new_tokens: int = 500,
    ) -> str:
        processor, model = self.get()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": SCENE_UNDERSTANDING_SYSTEM_PROMPT + "\n\n" + prompt,
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = self.move_inputs_to_device(inputs, self.model_device(model))
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = generated_ids[:, input_len:]
        return processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
