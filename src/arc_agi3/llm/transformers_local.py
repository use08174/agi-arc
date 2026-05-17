from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from arc_agi3.core.config import LLMConfig
from arc_agi3.core.types import Action, RankedAction, RuleHypothesis
from arc_agi3.llm.prompting import PromptBuilder
from arc_agi3.llm.types import LLMContext, LLMDecisionBundle

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass(slots=True)
class _LoadedModel:
    tokenizer: Any
    model: Any
    device: str


class TransformersLocalProvider:
    """Local transformers-backed provider for Kaggle/offline notebooks.

    Intended use:
    - rank a small list of candidate actions
    - optionally propose compact rule hypotheses
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.prompt_builder = PromptBuilder()
        self._loaded: _LoadedModel | None = None

    def analyze(self, context: LLMContext) -> LLMDecisionBundle:
        prompt = self._build_prompt(context)
        response_text = self._generate(prompt)
        bundle = self._parse_response(response_text, context.candidate_actions)
        bundle.raw_response = response_text
        return bundle

    def close(self) -> None:
        self._loaded = None

    def _load(self) -> _LoadedModel:
        if self._loaded is not None:
            return self._loaded
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError(
                "transformers is not available. Install transformers/torch first."
            )

        model_path = self.config.model_path or self.config.model
        if not model_path:
            raise RuntimeError("LLM model_path is empty")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.config.dtype != "auto" and torch is not None:
            model_kwargs["torch_dtype"] = getattr(torch, self.config.dtype)
        else:
            model_kwargs["torch_dtype"] = "auto"

        if self.config.device == "auto":
            model_kwargs["device_map"] = "auto"
            device = "auto"
        else:
            device = self.config.device

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if self.config.device not in {"auto", ""} and hasattr(model, "to"):
            model = model.to(self.config.device)

        self._loaded = _LoadedModel(tokenizer=tokenizer, model=model, device=device)
        return self._loaded

    def _build_prompt(self, context: LLMContext) -> str:
        base = self.prompt_builder.build(context)
        output_format = """

Respond in strict JSON with this schema:
{
  "ranked_actions": [
    {"action": "ACTION_NAME_OR_ACTION|x=..,y=..", "score": 0.0, "reason": "short reason"}
  ],
  "hypotheses": [
    {"summary": "short rule hypothesis", "confidence": 0.0, "evidence": ["fact 1", "fact 2"]}
  ]
}

Rules:
- Only mention actions from the candidate list.
- Return at most 3 ranked actions.
- Be concise.
"""
        return base + output_format

    def _generate(self, prompt: str) -> str:
        loaded = self._load()
        messages = [
            {
                "role": "system",
                "content": "You are a careful reasoning assistant for ARC-AGI-3 action ranking.",
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(loaded.tokenizer, "apply_chat_template"):
            text = loaded.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)

        model_inputs = loaded.tokenizer([text], return_tensors="pt")
        if torch is not None and hasattr(model_inputs, "to") and loaded.device != "auto":
            model_inputs = model_inputs.to(loaded.device)
        elif torch is not None and hasattr(loaded.model, "device"):
            model_inputs = model_inputs.to(loaded.model.device)

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "temperature": self.config.temperature if self.config.temperature > 0 else None,
            "top_p": self.config.top_p,
            "pad_token_id": (
                loaded.tokenizer.pad_token_id
                if loaded.tokenizer.pad_token_id is not None
                else loaded.tokenizer.eos_token_id
            ),
            "eos_token_id": loaded.tokenizer.eos_token_id,
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        generated_ids = loaded.model.generate(**model_inputs, **generation_kwargs)
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return loaded.tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

    def _parse_response(
        self,
        response_text: str,
        candidate_actions: list[Action],
    ) -> LLMDecisionBundle:
        action_by_key = {action.key: action for action in candidate_actions}
        parsed = self._extract_json_object(response_text)
        if parsed is None:
            return self._fallback_parse(response_text, candidate_actions)

        ranked_actions: list[RankedAction] = []
        for item in parsed.get("ranked_actions", [])[: self.config.max_ranked_actions]:
            key = item.get("action", "")
            if key not in action_by_key:
                continue
            ranked_actions.append(
                RankedAction(
                    action=action_by_key[key],
                    score=float(item.get("score", 0.0)),
                    reason=str(item.get("reason", ""))[:200],
                )
            )

        hypotheses: list[RuleHypothesis] = []
        if self.config.include_hypotheses:
            for item in parsed.get("hypotheses", [])[:3]:
                hypotheses.append(
                    RuleHypothesis(
                        summary=str(item.get("summary", ""))[:240],
                        confidence=float(item.get("confidence", 0.0)),
                        evidence=[str(x)[:160] for x in item.get("evidence", [])[:4]],
                    )
                )
        return LLMDecisionBundle(
            ranked_actions=ranked_actions,
            hypotheses=hypotheses,
        )

    def _extract_json_object(self, response_text: str) -> dict[str, Any] | None:
        fenced_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        if fenced_match:
            parsed = self._try_json_loads(fenced_match.group(1))
            if parsed is not None:
                return parsed

        stripped = response_text.strip()
        parsed = self._try_json_loads(stripped)
        if parsed is not None:
            return parsed

        candidates = re.findall(r"\{.*?\}", response_text, re.DOTALL)
        for candidate in candidates:
            parsed = self._try_json_loads(candidate)
            if parsed is not None and "ranked_actions" in parsed:
                return parsed

        first = response_text.find("{")
        if first == -1:
            return None
        depth = 0
        start = None
        for idx, char in enumerate(response_text[first:], start=first):
            if char == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    parsed = self._try_json_loads(response_text[start : idx + 1])
                    if parsed is not None:
                        return parsed
        return None

    def _try_json_loads(self, raw: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _fallback_parse(
        self,
        response_text: str,
        candidate_actions: list[Action],
    ) -> LLMDecisionBundle:
        ranked_actions: list[RankedAction] = []
        seen: set[str] = set()
        for line in response_text.splitlines():
            for action in candidate_actions:
                if action.key in line and action.key not in seen:
                    seen.add(action.key)
                    ranked_actions.append(
                        RankedAction(action=action, score=float(len(candidate_actions) - len(ranked_actions)), reason="fallback parse")
                    )
                    break
            if len(ranked_actions) >= self.config.max_ranked_actions:
                break
        return LLMDecisionBundle(ranked_actions=ranked_actions)
