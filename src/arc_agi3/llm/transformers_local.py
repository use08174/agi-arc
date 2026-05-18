from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from arc_agi3.core.config import LLMConfig
from arc_agi3.core.types import Action, ExperimentProposal, RankedAction, RuleHypothesis
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
    """Local transformers-backed provider for Kaggle/offline notebooks."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.prompt_builder = PromptBuilder()
        self._loaded: _LoadedModel | None = None

    def analyze(self, context: LLMContext) -> LLMDecisionBundle:
        prompt = self._build_prompt(context)
        response_text = self._generate(prompt)
        bundle = self._parse_response(response_text, context.candidate_actions, context.available_experiments)
        bundle.raw_response = response_text
        return bundle

    def close(self) -> None:
        self._loaded = None

    def _load(self) -> _LoadedModel:
        if self._loaded is not None:
            return self._loaded
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers is not available. Install transformers/torch first.")
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
Respond in strict JSON only. Do not write markdown. Do not continue the chat.
Use this schema:
{
  "next_test": {
    "key": "one exact experiment key from Available executable experiments",
    "confidence": 0-1
  },
  "ranked_actions": [
    {"action": "one exact action key from Candidate actions", "score": 0-100}
  ]
}
Rules:
- Prefer selecting one next_test when a listed experiment can reduce uncertainty.
- Only mention exact action keys from Candidate actions.
- Only mention exact experiment keys from Available executable experiments.
- Return at most 3 ranked actions.
- Use score 100 for the best safe action, lower for weaker actions.
- Penalize unsafe/deadly/blocked/HUD-only/noop/loop actions.
- Prefer safe path progress or object-rule clicks.
- Keep the JSON short. Do not include hypotheses unless explicitly requested.
"""
        return base + "\n" + output_format

    def _generate(self, prompt: str) -> str:
        loaded = self._load()
        messages = [
            {"role": "system", "content": "You are a careful reasoning assistant for ARC-AGI-3 action ranking. Return JSON only."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(loaded.tokenizer, "apply_chat_template"):
            text = loaded.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            "pad_token_id": loaded.tokenizer.pad_token_id if loaded.tokenizer.pad_token_id is not None else loaded.tokenizer.eos_token_id,
            "eos_token_id": loaded.tokenizer.eos_token_id,
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        generated_ids = loaded.model.generate(**model_inputs, **generation_kwargs)
        trimmed_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        decoded = loaded.tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
        for marker in ["<|repo_name|>", "<|im_start|>", "<|im_end|>", "\nuser\n", "\nsystem\n", "\nassistant\n"]:
            idx = decoded.find(marker)
            if idx != -1:
                decoded = decoded[:idx]
        return decoded.strip()

    def _parse_response(
        self,
        response_text: str,
        candidate_actions: list[Action],
        available_experiments: list[ExperimentProposal] | None = None,
    ) -> LLMDecisionBundle:
        action_by_key = {action.key: action for action in candidate_actions}
        experiments = available_experiments or []
        parsed = self._extract_json_object(response_text)
        if parsed is None:
            return self._fallback_parse(response_text, candidate_actions, experiments)
        ranked_actions: list[RankedAction] = []
        for item in parsed.get("ranked_actions", [])[: self.config.max_ranked_actions]:
            key = self._normalize_action_key(str(item.get("action", "")), action_by_key)
            if key not in action_by_key:
                continue
            try:
                score = float(item.get("score", 0.0))
            except Exception:
                score = 0.0
            ranked_actions.append(RankedAction(action=action_by_key[key], score=score, reason=str(item.get("reason", ""))[:200]))
        hypotheses: list[RuleHypothesis] = []
        if self.config.include_hypotheses:
            for item in parsed.get("hypotheses", [])[:3]:
                hypotheses.append(RuleHypothesis(summary=str(item.get("summary", ""))[:240], confidence=float(item.get("confidence", 0.0)), evidence=[str(x)[:160] for x in item.get("evidence", [])[:4]]))
        next_test = self._parse_next_test(parsed, experiments)
        return LLMDecisionBundle(ranked_actions=ranked_actions, hypotheses=hypotheses, next_test=next_test)

    def _parse_next_test(
        self,
        parsed: dict[str, Any],
        available_experiments: list[ExperimentProposal],
    ) -> ExperimentProposal | None:
        raw = parsed.get("next_test")
        if not isinstance(raw, dict):
            return None
        key = str(raw.get("key", ""))
        by_key = {proposal.key: proposal for proposal in available_experiments}
        proposal = by_key.get(key)
        if proposal is None:
            return None
        try:
            confidence = float(raw.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        return ExperimentProposal(
            key=proposal.key,
            kind=proposal.kind,
            target=proposal.target,
            rationale=str(raw.get("reason", proposal.rationale))[:240],
            expected_if_true=proposal.expected_if_true,
            failure_signal=proposal.failure_signal,
            source="llm",
            confidence=max(0.0, min(1.0, confidence)),
        )

    def _extract_json_object(self, response_text: str) -> dict[str, Any] | None:
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            parsed = self._try_json_loads(fenced_match.group(1))
            if parsed is not None:
                return parsed
        stripped = response_text.strip()
        parsed = self._try_json_loads(stripped)
        if parsed is not None:
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
        available_experiments: list[ExperimentProposal] | None = None,
    ) -> LLMDecisionBundle:
        experiments = available_experiments or []
        next_test = self._salvage_next_test(response_text, experiments)
        ranked_actions: list[RankedAction] = []
        seen: set[str] = set()
        for line in response_text.splitlines():
            for action in candidate_actions:
                variants = {action.key, self._loose_action_variant(action.key)}
                if any(variant and variant in line for variant in variants) and action.key not in seen:
                    seen.add(action.key)
                    ranked_actions.append(RankedAction(action=action, score=1.0, reason="fallback parse"))
                    break
            if len(ranked_actions) >= self.config.max_ranked_actions:
                break
        return LLMDecisionBundle(ranked_actions=ranked_actions, next_test=next_test)

    def _salvage_next_test(
        self,
        response_text: str,
        available_experiments: list[ExperimentProposal],
    ) -> ExperimentProposal | None:
        by_key = {proposal.key: proposal for proposal in available_experiments}
        for key, proposal in by_key.items():
            if f'"key": "{key}"' in response_text or f"'key': '{key}'" in response_text:
                confidence_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', response_text)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.0
                return ExperimentProposal(
                    key=proposal.key,
                    kind=proposal.kind,
                    target=proposal.target,
                    rationale=proposal.rationale,
                    expected_if_true=proposal.expected_if_true,
                    failure_signal=proposal.failure_signal,
                    source="llm_salvaged",
                    confidence=max(0.0, min(1.0, confidence)),
                )
        return None

    def _normalize_action_key(self, raw_key: str, action_by_key: dict[str, Action]) -> str:
        if raw_key in action_by_key:
            return raw_key
        normalized = re.sub(r"\|x=(-?\d+),\s*(-?\d+)$", r"|x=\1,y=\2", raw_key)
        if normalized in action_by_key:
            return normalized
        return raw_key

    def _loose_action_variant(self, key: str) -> str:
        return re.sub(r"\|x=(-?\d+),y=(-?\d+)$", r"|x=\1,\2", key)
