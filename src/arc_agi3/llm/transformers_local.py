from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from arc_agi3.core.config import LLMConfig
from arc_agi3.core.types import Action, ExperimentProposal, LLMDirective, RankedAction, RuleHypothesis
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
        response_text = self._generate(prompt, thinking_mode=self.config.thinking_mode)
        bundle = self._parse_response(response_text, context.candidate_actions, context.available_experiments)
        if not bundle.ranked_actions and bundle.next_test is None and bundle.directive is None:
            repair_prompt = self._build_repair_prompt(prompt, response_text)
            repair_response = self._generate(repair_prompt, thinking_mode="off")
            repaired = self._parse_response(repair_response, context.candidate_actions, context.available_experiments)
            if repaired.ranked_actions or repaired.next_test is not None or repaired.directive is not None:
                repaired.raw_response = response_text + "\n\n[repair]\n" + repair_response
                return repaired
            response_text = response_text + "\n\n[repair]\n" + repair_response
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
  "directive": {
    "goal_key": "one exact experiment key from Available executable experiments or empty string",
    "summary": "short interpretation of the current goal",
    "preferred_action": "one exact action key from Candidate actions or empty string",
    "avoid_actions": ["exact action keys to avoid"],
    "commitment_steps": 1-8,
    "confidence": 0-1
  },
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
- If one experiment should guide the next several steps, set directive.goal_key to it.
- Use directive.preferred_action only when one exact immediate action is clearly best.
- Use directive.avoid_actions for exact actions that are loops, feedback-only, or strategically wrong now.
- Only mention exact action keys from Candidate actions.
- Only mention exact experiment keys from Available executable experiments.
- Return at most 3 ranked actions.
- Use score 100 for the best safe action, lower for weaker actions.
- Penalize unsafe/deadly/blocked/HUD-only/noop/loop actions.
- Prefer safe path progress or object-rule clicks.
- Keep the JSON short. Do not include hypotheses unless explicitly requested.
- If thinking is enabled, think briefly, then output one final JSON object.
- Do not output markdown fences or any prose outside the final JSON object.
"""
        return base + "\n" + output_format

    def _generate(self, prompt: str, thinking_mode: str | None = None) -> str:
        loaded = self._load()
        mode = thinking_mode or self.config.thinking_mode
        thinking_enabled = mode in {"brief", "full"}
        if mode == "brief":
            prompt = (
                prompt
                + "\n\nThink briefly before answering: use at most 6 short sentences or 160 words inside the thinking block, "
                + "then finish with the final JSON object."
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an ARC-AGI-3 ranker. "
                    "Use careful reasoning, but always finish with one short JSON object."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(loaded.tokenizer, "apply_chat_template"):
            try:
                text = loaded.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking_enabled,
                )
            except TypeError:
                text = loaded.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        model_inputs = loaded.tokenizer([text], return_tensors="pt")
        if torch is not None and hasattr(model_inputs, "to") and loaded.device != "auto":
            model_inputs = model_inputs.to(loaded.device)
        elif torch is not None and hasattr(loaded.model, "device"):
            model_inputs = model_inputs.to(loaded.model.device)
        generation_kwargs = {
            "max_new_tokens": self.config.thinking_max_new_tokens if thinking_enabled else self.config.max_new_tokens,
            "do_sample": thinking_enabled or self.config.temperature > 0,
            "temperature": 0.6 if thinking_enabled and self.config.temperature <= 0 else (self.config.temperature if self.config.temperature > 0 else None),
            "top_p": 0.95 if thinking_enabled else self.config.top_p,
            "top_k": 20 if thinking_enabled else None,
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
        return self._strip_thinking(decoded).strip()

    def _parse_response(
        self,
        response_text: str,
        candidate_actions: list[Action],
        available_experiments: list[ExperimentProposal] | None = None,
    ) -> LLMDecisionBundle:
        action_by_key = {action.key: action for action in candidate_actions}
        experiments = available_experiments or []
        parsed = self._extract_json_object(self._strip_thinking(response_text))
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
        directive = self._parse_directive(parsed, action_by_key, experiments)
        return LLMDecisionBundle(
            ranked_actions=ranked_actions,
            hypotheses=hypotheses,
            next_test=next_test,
            directive=directive,
        )

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

    def _parse_directive(
        self,
        parsed: dict[str, Any],
        action_by_key: dict[str, Action],
        available_experiments: list[ExperimentProposal],
    ) -> LLMDirective | None:
        raw = parsed.get("directive")
        if not isinstance(raw, dict):
            return None
        experiments = {proposal.key: proposal for proposal in available_experiments}
        goal_key = str(raw.get("goal_key", ""))
        if goal_key and goal_key not in experiments:
            goal_key = ""
        preferred_key = self._normalize_action_key(str(raw.get("preferred_action", "")), action_by_key)
        preferred_action = action_by_key.get(preferred_key)
        avoid_action_keys: list[str] = []
        for item in list(raw.get("avoid_actions", []) or [])[:6]:
            normalized = self._normalize_action_key(str(item), action_by_key)
            if normalized in action_by_key and normalized not in avoid_action_keys:
                avoid_action_keys.append(normalized)
        try:
            commitment_steps = int(raw.get("commitment_steps", 0))
        except Exception:
            commitment_steps = 0
        try:
            confidence = float(raw.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        if not goal_key and preferred_action is None and not avoid_action_keys:
            return None
        return LLMDirective(
            goal_key=goal_key,
            goal_summary=str(raw.get("summary", ""))[:240],
            preferred_action=preferred_action,
            avoid_action_keys=avoid_action_keys,
            commitment_steps=max(1, min(8, commitment_steps or 4)),
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
        return LLMDecisionBundle(next_test=next_test)

    def _salvage_next_test(
        self,
        response_text: str,
        available_experiments: list[ExperimentProposal],
    ) -> ExperimentProposal | None:
        if '"next_test"' not in response_text and "'next_test'" not in response_text:
            return None
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

    def _build_repair_prompt(self, original_prompt: str, invalid_response: str) -> str:
        compact_invalid = invalid_response.strip().replace("\n", " ")[:320]
        return (
            original_prompt
            + "\n\nUse the previous reasoning only as context. Your previous answer was invalid because it was not one complete JSON object.\n"
            + "Do not think again. Return only one minified JSON object now. No <think>. No prose. No markdown.\n"
            + f"Invalid previous answer prefix: {compact_invalid}"
        )

    def _strip_thinking(self, text: str) -> str:
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        if "<think>" in stripped.lower() and "</think>" not in stripped.lower():
            return ""
        return stripped

    def _normalize_action_key(self, raw_key: str, action_by_key: dict[str, Action]) -> str:
        if raw_key in action_by_key:
            return raw_key
        normalized = re.sub(r"\|x=(-?\d+),\s*(-?\d+)$", r"|x=\1,y=\2", raw_key)
        if normalized in action_by_key:
            return normalized
        return raw_key

    def _loose_action_variant(self, key: str) -> str:
        return re.sub(r"\|x=(-?\d+),y=(-?\d+)$", r"|x=\1,\2", key)
