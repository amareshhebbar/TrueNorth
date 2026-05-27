"""
Extract structured field values from free-text user messages.
Uses cheap/fast LLM. Returns values with confidence scores.
"""

from __future__ import annotations
import json
from truenorth.core.graph_state import FieldValue, GraphState
from truenorth.core.yaml_loader import FieldConfig, GoalConfig
from truenorth.intelligence.confidence_scorer import score_confidence
from truenorth.llm.router import LLMRouter

SYSTEM = """You are a precise data extraction engine.
Extract structured field values from user messages for a data collection form.
Be conservative — only extract values that are clearly stated or strongly implied.
Never invent or guess values not present in the text."""

PROMPT_TEMPLATE = """
Goal context: {goal_description}

Fields to extract (extract ONLY these):
{fields_spec}

Already collected (do NOT re-extract unless user is correcting):
{collected}

User message: "{message}"

For each field found in the message, extract the value.
Handle these formats:
- "I'm 25" → age: 25
- "seventy-five kilos" → weight: 75
- "मेरा वजन 75 किलो है" → weight: 75 (handle any language)
- "I think I'm around 170cm" → height: 170 (note: uncertain)

Respond ONLY in JSON:
{{
  "extracted": {{
    "<field_name>": {{
      "value": <typed_value>,
      "raw_text": "<exact phrase from message>",
      "confidence_hint": "certain | uncertain | inferred"
    }}
  }},
  "user_is_correcting": false
}}
If nothing to extract, return: {{"extracted": {{}}, "user_is_correcting": false}}
"""


class FieldExtractor:
    def __init__(self, router: LLMRouter):
        self.router = router

    async def extract(self, message: str, state: GraphState,
                      config: GoalConfig) -> dict[str, FieldValue]:
        """
        Extract field values from a user message.
        Returns dict of field_name → FieldValue.
        """
        target_fields = self._get_target_fields(state, config)
        if not target_fields:
            return {}

        fields_spec = self._format_fields_spec(target_fields)
        collected = {k: v.value for k, v in state.profile.items()}
        goal_desc = config.persona.base if hasattr(config.persona, 'base') else "data collection"

        prompt = PROMPT_TEMPLATE.format(
            goal_description=goal_desc,
            fields_spec=fields_spec,
            collected=json.dumps(collected, default=str) if collected else "nothing yet",
            message=message,
        )

        try:
            data, _ = await self.router.complete_json(
                task="field_extraction", prompt=prompt,
                system=SYSTEM, temperature=0.0, max_tokens=500
            )
        except Exception:
            return {}

        results = {}
        for field_name, info in data.get("extracted", {}).items():
            raw = info.get("raw_text", message)
            hint = info.get("confidence_hint", "certain")

            base_confidence = score_confidence(raw, info["value"])
            if hint == "uncertain":
                base_confidence *= 0.7
            elif hint == "inferred":
                base_confidence *= 0.8

            # Find privacy level from config
            privacy = self._get_field_privacy(field_name, config)

            results[field_name] = FieldValue(
                value=self._coerce_value(field_name, info["value"], config),
                confidence=base_confidence,
                source="user_stated" if hint == "certain" else "inferred",
                raw_text=raw,
                privacy_level=privacy,
            )

        return results

    def _get_target_fields(self, state: GraphState, config: GoalConfig) -> list[FieldConfig]:
        all_fields = config.required_fields + config.optional_fields
        # Return fields not yet collected with high confidence
        return [
            f for f in all_fields
            if f.name not in state.profile
            or state.profile[f.name].confidence < 0.6
        ]

    def _format_fields_spec(self, fields: list[FieldConfig]) -> str:
        lines = []
        for f in fields[:15]:  # Cap at 15 to keep prompt small
            spec = f"- {f.name} ({f.type})"
            if f.values:
                spec += f" — one of: {', '.join(f.values)}"
            if f.optional:
                spec += " [optional]"
            lines.append(spec)
        return "\n".join(lines)

    def _get_field_privacy(self, field_name: str, config: GoalConfig) -> str:
        for f in config.required_fields + config.optional_fields:
            if f.name == field_name:
                return f.privacy
        return "low"

    def _coerce_value(self, field_name: str, value: any, config: GoalConfig) -> any:
        for f in config.required_fields + config.optional_fields:
            if f.name == field_name:
                try:
                    if f.type == "integer":
                        return int(float(str(value).replace(",", "")))
                    if f.type == "float":
                        return float(str(value).replace(",", ""))
                    if f.type == "boolean":
                        if isinstance(value, bool):
                            return value
                        return str(value).lower() in ("true", "yes", "1")
                except (ValueError, TypeError):
                    pass
        return value
