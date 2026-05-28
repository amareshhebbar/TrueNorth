"""
truenorth/core/field_extractor.py

Extracts structured field values from unstructured user messages using an LLM.
Uses Gemini Flash by default (cheap, fast, good at JSON extraction).

For each turn it extracts ALL fields the message might contain — not just
the one that was asked. A user answering "age?" might also mention their goal.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from truenorth.llm.router import LLMRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ExtractedField:
    name:       str
    value:      Any
    confidence: float   
    raw_text:   str    

    def to_dict(self) -> dict:
        return {
            "name":       self.name,
            "value":      self.value,
            "confidence": round(self.confidence, 3),
            "raw_text":   self.raw_text,
        }


@dataclass
class ExtractionResult:
    fields:       List[ExtractedField]
    input_tokens: int = 0
    output_tokens: int = 0
    skipped:      List[str] = field(default_factory=list)  # fields that couldn't be extracted

    def to_dict(self) -> dict:
        return {
            "fields":   [f.to_dict() for f in self.fields],
            "count":    len(self.fields),
            "skipped":  self.skipped,
        }

    def as_map(self) -> Dict[str, Any]:
        return {f.name: f.value for f in self.fields}

    def confidence_map(self) -> Dict[str, float]:
        return {f.name: f.confidence for f in self.fields}


# ---------------------------------------------------------------------------
# FieldExtractor
# ---------------------------------------------------------------------------

class FieldExtractor:
    """
    Extracts one or more field values from a user message.

    Uses a structured JSON prompt so the LLM returns predictable output.
    Falls back to rule-based extraction when:
      - LLM is not configured
      - Response is not valid JSON
      - A field has a simple type (boolean, number) that regex can handle
    """

    _SYSTEM_PROMPT = """\
You are a precise data extraction assistant. Extract field values from user messages.
Return ONLY valid JSON. No markdown, no explanation, just the JSON object.
If a field's value is not present in the message, omit it from the output.
For confidence, use 1.0 if the value is stated explicitly, 0.7 if inferred, 0.4 if uncertain.
"""

    def __init__(self, router: Optional["LLMRouter"] = None):
        self._router = router

    async def extract(
        self,
        user_message:  str,
        fields_config: Dict[str, dict],
        context:       Optional[Dict[str, Any]] = None,  
        target_field:  Optional[str] = None,          
    ) -> ExtractionResult:
        """
        Extract field values from a user message.

        Args:
            user_message:   The raw user input this turn
            fields_config:  {field_name: field_spec} from goal YAML
            context:        Already collected fields (provides context for extraction)
            target_field:   The field the agent just asked about (extraction hint)

        Returns:
            ExtractionResult with all extracted fields
        """
        if not user_message or not user_message.strip():
            return ExtractionResult(fields=[])

        if not fields_config:
            return ExtractionResult(fields=[])

        # Try LLM extraction first
        if self._router is not None:
            try:
                return await self._llm_extract(
                    user_message, fields_config, context, target_field
                )
            except Exception as e:
                logger.warning("field_extractor LLM failed, falling back to rules: %s", e)

        return self._rule_extract(user_message, fields_config, target_field=target_field)

    def extract_sync(
        self,
        user_message:  str,
        fields_config: Dict[str, dict],
    ) -> ExtractionResult:
        """Synchronous rule-based extraction (for dry-run / testing)."""
        return self._rule_extract(user_message, fields_config)

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    async def _llm_extract(
        self,
        message:       str,
        fields_config: Dict[str, dict],
        context:       Optional[Dict[str, Any]],
        target_field:  Optional[str],
    ) -> ExtractionResult:
        from truenorth.llm.base import Message as LLMMessage
        from truenorth.llm.router import TASK_EXTRACT

        field_lines = []
        for name, cfg in fields_config.items():
            ftype = cfg.get("type", "text")
            desc  = cfg.get("description", name)
            allowed = cfg.get("allowed_values") or cfg.get("enum", [])
            hint  = f" (allowed: {allowed})" if allowed else ""
            field_lines.append(f'  "{name}": {{"type": "{ftype}", "description": "{desc}"{hint}}}')

        target_hint = f"\nThe agent just asked about: '{target_field}'." if target_field else ""
        context_hint = (
            f"\nAlready collected: {json.dumps(context, default=str)}"
            if context else ""
        )

        prompt = (
            f"Extract field values from this user message.\n\n"
            f"Fields to extract:\n{{\n"
            + ",\n".join(field_lines)
            + f"\n}}\n\n"
            f"User message: {message!r}"
            f"{target_hint}"
            f"{context_hint}\n\n"
            f"Return JSON: {{\"extractions\": ["
            f"{{\"name\": \"field_name\", \"value\": <value>, "
            f"\"confidence\": 0.0-1.0, \"raw_text\": \"exact text from message\"}}"
            f"]}}"
        )

        resp = await self._router.generate(
            task=TASK_EXTRACT,
            messages=[LLMMessage(role="user", content=prompt)],
            max_tokens=512,
            temperature=0.0,
        )

        result = self._parse_llm_response(
            resp.content,
            fields_config,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )

        if not result.fields:
            rule_result = self._rule_extract(message, fields_config, target_field=target_field)
            if rule_result.fields:
                return rule_result

        return result

    def _parse_llm_response(
        self,
        raw:           str,
        fields_config: Dict[str, dict],
        input_tokens:  int = 0,
        output_tokens: int = 0,
    ) -> ExtractionResult:
        """Parse the LLM's JSON response into ExtractedField objects."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("field_extractor: JSON parse failed: %s — raw: %r", e, raw[:200])
            return ExtractionResult(fields=[], input_tokens=input_tokens, output_tokens=output_tokens)

        extractions = data.get("extractions", [])
        if not isinstance(extractions, list):
            return ExtractionResult(fields=[], input_tokens=input_tokens, output_tokens=output_tokens)

        extracted = []
        for item in extractions:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            val  = item.get("value")
            conf = float(item.get("confidence", 0.7))
            raw_text = item.get("raw_text", "")

            if name not in fields_config or val is None:
                continue

            # Cast to declared type
            val = self._cast_value(val, fields_config[name].get("type", "text"))

            extracted.append(ExtractedField(
                name=name, value=val, confidence=conf, raw_text=raw_text,
            ))

        return ExtractionResult(
            fields=extracted,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # ------------------------------------------------------------------
    # Rule-based extraction (fallback / simple types)
    # ------------------------------------------------------------------

    def _rule_extract(
        self,
        message:       str,
        fields_config: Dict[str, dict],
        target_field:  Optional[str] = None,
    ) -> ExtractionResult:
        """
        Rule-based extraction. When target_field is provided (the field the agent
        just asked about), assign the user message directly to that field.
        This gives the dry-runner and mock-LLM mode reliable field collection.
        """
        msg = message.strip()
        if not msg or msg.lower() in ("[skip]", "skip"):
            return ExtractionResult(fields=[])

        extracted = []
        
        if target_field and target_field in fields_config:
            cfg   = fields_config[target_field]
            ftype = cfg.get("type", "text")
            val   = self._cast_to_type(msg, ftype)
            if val is not None:
                extracted.append(ExtractedField(
                    name=target_field, value=val,
                    confidence=0.75, raw_text=msg,
                ))
                return ExtractionResult(fields=extracted)

        # ── General scan when no target hint ────────────────────────────────────
        
        for name, cfg in fields_config.items():
            ftype = cfg.get("type", "text")

            if ftype in ("integer", "number", "float"):
                val = self._extract_number(msg)
                if val is not None:
                    extracted.append(ExtractedField(name=name, value=val, confidence=0.70, raw_text=msg))
                    break  # only assign a number to the FIRST numeric field

            elif ftype == "boolean":
                val = self._extract_boolean(msg)
                if val is not None:
                    extracted.append(ExtractedField(name=name, value=val, confidence=0.80, raw_text=msg))
                    break

            elif ftype == "email":
                m = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", msg)
                if m:
                    extracted.append(ExtractedField(name=name, value=m.group(), confidence=0.95, raw_text=m.group()))
                    break

        return ExtractionResult(fields=extracted)

    def _cast_to_type(self, text: str, ftype: str) -> Optional[Any]:
        """Try to cast raw text to the declared field type. Returns None on failure."""
        text = text.strip()
        if not text:
            return None
        if ftype in ("integer", "int"):
            n = self._extract_number(text)
            return int(n) if n is not None else text  
        if ftype in ("number", "float"):
            n = self._extract_number(text)
            return float(n) if n is not None else text
        if ftype == "boolean":
            b = self._extract_boolean(text)
            return b if b is not None else text
        return text  # text, email, etc — return as-is

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_boolean(text: str) -> Optional[bool]:
        t = text.strip().lower()
        if re.match(r"\b(yes|yeah|yep|true|1|absolutely|definitely|sure|correct)\b", t):
            return True
        if re.match(r"\b(no|nope|nah|false|0|never|not|incorrect)\b", t):
            return False
        return None

    @staticmethod
    def _cast_value(value: Any, ftype: str) -> Any:
        """Cast extracted value to the declared field type."""
        try:
            if ftype in ("integer", "int"):
                return int(float(str(value).replace(",", "")))
            if ftype in ("number", "float"):
                return float(str(value).replace(",", ""))
            if ftype == "boolean":
                if isinstance(value, bool):
                    return value
                return str(value).lower() in ("true", "yes", "1", "y")
            return value
        except (ValueError, TypeError):
            return value