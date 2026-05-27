"""YAML goal config loader with validation and inheritance support."""

from __future__ import annotations
import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator


class FieldConfig(BaseModel):
    name: str
    type: str = "text"
    optional: bool = False
    privacy: str = "low"            # low | medium | high | critical
    temporal: bool = False
    max_age_days: int | None = None
    values: list[str] | None = None # for enum fields
    follow_ups: list[str] = []
    accept_from_document: bool = False
    accept_from_image: bool = False
    if_true: list["FieldConfig"] = []
    if_false: list["FieldConfig"] = []
    if_value_is: dict[str, list["FieldConfig"]] = {}
    validation: str | None = None   # Python expression string

    model_config = {"arbitrary_types_allowed": True}


class PersonaConfig(BaseModel):
    base: str = "helpful_assistant"
    adaptive: bool = False
    available_personas: list[dict] = []
    default_persona: str = "default"
    allow_user_switch: bool = False


class OutputConfig(BaseModel):
    format: str = "structured_report"      # structured_report | json | pdf | whatsapp | email
    sections: list[dict] = []
    low_confidence_handling: dict = {}
    formats: list[dict] = []


class ComplianceConfig(BaseModel):
    mode: str = "none"              # none | gdpr | hipaa | dpdp
    consent_required: bool = False
    retention_days: int = 365
    audit_trail: bool = True


class RateLimitConfig(BaseModel):
    sessions_per_day: int = 10
    sessions_per_hour: int = 3
    messages_per_session: int = 50


class CostConfig(BaseModel):
    max_tokens_per_session: int = 10000
    max_cost_per_session_usd: float = 0.05
    fallback_model: str = "gemini-2.0-flash-lite"
    on_budget_exceeded: str = "switch_to_cheaper_model"


class GoalConfig(BaseModel):
    goal_id: str
    abstract: bool = False
    extends: str | None = None
    persona: PersonaConfig = PersonaConfig()
    required_fields: list[FieldConfig] = []
    optional_fields: list[FieldConfig] = []
    output: OutputConfig = OutputConfig()
    compliance: ComplianceConfig = ComplianceConfig()
    rate_limits: RateLimitConfig = RateLimitConfig()
    cost_management: CostConfig = CostConfig()
    escalation: dict = {}
    webhooks: dict = {}
    language: dict = {"auto_detect": True, "respond_in": "user_language"}
    session: dict = {"persist": True, "ttl_hours": 168}
    ab_tests: list[dict] = []
    related_goals: list[dict] = []
    feedback: dict = {}

    @field_validator("goal_id")
    @classmethod
    def goal_id_must_be_slug(cls, v: str) -> str:
        import re
        if not re.match(r"^[a-z0-9_]+$", v):
            raise ValueError("goal_id must be lowercase alphanumeric with underscores")
        return v


class YamlLoader:
    """
    Loads and validates TrueNorth YAML goal configs.
    Handles inheritance (extends:) and abstract goals.
    """

    def __init__(self, goals_dir: Path | None = None):
        self.goals_dir = goals_dir or Path(".")
        self._cache: dict[str, GoalConfig] = {}

    def load(self, source: str | Path) -> GoalConfig:
        """Load a goal from a file path or YAML string."""
        if isinstance(source, str) and not source.endswith(".yaml"):
            raw = yaml.safe_load(source)
        else:
            path = Path(source)
            raw = yaml.safe_load(path.read_text())

        return self._parse(raw)

    def _parse(self, raw: dict) -> GoalConfig:
        if "extends" in raw:
            parent = self._load_parent(raw["extends"])
            raw = self._merge(parent, raw)

        config = GoalConfig(**raw)

        if config.abstract:
            raise ValueError(f"Goal '{config.goal_id}' is abstract and cannot be run directly")

        return config

    def _load_parent(self, goal_id: str) -> dict:
        candidates = list(self.goals_dir.rglob(f"{goal_id}.yaml"))
        if not candidates:
            raise FileNotFoundError(f"Parent goal '{goal_id}' not found in {self.goals_dir}")
        raw = yaml.safe_load(candidates[0].read_text())
        # Allow abstract parents
        raw.pop("abstract", None)
        return raw

    def _merge(self, parent: dict, child: dict) -> dict:
        """Deep merge child into parent. Child wins on conflicts."""
        merged = copy.deepcopy(parent)
        merged.pop("abstract", None)
        merged.pop("extends", None)

        for key, value in child.items():
            if key == "extends":
                continue
            if key == "required_fields" and key in merged:
                existing_names = {f["name"] for f in merged[key]}
                for field in value:
                    if field["name"] in existing_names:
                        # Override existing field
                        merged[key] = [f if f["name"] != field["name"] else field for f in merged[key]]
                    else:
                        merged[key].append(field)
            else:
                merged[key] = value

        return merged

    def get_all_fields(self, config: GoalConfig) -> list[FieldConfig]:
        """Flatten all fields including conditional sub-trees."""
        fields = []
        for f in config.required_fields + config.optional_fields:
            fields.append(f)
            fields.extend(self._flatten_conditional_fields(f))
        return fields

    def _flatten_conditional_fields(self, field: FieldConfig) -> list[FieldConfig]:
        sub = []
        for f in field.if_true:
            sub.append(f)
            sub.extend(self._flatten_conditional_fields(f))
        for f in field.if_false:
            sub.append(f)
        for branch_fields in field.if_value_is.values():
            for f in branch_fields:
                sub.append(f)
                sub.extend(self._flatten_conditional_fields(f))
        return sub
