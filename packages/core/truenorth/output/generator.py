"""
Generate structured output from a completed profile.
Supports conditional sections based on collected data.
"""

from __future__ import annotations
from truenorth.core.graph_state import GraphState
from truenorth.core.yaml_loader import GoalConfig
from truenorth.llm.router import LLMRouter

SYSTEM = """You are an expert {domain} advisor generating a personalized plan.
Use ONLY the information provided. Be specific and actionable.
Format your response in clear markdown with sections."""

PROMPT_TEMPLATE = """
Generate a personalized plan based on this profile:

{profile}

Include these sections (skip any section if the data isn't relevant):
{sections}

Be specific, practical, and encouraging. Use the person's actual numbers."""


class OutputGenerator:
    def __init__(self, router: LLMRouter):
        self.router = router

    async def generate(self, state: GraphState, config: GoalConfig) -> dict:
        """Generate the final output for a completed session."""
        profile_text = "\n".join(
            f"- {k}: {v.value}"
            for k, v in state.profile.items()
        )

        # Build sections list based on conditional config
        sections = self._build_sections(state, config)
        sections_text = "\n".join(f"- {s}" for s in sections)

        domain = config.goal_id.replace("_", " ")
        system = SYSTEM.format(domain=domain)
        prompt = PROMPT_TEMPLATE.format(
            profile=profile_text,
            sections=sections_text,
        )

        response = await self.router.complete(
            task="output_generation", prompt=prompt, system=system,
            temperature=0.7, max_tokens=2000
        )

        return {
            "content": response.content,
            "format": config.output.format,
            "profile_snapshot": state.collected_fields,
            "sections_included": sections,
            "tokens_used": response.total_tokens,
            "model": response.model,
        }

    def _build_sections(self, state: GraphState, config: GoalConfig) -> list[str]:
        sections = []
        profile = state.collected_fields

        for section in config.output.sections:
            name = section.get("name", "")
            if section.get("always_include"):
                sections.append(name)
                continue

            include_if = section.get("include_if", "")
            if include_if:
                try:
                    if eval(include_if, {"profile": profile}):  # noqa: S307
                        sections.append(name)
                except Exception:
                    pass

        # If no sections configured, use smart defaults
        if not sections:
            sections = ["Summary", "Personalized Plan", "Key Recommendations", "Next Steps"]
            if profile.get("injuries") and profile["injuries"] not in ("none", "no", ""):
                sections.insert(2, "Injury Modifications")
            if profile.get("medical_conditions"):
                sections.insert(2, "Medical Adaptations")

        return sections
