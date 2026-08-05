"""
Built-in specialist agents — ready to use, no configuration needed.

Agents:
  ExtractionAgent  — extracts structured fields from user messages
  ValidationAgent  — validates extracted values against schema rules
  ResearchAgent    — searches web / calls MCP tools for supplementary info
  WriterAgent      — generates structured final output
"""

from truenorth.agents.specialist.extraction_agent  import ExtractionAgent
from truenorth.agents.specialist.validation_agent  import ValidationAgent
from truenorth.agents.specialist.research_agent    import ResearchAgent
from truenorth.agents.specialist.writer_agent      import WriterAgent

__all__ = [
    "ExtractionAgent",
    "ValidationAgent",
    "ResearchAgent",
    "WriterAgent",
]
