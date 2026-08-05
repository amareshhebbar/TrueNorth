"""
Canonical token pricing table for all supported LLM providers.

Rules:
  - This is the SINGLE SOURCE OF TRUTH for model prices.
  - cost_tracker.py and router.py both import from here.
  - Local / on-device models always return 0.00.
  - Unknown models fall back to FALLBACK (conservative over-estimate).
  - Provider sections are clearly separated for easy updates.
  - Each entry is (input_price_per_1M, output_price_per_1M) in USD.

To add a new model: add a row to PRICING. Nothing else changes.

Used by:
  - truenorth.llm.cost_tracker._compute_cost()
  - truenorth.llm.router._compute_cost()
  - truenorth.cli.main  (truenorth cost --session SESSION_ID)
  - truenorth.cli.main  (truenorth pricing --provider anthropic)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

PRICING: Dict[str, Tuple[float, float]] = {

    "claude-opus-4-20250514":     (15.00, 75.00),
    "claude-opus-4-7":            (15.00, 75.00),
    "claude-opus-4-8":            (15.00, 75.00),
    "claude-sonnet-4-20250514":   ( 3.00, 15.00),
    "claude-haiku-4-5-20251001":  ( 0.80,  4.00),
    "claude-haiku-3-5":           ( 0.80,  4.00),
    "claude-3-5-sonnet-20241022": ( 3.00, 15.00),
    "claude-3-5-haiku-20241022":  ( 0.80,  4.00),
    "claude-3-opus-20240229":     (15.00, 75.00),
    "claude-3-sonnet-20240229":   ( 3.00, 15.00),
    "claude-3-haiku-20240307":    ( 0.25,  1.25),

    "gemini-3.5-flash":           ( 0.075, 0.30),
    "gemini-3.5-flash-8b":        ( 0.0375, 0.15),
    "gemini-1.5-pro":             ( 3.50, 10.50),
    "gemini-2.0-flash":           ( 0.10,  0.40),
    "gemini-2.0-flash-lite":      ( 0.075, 0.30),
    "gemini-2.0-pro":             ( 3.50, 10.50),
    "gemini-exp-1206":            ( 0.00,  0.00),

    "gpt-4o":                     ( 2.50, 10.00),
    "gpt-4o-mini":                ( 0.15,  0.60),
    "gpt-4o-mini-2024-07-18":     ( 0.15,  0.60),
    "gpt-4-turbo":                (10.00, 30.00),
    "gpt-4-turbo-preview":        (10.00, 30.00),
    "gpt-4":                      (30.00, 60.00),
    "gpt-3.5-turbo":              ( 0.50,  1.50),
    "o1":                         (15.00, 60.00),
    "o1-mini":                    ( 3.00, 12.00),
    "o1-preview":                 (15.00, 60.00),
    "o3":                         (10.00, 40.00),
    "o3-mini":                    ( 1.10,  4.40),
    "o4-mini":                    ( 1.10,  4.40),

    "command-r":                  ( 0.50,  1.50),
    "command-r-plus":             ( 3.00, 15.00),
    "command-r-08-2024":          ( 0.50,  1.50),

    "llama-3.1-70b-versatile":    ( 0.59,  0.79),
    "llama-3.1-8b-instant":       ( 0.05,  0.08),
    "llama-3.3-70b-versatile":    ( 0.59,  0.79),
    "mixtral-8x7b-32768":         ( 0.24,  0.24),

    "meta-llama/Llama-3.1-70B-Instruct-Turbo": (0.88, 0.88),
    "meta-llama/Llama-3.1-8B-Instruct-Turbo":  (0.18, 0.18),
    "mistralai/Mixtral-8x7B-Instruct-v0.1":    (0.60, 0.60),

    "mistral-large-latest":       ( 3.00,  9.00),
    "mistral-small-latest":       ( 0.20,  0.60),
    "mistral-nemo":               ( 0.15,  0.15),
    "open-mistral-7b":            ( 0.25,  0.25),

    "ollama":                     ( 0.00,  0.00),
    "local":                      ( 0.00,  0.00),
    "llama-cpp":                  ( 0.00,  0.00),
    "lmstudio":                   ( 0.00,  0.00),

    "apple/on-device-3b":         ( 0.00,  0.00),
    "gemini-nano":                ( 0.00,  0.00),
    "gemini-nano-2":              ( 0.00,  0.00),
    "on-device":                  ( 0.00,  0.00),
}

FALLBACK: Tuple[float, float] = (1.00, 5.00)

PROVIDERS: Dict[str, List[str]] = {
    "anthropic": [
        "claude-opus-4-20250514", "claude-opus-4-7", "claude-opus-4-8",
        "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
    ],
    "google": [
        "gemini-3.5-flash", "gemini-3.5-flash-8b", "gemini-1.5-pro",
        "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.0-pro",
    ],
    "openai": [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
        "o1", "o1-mini", "o3", "o3-mini", "o4-mini",
    ],
    "cohere":   ["command-r", "command-r-plus"],
    "groq":     ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "together": ["meta-llama/Llama-3.1-70B-Instruct-Turbo", "meta-llama/Llama-3.1-8B-Instruct-Turbo"],
    "mistral":  ["mistral-large-latest", "mistral-small-latest", "mistral-nemo"],
    "local":    ["ollama", "local", "llama-cpp", "lmstudio",
                 "apple/on-device-3b", "gemini-nano", "on-device"],
}

_FREE_PREFIXES = ("ollama", "local", "apple/", "gemini-nano", "on-device",
                  "mobile", "llama-cpp", "lmstudio")

def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Compute USD cost for one LLM call.
    Returns 0.0 for local/on-device models.
    Falls back to FALLBACK pricing for unknown models.
    """
    if not model:
        return 0.0
    base = model.split(":")[0].strip()
    if any(base.startswith(pfx) for pfx in _FREE_PREFIXES):
        return 0.0
    pin, pout = PRICING.get(base, FALLBACK)
    return round((input_tokens * pin + output_tokens * pout) / 1_000_000, 8)

def get_model_price(model: str) -> Tuple[float, float]:
    """Return (input_per_1M, output_per_1M) for a model."""
    base = model.split(":")[0].strip()
    if any(base.startswith(pfx) for pfx in _FREE_PREFIXES):
        return (0.0, 0.0)
    return PRICING.get(base, FALLBACK)

def get_provider(model: str) -> str:
    """Return the provider name for a model string."""
    m = model.split(":")[0].strip().lower()
    for provider, models in PROVIDERS.items():
        if any(m == pm.lower() or m.startswith(pm.lower().split("/")[0]) for pm in models):
            return provider
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    if m.startswith("gemini"):
        return "google"
    return "unknown"

def list_models(provider: Optional[str] = None) -> List[dict]:
    """
    Return a list of model pricing dicts.
    Optionally filter by provider name.
    """
    result = []
    for model, (pin, pout) in PRICING.items():
        prov = get_provider(model)
        if provider and prov.lower() != provider.lower():
            continue
        result.append({
            "model":            model,
            "provider":         prov,
            "input_per_1m":     pin,
            "output_per_1m":    pout,
            "cost_1k_tokens":   round((pin * 500 + pout * 500) / 1_000_000, 6),
            "free":             pin == 0.0 and pout == 0.0,
        })
    return sorted(result, key=lambda x: (x["provider"], x["model"]))

def cheapest_for_task(task: str, exclude_local: bool = False) -> Optional[str]:
    """
    Return the cheapest model appropriate for a given task type.
    Tasks: "extract", "converse", "output", "verify"
    """
    requires_top = task in ("output", "verify")

    candidates = []
    for model, (pin, pout) in PRICING.items():
        if exclude_local and (pin == 0.0 and pout == 0.0):
            continue
        if requires_top and pin < 1.0:
            continue
        avg = (pin + pout) / 2
        candidates.append((avg, model))

    if not candidates:
        return None
    return sorted(candidates)[0][1]
