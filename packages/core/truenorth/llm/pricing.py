"""Token pricing per model (USD per 1M tokens)."""

PRICING = {
    # input_per_1m, output_per_1m
    "claude-sonnet-4-20250514":      (3.00, 15.00),
    "claude-3-5-sonnet-20241022":    (3.00, 15.00),
    "claude-3-5-haiku-20241022":     (0.80, 4.00),
    "gpt-4o":                        (2.50, 10.00),
    "gpt-4o-mini":                   (0.15, 0.60),
    "gemini-2.0-flash":              (0.10, 0.40),
    "gemini-2.0-flash-lite":         (0.075, 0.30),
}


def get_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING.get(model, (1.0, 4.0))
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000
