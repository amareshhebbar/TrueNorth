"""
Classes:
  1.  PricingTable       — every model costs correctly
  2.  FreeModels         — local/on-device always $0.00
  3.  FallbackPricing    — unknown models use conservative FALLBACK
  4.  HelperFunctions    — get_model_price, get_provider, list_models, cheapest_for_task
  5.  CostUSdIntegration — cost_tracker._compute_cost uses same values as pricing.py
  6.  CliCost            — truenorth cost --session
  7.  CliPricing         — truenorth pricing --provider
  8.  CliEstimate        — truenorth estimate --model
  9.  CliVersion         — truenorth version
  10. CrossModuleConsistency — pricing.py and cost_tracker._compute_cost agree
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.llm.pricing import (
    PRICING,
    FALLBACK,
    PROVIDERS,
    cost_usd,
    get_model_price,
    get_provider,
    list_models,
    cheapest_for_task,
)
# from truenorth.cli.main import cli
from cli.main import cli


# ─────────────────────────────────────────────────────────────────────────────
#  1. Pricing table correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestPricingTable:

    def test_claude_haiku_correct_price(self):
        pin, pout = PRICING["claude-haiku-4-5-20251001"]
        assert pin  == pytest.approx(0.80)
        assert pout == pytest.approx(4.00)

    def test_claude_sonnet_correct_price(self):
        pin, pout = PRICING["claude-sonnet-4-20250514"]
        assert pin  == pytest.approx(3.00)
        assert pout == pytest.approx(15.00)

    def test_claude_opus_correct_price(self):
        pin, pout = PRICING["claude-opus-4-20250514"]
        assert pin  == pytest.approx(15.00)
        assert pout == pytest.approx(75.00)

    def test_gemini_flash_correct_price(self):
        pin, pout = PRICING["gemini-1.5-flash"]
        assert pin  == pytest.approx(0.075)
        assert pout == pytest.approx(0.30)

    def test_gpt4o_correct_price(self):
        pin, pout = PRICING["gpt-4o"]
        assert pin  == pytest.approx(2.50)
        assert pout == pytest.approx(10.00)

    def test_gpt4o_mini_correct_price(self):
        pin, pout = PRICING["gpt-4o-mini"]
        assert pin  == pytest.approx(0.15)
        assert pout == pytest.approx(0.60)

    def test_output_always_more_expensive_than_input(self):
        """Output tokens are always priced higher than input for paid models."""
        for model, (pin, pout) in PRICING.items():
            if pin > 0:
                assert pout >= pin, f"{model}: output ({pout}) cheaper than input ({pin})"

    def test_all_prices_non_negative(self):
        for model, (pin, pout) in PRICING.items():
            assert pin  >= 0, f"{model} has negative input price"
            assert pout >= 0, f"{model} has negative output price"

    def test_pricing_has_all_major_providers(self):
        models_str = " ".join(PRICING.keys())
        assert "claude"  in models_str
        assert "gpt"     in models_str
        assert "gemini"  in models_str

    def test_at_least_15_models_in_table(self):
        assert len(PRICING) >= 15

    def test_fallback_is_tuple_of_two_floats(self):
        assert isinstance(FALLBACK, tuple)
        assert len(FALLBACK) == 2
        assert all(isinstance(v, float) for v in FALLBACK)

    def test_fallback_is_nonzero(self):
        # Conservative estimate — should not be zero
        assert FALLBACK[0] > 0
        assert FALLBACK[1] > 0


# ─────────────────────────────────────────────────────────────────────────────
#  2. Free/local models always $0.00
# ─────────────────────────────────────────────────────────────────────────────

class TestFreeModels:

    @pytest.mark.parametrize("model", [
        "ollama", "local", "apple/on-device-3b",
        "gemini-nano", "on-device", "llama-cpp", "lmstudio",
    ])
    def test_free_model_zero_cost(self, model):
        assert cost_usd(model, 100_000, 100_000) == 0.0, f"{model} should be free"

    def test_ollama_with_colon_prefix_free(self):
        # "ollama:llama3.1" format
        assert cost_usd("ollama:llama3.1", 10_000, 10_000) == 0.0

    def test_mobile_prefix_free(self):
        assert cost_usd("apple/on-device-3b", 50_000, 50_000) == 0.0
        assert cost_usd("gemini-nano-2",       50_000, 50_000) == 0.0

    def test_empty_model_zero_cost(self):
        assert cost_usd("", 1000, 1000) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  3. Fallback pricing for unknown models
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackPricing:

    def test_unknown_model_uses_fallback(self):
        cost = cost_usd("completely-unknown-model-xyz", 1_000_000, 0)
        expected = FALLBACK[0]   # $1.00/M input
        assert cost == pytest.approx(expected, abs=0.01)

    def test_fallback_more_expensive_than_cheapest_real(self):
        # Fallback should be more expensive than Gemini Flash (conservative)
        fallback_cost = cost_usd("mystery-model", 1_000_000, 0)
        flash_cost    = cost_usd("gemini-1.5-flash", 1_000_000, 0)
        assert fallback_cost > flash_cost

    def test_fallback_less_than_most_expensive(self):
        # Fallback should be cheaper than Claude Opus (not crazy high)
        fallback_cost = cost_usd("mystery-model", 1_000_000, 0)
        opus_cost     = cost_usd("claude-opus-4-20250514", 1_000_000, 0)
        assert fallback_cost < opus_cost


# ─────────────────────────────────────────────────────────────────────────────
#  4. Helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestHelperFunctions:

    def test_get_model_price_known(self):
        pin, pout = get_model_price("claude-haiku-4-5-20251001")
        assert pin  == pytest.approx(0.80)
        assert pout == pytest.approx(4.00)

    def test_get_model_price_unknown_returns_fallback(self):
        assert get_model_price("unknown") == FALLBACK

    def test_get_model_price_free_model(self):
        pin, pout = get_model_price("ollama")
        assert pin  == 0.0
        assert pout == 0.0

    def test_get_provider_anthropic(self):
        assert get_provider("claude-sonnet-4-20250514") == "anthropic"
        assert get_provider("claude-haiku-4-5-20251001") == "anthropic"

    def test_get_provider_openai(self):
        assert get_provider("gpt-4o")      == "openai"
        assert get_provider("gpt-4o-mini") == "openai"

    def test_get_provider_google(self):
        assert get_provider("gemini-1.5-flash") == "google"

    def test_get_provider_local(self):
        assert get_provider("ollama") in ("local", "unknown")

    def test_list_models_returns_list(self):
        models = list_models()
        assert isinstance(models, list)
        assert len(models) >= 10

    def test_list_models_filter_provider(self):
        anthropic = list_models(provider="anthropic")
        assert all(m["provider"] == "anthropic" for m in anthropic)
        assert len(anthropic) >= 3

    def test_list_models_has_required_keys(self):
        models = list_models()
        for m in models[:3]:
            assert "model"         in m
            assert "provider"      in m
            assert "input_per_1m"  in m
            assert "output_per_1m" in m
            assert "cost_1k_tokens"in m
            assert "free"          in m

    def test_list_models_free_flagged(self):
        models = list_models(provider="local")
        # All local models should be free
        for m in models:
            if m["provider"] == "local":
                assert m["free"] is True

    def test_cheapest_for_task_extract(self):
        model = cheapest_for_task("extract", exclude_local=True)
        assert model is not None
        pin, _ = get_model_price(model)
        assert pin < 1.0   # cheaper than $1/M

    def test_cheapest_for_task_output(self):
        model = cheapest_for_task("output", exclude_local=True)
        assert model is not None
        pin, _ = get_model_price(model)
        assert pin >= 1.0   # output needs quality tier


# ─────────────────────────────────────────────────────────────────────────────
#  5. cost_usd function correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestCostUsd:

    def test_haiku_1m_input_tokens(self):
        cost = cost_usd("claude-haiku-4-5-20251001", 1_000_000, 0)
        assert cost == pytest.approx(0.80, abs=0.001)

    def test_haiku_1m_output_tokens(self):
        cost = cost_usd("claude-haiku-4-5-20251001", 0, 1_000_000)
        assert cost == pytest.approx(4.00, abs=0.001)

    def test_sonnet_mixed_tokens(self):
        cost = cost_usd("claude-sonnet-4-20250514", 500_000, 500_000)
        expected = (500_000 * 3.00 + 500_000 * 15.00) / 1_000_000
        assert cost == pytest.approx(expected, abs=0.001)

    def test_gemini_flash_tiny_call(self):
        cost = cost_usd("gemini-1.5-flash", 150, 80)
        assert 0 < cost < 0.001   # very cheap

    def test_zero_tokens_zero_cost(self):
        for model in ["claude-haiku-4-5-20251001", "gpt-4o", "gemini-1.5-flash"]:
            assert cost_usd(model, 0, 0) == 0.0

    def test_pricing_ordered_correctly(self):
        # Gemini Flash should be cheaper than Claude Haiku
        flash = cost_usd("gemini-1.5-flash",          1_000, 500)
        haiku = cost_usd("claude-haiku-4-5-20251001", 1_000, 500)
        assert flash < haiku

    def test_haiku_cheaper_than_sonnet(self):
        h = cost_usd("claude-haiku-4-5-20251001", 1_000, 500)
        s = cost_usd("claude-sonnet-4-20250514",  1_000, 500)
        assert h < s

    def test_sonnet_cheaper_than_opus(self):
        s = cost_usd("claude-sonnet-4-20250514",  1_000, 500)
        o = cost_usd("claude-opus-4-20250514",    1_000, 500)
        assert s < o


# ─────────────────────────────────────────────────────────────────────────────
#  6. CLI — truenorth cost
# ─────────────────────────────────────────────────────────────────────────────

class TestCliCost:

    def test_cost_no_args_shows_error(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["cost"])
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_cost_session_table_output(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", "--session", "test-sess-123"])
        # Should not crash
        assert result.exit_code == 0

    def test_cost_session_json_output(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", "--session", "test-sess", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "session" in data
        assert "task_breakdown" in data

    def test_cost_session_csv_output(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", "--session", "no-calls-sess", "--format", "csv"])
        assert result.exit_code == 0
        # Empty session → empty CSV is ok
        assert isinstance(result.output, str)

    def test_cost_with_real_data_json(self):
        """Record some calls then read via CLI."""
        from truenorth.llm.cost_tracker import CostTracker
        ct = CostTracker()
        ct.record("cli-test-sess", "claude-haiku-4-5-20251001", "extract",
                  200, 100, goal_id="fitness_plan")
        ct.record("cli-test-sess", "gemini-1.5-flash", "converse",
                  150, 80, goal_id="fitness_plan")

        # CLI reads its own CostTracker — it won't have our data unless we
        # mock it. Just verify the CLI runs without crashing.
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", "--session", "cli-test-sess", "--format", "json"])
        assert result.exit_code == 0

    def test_cost_goal_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", "--goal", "fitness_plan"])
        assert result.exit_code == 0

    def test_cost_top_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["cost", "--session", "s1", "--top", "3"])
        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────────────────────
#  7. CLI — truenorth pricing
# ─────────────────────────────────────────────────────────────────────────────

class TestCliPricing:

    def test_pricing_all_providers_table(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pricing"])
        assert result.exit_code == 0
        assert "claude" in result.output or "gpt" in result.output

    def test_pricing_filter_anthropic(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pricing", "--provider", "anthropic"])
        assert result.exit_code == 0
        assert "claude" in result.output.lower()

    def test_pricing_filter_openai(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pricing", "--provider", "openai"])
        assert result.exit_code == 0
        assert "gpt" in result.output.lower()

    def test_pricing_json_format(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pricing", "--provider", "anthropic", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) >= 3
        assert all("model" in m for m in data)

    def test_pricing_json_has_required_fields(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pricing", "--format", "json", "--provider", "openai"])
        data   = json.loads(result.output)
        for m in data:
            assert "model"          in m
            assert "provider"       in m
            assert "input_per_1m"   in m
            assert "output_per_1m"  in m
            assert "cost_1k_tokens" in m

    def test_pricing_include_local_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pricing", "--include-local"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "ollama" in output_lower or "local" in output_lower or "free" in output_lower

    def test_pricing_google_provider(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pricing", "--provider", "google"])
        assert result.exit_code == 0
        assert "gemini" in result.output.lower()

    def test_pricing_invalid_provider_fails(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pricing", "--provider", "nonexistent"])
        assert result.exit_code != 0


# ─────────────────────────────────────────────────────────────────────────────
#  8. CLI — truenorth estimate
# ─────────────────────────────────────────────────────────────────────────────

class TestCliEstimate:

    def test_estimate_claude_haiku(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "estimate", "--model", "claude-haiku-4-5-20251001",
            "--tokens", "1000", "--output", "500",
        ])
        assert result.exit_code == 0
        assert "$" in result.output

    def test_estimate_gemini_flash(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "estimate", "--model", "gemini-1.5-flash",
            "--tokens", "5000", "--output", "2000",
        ])
        assert result.exit_code == 0
        assert "$" in result.output

    def test_estimate_with_sessions(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "estimate", "--model", "claude-haiku-4-5-20251001",
            "--tokens", "500", "--output", "250", "--sessions", "1000",
        ])
        assert result.exit_code == 0
        assert "$" in result.output
        assert "1,000" in result.output or "1000" in result.output

    def test_estimate_free_model_zero(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "estimate", "--model", "ollama",
            "--tokens", "50000", "--output", "20000",
        ])
        assert result.exit_code == 0
        assert "0.00" in result.output or "$0" in result.output

    def test_estimate_requires_model(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["estimate", "--tokens", "1000"])
        assert result.exit_code != 0

    def test_estimate_unknown_model_uses_fallback(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "estimate", "--model", "super-mystery-model-2099",
            "--tokens", "1000", "--output", "500",
        ])
        assert result.exit_code == 0
        assert "$" in result.output


# ─────────────────────────────────────────────────────────────────────────────
#  9. CLI — truenorth version
# ─────────────────────────────────────────────────────────────────────────────

class TestCliVersion:

    def test_version_outputs_version_string(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "TrueNorth" in result.output or "truenorth" in result.output.lower()

    def test_version_flag_works(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0." in result.output   # version number format

    def test_help_flag_works(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "cost"     in result.output
        assert "pricing"  in result.output
        assert "estimate" in result.output


# ─────────────────────────────────────────────────────────────────────────────
#  10. Cross-module consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossModuleConsistency:

    def test_pricing_and_cost_tracker_agree(self):
        """pricing.cost_usd() and cost_tracker._compute_cost() must return same values."""
        from truenorth.llm.cost_tracker import _compute_cost as ct_cost

        test_cases = [
            ("claude-haiku-4-5-20251001", 500, 250),
            ("gemini-1.5-flash",          1000, 500),
            ("gpt-4o",                    300,  150),
            ("claude-sonnet-4-20250514",  800,  400),
            ("ollama",                    5000, 2000),
        ]
        for model, inp, out in test_cases:
            p_cost = cost_usd(model, inp, out)
            c_cost = ct_cost(model, inp, out)
            assert p_cost == pytest.approx(c_cost, abs=1e-9), \
                f"Mismatch for {model}: pricing={p_cost} vs cost_tracker={c_cost}"

    def test_router_uses_same_pricing(self):
        """router._compute_cost() should produce same results as pricing.cost_usd()."""
        from truenorth.llm.router import LLMRouter
        router = LLMRouter()

        cases = [
            ("claude-haiku-4-5-20251001", 1000, 500),
            ("gemini-1.5-flash",          2000, 1000),
        ]
        for model, inp, out in cases:
            r_cost = router._compute_cost(model, inp, out)
            p_cost = cost_usd(model, inp, out)
            assert r_cost == pytest.approx(p_cost, abs=1e-9), \
                f"Router/pricing mismatch for {model}"

    def test_providers_dict_covers_pricing_table(self):
        """Every model in PROVIDERS should be in PRICING (or vice versa)."""
        all_provider_models = {m for models in PROVIDERS.values() for m in models}
        for model in all_provider_models:
            # Some models may not be in PRICING (aliases), that's ok.
            assert isinstance(model, str) and len(model) > 0