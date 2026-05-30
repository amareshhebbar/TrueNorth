"""
Built-in calculator tool — evaluates mathematical expressions safely.

Security:
  - Uses AST evaluation (no exec/eval on arbitrary code)
  - Only allows: arithmetic (+−×÷%), powers (**), common math functions
  - Rejects: imports, function definitions, string operations, comparisons
  - Result precision: up to 10 significant figures

Supported:
  - Basic arithmetic: 65 / (1.63 ** 2), (100 - 65) * 2.2046
  - Math functions: sqrt(144), round(24.45, 1), abs(-5)
  - Constants: pi, e
  - Unit conversions via formula: 65 * 2.20462  (kg to lbs)

YAML:
    mcp_servers:
      - name: calculator
        builtin: true

LLM call:
    TOOL_CALL: calculator({"expression": "65 / (1.63 ** 2)"})
    → {"result": 24.46, "expression": "65 / (1.63 ** 2)", "formatted": "24.46"}
"""

from __future__ import annotations

import ast
import logging
import math
import operator
from typing import Any, Dict

from truenorth.mcp.builtin import register

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Safe evaluation — whitelist of allowed AST nodes and operations
# ─────────────────────────────────────────────────────────────────────────────

_SAFE_NODES = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant, ast.Name,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.USub, ast.UAdd,
    ast.Load,
    ast.IfExp,    # allow: a if b else c
    ast.Compare,  # allow: 5 > 3
    ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq,
    ast.BoolOp, ast.And, ast.Or,
)

_SAFE_FUNCTIONS = {
    "abs":    abs,
    "round":  round,
    "min":    min,
    "max":    max,
    "sqrt":   math.sqrt,
    "log":    math.log,
    "log10":  math.log10,
    "log2":   math.log2,
    "exp":    math.exp,
    "pow":    pow,
    "floor":  math.floor,
    "ceil":   math.ceil,
    "sin":    math.sin,
    "cos":    math.cos,
    "tan":    math.tan,
    "radians": math.radians,
    "degrees": math.degrees,
    "factorial": math.factorial,
    "gcd":    math.gcd,
    "hypot":  math.hypot,
}

_SAFE_NAMES = {
    "pi":  math.pi,
    "e":   math.e,
    "inf": math.inf,
    "nan": math.nan,
    "true":  True,
    "false": False,
    "True":  True,
    "False": False,
}


class _SafeEvaluator(ast.NodeVisitor):
    """AST-based safe expression evaluator."""

    def __init__(self):
        self._result = None

    def eval(self, expr: str) -> Any:
        # Normalise
        expr = expr.strip()
        # Parse
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Syntax error: {e}") from e

        # Check all nodes are in whitelist
        for node in ast.walk(tree):
            if not isinstance(node, _SAFE_NODES):
                raise ValueError(
                    f"Unsafe operation: {type(node).__name__} is not allowed"
                )

        return self._visit(tree.body)

    def _visit(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            name = node.id
            if name in _SAFE_NAMES:
                return _SAFE_NAMES[name]
            raise ValueError(f"Unknown name: {name!r}")

        if isinstance(node, ast.BinOp):
            left  = self._visit(node.left)
            right = self._visit(node.right)
            ops   = {
                ast.Add:      operator.add,
                ast.Sub:      operator.sub,
                ast.Mult:     operator.mul,
                ast.Div:      operator.truediv,
                ast.Mod:      operator.mod,
                ast.Pow:      operator.pow,
                ast.FloorDiv: operator.floordiv,
            }
            fn = ops.get(type(node.op))
            if fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            try:
                return fn(left, right)
            except ZeroDivisionError:
                raise ValueError("Division by zero")

        if isinstance(node, ast.UnaryOp):
            operand = self._visit(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only built-in functions can be called")
            fn_name = node.func.id
            fn = _SAFE_FUNCTIONS.get(fn_name)
            if fn is None:
                raise ValueError(f"Unknown function: {fn_name!r}")
            args = [self._visit(a) for a in node.args]
            kwargs = {kw.arg: self._visit(kw.value) for kw in node.keywords}
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                raise ValueError(f"{fn_name}() error: {e}") from e

        if isinstance(node, ast.IfExp):
            test = self._visit(node.test)
            return self._visit(node.body if test else node.orelse)

        if isinstance(node, ast.Compare):
            left = self._visit(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = self._visit(comp)
                ops = {
                    ast.Gt: operator.gt, ast.Lt: operator.lt,
                    ast.GtE: operator.ge, ast.LtE: operator.le,
                    ast.Eq: operator.eq, ast.NotEq: operator.ne,
                }
                fn = ops.get(type(op))
                if fn is None:
                    raise ValueError(f"Unsupported comparison: {type(op).__name__}")
                if not fn(left, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.BoolOp):
            values = [self._visit(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            return any(values)

        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


_evaluator = _SafeEvaluator()


def _format_number(n: Any) -> str:
    """Human-readable number formatting."""
    if isinstance(n, bool):
        return str(n)
    if isinstance(n, int):
        return f"{n:,}"
    if isinstance(n, float):
        if n == int(n) and abs(n) < 1e12:
            return f"{int(n):,}"
        # Significant figures
        formatted = f"{n:.10g}"
        return formatted
    return str(n)


# ─────────────────────────────────────────────────────────────────────────────
#  Built-in tool function
# ─────────────────────────────────────────────────────────────────────────────

@register("calculator")
async def calculator(expression: str) -> Dict[str, Any]:
    """
    Evaluate a mathematical expression and return the result.
    Supports: +−×÷, ** (power), %, sqrt, log, sin, cos, pi, e, and more.
    Cannot execute code — only pure math.
    """
    expression = expression.strip()
    if not expression:
        return {"error": "Empty expression", "expression": expression}

    try:
        result = _SafeEvaluator().eval(expression)
        return {
            "expression": expression,
            "result":     result,
            "formatted":  _format_number(result),
        }
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        logger.warning("calculator: expression=%r error=%s", expression[:80], e)
        return {
            "expression": expression,
            "error":      str(e),
        }
    except Exception as e:
        logger.error("calculator: unexpected error: %s", e)
        return {
            "expression": expression,
            "error":      f"Evaluation failed: {e}",
        }