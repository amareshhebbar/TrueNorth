"""
On-device LLM bridge for iOS and Android.

Why this matters:
  Mobile LLMs run entirely on the user's device — zero cloud cost, zero
  latency, zero privacy risk for sensitive data (medical, financial, legal).
  TrueNorth routes privacy-sensitive tasks (PII extraction, medical intake)
  to the on-device model and keeps only summarisation on the cloud.

Supported platforms:
  iOS 18+    : Apple Intelligence (Foundation Models framework)
               Exposed via a local HTTP bridge app (TrueNorth Mobile SDK)
               Default endpoint: http://127.0.0.1:49152
               Models: apple/on-device-3b, apple/on-device-vision

  Android 9+ : Google Gemini Nano / AICore
               Exposed via local HTTP bridge (Android TrueNorth SDK)
               Default endpoint: http://127.0.0.1:49153
               Models: gemini-nano, gemini-nano-2

  Generic    : Any OpenAI-compatible endpoint on the device.
               Works for: MLX (Apple Silicon Mac), llama.cpp on phone via
               Termux, MLC-LLM, MediaPipe LLM Inference API.

Architecture:
  Mobile SDK (Swift/Kotlin) ←→ Local HTTP server ←→ TrueNorth (this file)

  The mobile SDK wraps the platform's on-device ML framework and exposes
  a minimal HTTP API that TrueNorth calls. The SDK handles:
    - Model loading and warm-up
    - Token streaming
    - Battery/thermal throttling
    - Capability reporting (context window, supported tasks)
"""

from __future__ import annotations

import logging
import os
import platform as _platform
import sys
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from truenorth.llm.base import LLMBase, LLMResponse, Message, StreamChunk, _Timer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Platform defaults
# ─────────────────────────────────────────────────────────────────────────────

class MobilePlatform:
    IOS     = "ios"
    ANDROID = "android"
    GENERIC = "generic"
    AUTO    = "auto"


_PLATFORM_DEFAULTS = {
    MobilePlatform.IOS: {
        "endpoint":    "http://127.0.0.1:49152",
        "model":       "apple/on-device-3b",
        "health_path": "/health",
        "chat_path":   "/chat/completions",
        "models_path": "/models",
        "caps_path":   "/capabilities",
    },
    MobilePlatform.ANDROID: {
        "endpoint":    "http://127.0.0.1:49153",
        "model":       "gemini-nano",
        "health_path": "/health",
        "chat_path":   "/chat/completions",
        "models_path": "/models",
        "caps_path":   "/capabilities",
    },
    MobilePlatform.GENERIC: {
        "endpoint":    "http://127.0.0.1:49154",
        "model":       "on-device",
        "health_path": "/health",
        "chat_path":   "/chat/completions",
        "models_path": "/models",
        "caps_path":   "/capabilities",
    },
}

MOBILE_PREFERRED_TASKS = {"extract", "classify", "embed"}


# ─────────────────────────────────────────────────────────────────────────────
#  Device capability report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeviceCapabilities:
    """Reported by the mobile SDK's /capabilities endpoint."""
    platform:          str
    model_name:        str
    context_window:    int     = 4096
    supports_streaming: bool   = True
    supports_vision:   bool    = False
    max_tokens:        int     = 512     # mobile models have smaller output limits
    battery_level:     float   = 1.0    # 0–1; throttle inference if low
    thermal_state:     str     = "nominal"  # nominal | fair | serious | critical
    is_available:      bool    = True
    preferred_tasks:   List[str] = None  # tasks this device handles well

    def __post_init__(self):
        if self.preferred_tasks is None:
            self.preferred_tasks = list(MOBILE_PREFERRED_TASKS)

    @property
    def should_throttle(self) -> bool:
        """True when device conditions suggest reducing inference load."""
        return (
            self.battery_level < 0.20
            or self.thermal_state in ("serious", "critical")
        )

    def to_dict(self) -> dict:
        return {
            "platform":          self.platform,
            "model_name":        self.model_name,
            "context_window":    self.context_window,
            "supports_streaming": self.supports_streaming,
            "max_tokens":        self.max_tokens,
            "battery_level":     self.battery_level,
            "thermal_state":     self.thermal_state,
            "is_available":      self.is_available,
            "should_throttle":   self.should_throttle,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  MobileLLMClient
# ─────────────────────────────────────────────────────────────────────────────

class MobileLLMClient(LLMBase):
    """
    On-device LLM client. Connects to the TrueNorth Mobile SDK running
    on iOS or Android and calls the on-device model via a local HTTP server.

    Zero cloud cost. Zero latency. Zero privacy risk for sensitive fields.

    Config keys:
      platform           : "ios" | "android" | "generic" | "auto"
      endpoint           : str   — override default localhost port
      model              : str   — model identifier (platform-specific)
      fallback_to_cloud  : bool  — if device unavailable, raise instead of fail
      max_tokens_override: int   — cap output tokens for battery saving
      connect_timeout_s  : float — seconds to wait for device connection
    """

    supports_streaming: bool = True
    max_context_tokens: int  = 4096      # conservative for mobile

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self._platform = self._resolve_platform(
            self.config.get("platform", MobilePlatform.AUTO)
        )
        platform_defaults = _PLATFORM_DEFAULTS.get(
            self._platform, _PLATFORM_DEFAULTS[MobilePlatform.GENERIC]
        )

        self._endpoint    = (
            self.config.get("endpoint")
            or os.environ.get("TRUENORTH_MOBILE_ENDPOINT")
            or platform_defaults["endpoint"]
        )
        self.model_name   = (
            self.config.get("model")
            or os.environ.get("TRUENORTH_MOBILE_MODEL")
            or platform_defaults["model"]
        )
        self._chat_path   = platform_defaults["chat_path"]
        self._health_path = platform_defaults["health_path"]
        self._models_path = platform_defaults["models_path"]
        self._caps_path   = platform_defaults["caps_path"]

        self._fallback    = self.config.get("fallback_to_cloud", True)
        self._connect_to  = self.config.get("connect_timeout_s", 2.0)
        self._max_tok_cap = self.config.get("max_tokens_override")
        self._client      = None
        self._capabilities: Optional[DeviceCapabilities] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def auto_detect(cls, config: Optional[dict] = None) -> "MobileLLMClient":
        cfg = dict(config or {})
        cfg["platform"] = MobilePlatform.AUTO
        return cls(config=cfg)

    # ------------------------------------------------------------------
    # LLMBase implementation
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages:    List[Message],
        system:      Optional[str] = None,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        if self._max_tok_cap:
            max_tokens = min(max_tokens, self._max_tok_cap)

        # Check throttling
        caps = await self._get_capabilities()
        if caps and caps.should_throttle:
            logger.warning(
                "mobile_llm: device throttling (battery=%.0f%% thermal=%s) "
                "— capping tokens to 256",
                caps.battery_level * 100, caps.thermal_state,
            )
            max_tokens = min(max_tokens, 256)

        client  = self._get_client()
        payload = self._build_payload(messages, system, max_tokens, temperature)

        with _Timer() as t:
            try:
                resp = await client.post(
                    self._chat_path,
                    json=payload,
                    timeout=self._connect_to + max_tokens * 0.05,  # generous for inference
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning("mobile_llm: request failed: %s", e)
                raise MobileUnavailableError(
                    platform=self._platform, endpoint=self._endpoint, cause=str(e)
                ) from e

        return self._parse_response(data, t.ms)

    async def generate_stream(
        self,
        messages:    List[Message],
        system:      Optional[str] = None,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        import json as _json

        if self._max_tok_cap:
            max_tokens = min(max_tokens, self._max_tok_cap)

        client  = self._get_client()
        payload = self._build_payload(messages, system, max_tokens, temperature, stream=True)

        try:
            async with client.stream(
                "POST", self._chat_path, json=payload,
                timeout=self._connect_to + max_tokens * 0.05,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        chunk = _json.loads(line)
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {}).get("content", "")
                        )
                        if delta:
                            yield StreamChunk(delta=delta)
                    except (_json.JSONDecodeError, IndexError, KeyError):
                        continue
        except MobileUnavailableError:
            raise
        except Exception as e:
            raise MobileUnavailableError(
                platform=self._platform, endpoint=self._endpoint, cause=str(e)
            ) from e

        yield StreamChunk(delta="", is_final=True)

    # ------------------------------------------------------------------
    # Health and capabilities
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        client = self._get_client()
        try:
            resp = await client.get(self._health_path, timeout=self._connect_to)
            return resp.status_code == 200
        except Exception:
            return False

    async def get_device_capabilities(self) -> Optional[DeviceCapabilities]:
        caps = await self._get_capabilities()
        return caps

    async def list_models(self) -> List[str]:
        client = self._get_client()
        try:
            resp = await client.get(self._models_path, timeout=self._connect_to)
            data = resp.json()
            return [m.get("id", m) for m in data.get("data", [])]
        except Exception as e:
            logger.debug("mobile_llm: list_models failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Task routing helpers
    # ------------------------------------------------------------------

    def can_handle(self, task: str) -> bool:
        return task in MOBILE_PREFERRED_TASKS

    @property
    def platform(self) -> str:
        return self._platform

    @property
    def endpoint(self) -> str:
        return self._endpoint

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url = self._endpoint,
                    timeout  = httpx.Timeout(
                        connect = self._connect_to,
                        read    = 120.0,
                        write   = 10.0,
                        pool    = 5.0,
                    ),
                    headers  = {
                        "Content-Type":        "application/json",
                        "X-TrueNorth-Client":  "truenorth-mobile-bridge/0.1",
                        "X-TrueNorth-Platform": self._platform,
                    },
                )
            except ImportError as e:
                raise RuntimeError("httpx not installed. Run: pip install httpx") from e
        return self._client

    async def _get_capabilities(self) -> Optional[DeviceCapabilities]:
        """Fetch and cache device capabilities."""
        if self._capabilities is not None:
            return self._capabilities
        client = self._get_client()
        try:
            resp = await client.get(self._caps_path, timeout=self._connect_to)
            data = resp.json()
            self._capabilities = DeviceCapabilities(
                platform          = data.get("platform", self._platform),
                model_name        = data.get("model_name", self.model_name),
                context_window    = data.get("context_window", 4096),
                supports_streaming= data.get("supports_streaming", True),
                supports_vision   = data.get("supports_vision", False),
                max_tokens        = data.get("max_tokens", 512),
                battery_level     = data.get("battery_level", 1.0),
                thermal_state     = data.get("thermal_state", "nominal"),
                is_available      = data.get("is_available", True),
                preferred_tasks   = data.get("preferred_tasks", list(MOBILE_PREFERRED_TASKS)),
            )
            return self._capabilities
        except Exception:
            return None

    def _build_payload(
        self,
        messages:    List[Message],
        system:      Optional[str],
        max_tokens:  int,
        temperature: float,
        stream:      bool = False,
    ) -> dict:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs += [{"role": m.role, "content": m.content} for m in messages]
        return {
            "model":       self.model_name,
            "messages":    msgs,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      stream,
        }

    def _parse_response(self, data: dict, latency_ms: int) -> LLMResponse:
        choices  = data.get("choices", [{}])
        content  = choices[0].get("message", {}).get("content", "") if choices else ""
        usage    = data.get("usage", {})
        return LLMResponse(
            content       = content,
            model         = self.model_name,
            input_tokens  = usage.get("prompt_tokens",     self._count_tokens_approx(content)),
            output_tokens = usage.get("completion_tokens", self._count_tokens_approx(content)),
            latency_ms    = latency_ms,
            raw           = data,
        )

    @staticmethod
    def _resolve_platform(platform: str) -> str:
        if platform != MobilePlatform.AUTO:
            return platform
        env_platform = os.environ.get("TRUENORTH_MOBILE_PLATFORM", "").lower()
        if env_platform in (MobilePlatform.IOS, MobilePlatform.ANDROID):
            return env_platform
        if sys.platform == "darwin" and _platform.machine() == "arm64":
            return MobilePlatform.IOS

        return MobilePlatform.GENERIC


# ─────────────────────────────────────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class MobileUnavailableError(Exception):
    """
    Raised when the on-device bridge is unreachable.
    The router catches this and falls back to the cloud model.
    """
    def __init__(self, platform: str, endpoint: str, cause: str):
        self.platform = platform
        self.endpoint = endpoint
        self.cause    = cause
        super().__init__(
            f"Mobile LLM unavailable: platform={platform} endpoint={endpoint} — {cause}"
        )