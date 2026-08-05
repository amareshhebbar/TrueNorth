"""
Authentication middleware for TrueNorth API.

Two authentication schemes:
  1. API Key (X-TrueNorth-Key header)
     — for server-to-server, SDK clients, CLI
     — fast: Redis lookup of key → tenant metadata
     — format: tn_live_<32 hex chars> or tn_test_<32 hex chars>

  2. JWT Bearer token (Authorization: Bearer <token>)
     — for Studio dashboard (browser sessions)
     — contains: tenant_id, user_id, plan, roles
     — verified with HMAC-SHA256 or RS256

Key storage schema (Redis hash):
    tn:apikey:{hash_of_key} → {
        "tenant_id":  "ten-abc",
        "plan":       "pro",
        "active":     true,
        "created_at": 1720000000,
        "label":      "Production key",
        "scopes":     ["read", "write", "admin"],
    }

Usage (FastAPI):
    auth = AuthMiddleware.from_env()

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        result = await auth.verify(request)
        if not result.authenticated:
            return JSONResponse({"error": result.reason}, status_code=401)
        request.state.auth = result
        return await call_next(request)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

class AuthScheme(str, Enum):
    API_KEY  = "api_key"
    JWT      = "jwt"
    NONE     = "none"

@dataclass
class AuthResult:
    authenticated: bool
    scheme:        AuthScheme  = AuthScheme.NONE
    tenant_id:     str         = ""
    user_id:       str         = ""
    plan:          str         = "free"
    scopes:        List[str]   = field(default_factory=list)
    reason:        str         = ""
    api_key_id:    str         = ""
    is_test_key:   bool        = False

    @property
    def is_admin(self) -> bool:
        return "admin" in self.scopes

    @property
    def can_write(self) -> bool:
        return "write" in self.scopes or "admin" in self.scopes

    def to_dict(self) -> dict:
        return {
            "authenticated": self.authenticated,
            "scheme":        self.scheme.value,
            "tenant_id":     self.tenant_id,
            "user_id":       self.user_id,
            "plan":          self.plan,
            "scopes":        self.scopes,
            "is_test_key":   self.is_test_key,
        }

class APIKeyManager:
    """
    Creates, validates, and revokes API keys.

    Keys are stored in Redis (or in-memory for dev) as a hash of the key.
    The raw key is shown only once at creation — after that only the hash
    is stored (same pattern as Stripe / Anthropic API keys).
    """

    KEY_PREFIX_LIVE = "tn_live_"
    KEY_PREFIX_TEST = "tn_test_"
    REDIS_PREFIX    = "tn:apikey:"
    REDIS_TTL       = 86_400 * 365

    def __init__(self, redis: Optional[Any] = None):
        self._redis  = redis
        self._memory: Dict[str, dict] = {}

    def create_key(
        self,
        tenant_id:  str,
        plan:       str          = "starter",
        label:      str          = "",
        scopes:     List[str]    = None,
        is_test:    bool         = False,
    ) -> tuple[str, str]:
        """
        Create a new API key.

        Returns (raw_key, key_id).
        raw_key is shown to the user ONCE — the caller must store it.
        key_id is the hashed identifier used internally.
        """
        prefix  = self.KEY_PREFIX_TEST if is_test else self.KEY_PREFIX_LIVE
        raw_key = prefix + uuid.uuid4().hex + uuid.uuid4().hex[:8]
        key_id  = self._hash_key(raw_key)

        metadata = {
            "tenant_id":  tenant_id,
            "plan":       plan,
            "active":     True,
            "created_at": time.time(),
            "label":      label or f"Key {key_id[:8]}",
            "scopes":     scopes or ["read", "write"],
            "is_test":    is_test,
        }

        self._store(key_id, metadata)
        return raw_key, key_id

    def revoke_key(self, raw_key_or_id: str) -> bool:
        """Revoke an API key. Returns True if the key existed."""
        key_id = (
            raw_key_or_id if len(raw_key_or_id) == 64
            else self._hash_key(raw_key_or_id)
        )
        meta = self._load(key_id)
        if meta is None:
            return False
        meta["active"] = False
        self._store(key_id, meta)
        return True

    def validate(self, raw_key: str) -> Optional[dict]:
        """
        Validate a raw API key. Returns metadata dict or None.
        Fast path: Redis lookup by hash.
        """
        if not raw_key:
            return None
        if not (raw_key.startswith(self.KEY_PREFIX_LIVE)
                or raw_key.startswith(self.KEY_PREFIX_TEST)):
            return None

        key_id = self._hash_key(raw_key)
        meta   = self._load(key_id)
        if meta is None or not meta.get("active", False):
            return None

        return meta

    def register_key(
        self,
        raw_key:   str,
        tenant_id: str,
        plan:      str       = "starter",
        scopes:    List[str] = None,
    ) -> str:
        """
        Register a known raw key (for bootstrapping / seeding).
        Returns the key_id hash.
        """
        key_id = self._hash_key(raw_key)
        self._store(key_id, {
            "tenant_id":  tenant_id,
            "plan":       plan,
            "active":     True,
            "created_at": time.time(),
            "label":      "registered",
            "scopes":     scopes or ["read", "write"],
            "is_test":    raw_key.startswith(self.KEY_PREFIX_TEST),
        })
        return key_id

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def _store(self, key_id: str, meta: dict) -> None:
        self._memory[key_id] = meta
        if self._redis:
            try:
                self._redis.setex(
                    f"{self.REDIS_PREFIX}{key_id}",
                    self.REDIS_TTL,
                    json.dumps(meta),
                )
            except Exception:
                pass

    def _load(self, key_id: str) -> Optional[dict]:
        if self._redis:
            try:
                raw = self._redis.get(f"{self.REDIS_PREFIX}{key_id}")
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        return self._memory.get(key_id)

class JWTHandler:
    """
    Minimal HS256 JWT for Studio dashboard sessions.
    No PyJWT dependency — pure stdlib.

    For production RS256 (asymmetric), swap in PyJWT with public key.
    """

    ALGORITHM = "HS256"
    DEFAULT_TTL = 3600 * 8

    def __init__(self, secret: str):
        if not secret:
            raise ValueError("JWT secret must not be empty")
        self._secret = secret.encode()

    def issue(
        self,
        tenant_id:  str,
        user_id:    str,
        plan:       str       = "starter",
        scopes:     List[str] = None,
        ttl:        int       = DEFAULT_TTL,
    ) -> str:
        """Issue a signed JWT token."""
        import base64
        now = int(time.time())
        header  = {"alg": self.ALGORITHM, "typ": "JWT"}
        payload = {
            "iss":       "truenorth",
            "sub":       user_id,
            "tenant_id": tenant_id,
            "plan":      plan,
            "scopes":    scopes or ["read", "write"],
            "iat":       now,
            "exp":       now + ttl,
        }
        h = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=")
        p = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
        sig = hmac.new(self._secret, h + b"." + p, hashlib.sha256).digest()
        s   = base64.urlsafe_b64encode(sig).rstrip(b"=")
        return f"{h.decode()}.{p.decode()}.{s.decode()}"

    def verify(self, token: str) -> Optional[dict]:
        """
        Verify and decode a JWT. Returns payload dict or None.
        Checks signature and expiry.
        """
        import base64

        def _pad(b: str) -> bytes:
            return base64.urlsafe_b64decode(b + "=" * (-len(b) % 4))

        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            h, p, s = parts

            expected_sig = hmac.new(self._secret, f"{h}.{p}".encode(), hashlib.sha256).digest()
            actual_sig   = _pad(s)
            if not hmac.compare_digest(expected_sig, actual_sig):
                return None

            payload = json.loads(_pad(p))

            if payload.get("exp", 0) < time.time():
                return None

            return payload
        except Exception:
            return None

class AuthMiddleware:
    """
    Dual-scheme authentication middleware.
    Tries API key first (X-TrueNorth-Key), then JWT (Authorization: Bearer).

    Usage:
        auth = AuthMiddleware(
            key_manager = APIKeyManager(redis=redis_client),
            jwt_handler = JWTHandler(secret=os.environ["JWT_SECRET"]),
        )
        result = await auth.verify(request)

    FastAPI integration:
        from fastapi import Request, Depends
        async def require_auth(request: Request) -> AuthResult:
            r = await auth.verify(request)
            if not r.authenticated:
                raise HTTPException(status_code=401, detail=r.reason)
            return r
    """

    HEADER_API_KEY = "X-TrueNorth-Key"
    HEADER_AUTH    = "Authorization"

    def __init__(
        self,
        key_manager:     Optional[APIKeyManager] = None,
        jwt_handler:     Optional[JWTHandler]    = None,
        require_auth:    bool                    = True,
        public_paths:    Optional[List[str]]     = None,
    ):
        self._keys        = key_manager or APIKeyManager()
        self._jwt         = jwt_handler
        self._require     = require_auth
        self._public      = set(public_paths or [
            "/health", "/ready", "/.well-known/agent.json",
        ])

    @classmethod
    def from_env(cls) -> "AuthMiddleware":
        """Build from environment variables."""
        redis = None
        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            try:
                import redis as redis_lib
                redis = redis_lib.from_url(redis_url)
            except ImportError:
                pass

        jwt_secret = os.environ.get("TRUENORTH_JWT_SECRET", "")
        jwt_handler = JWTHandler(jwt_secret) if jwt_secret else None

        keys = APIKeyManager(redis=redis)
        env_key = os.environ.get("TRUENORTH_API_KEY")
        if env_key:
            keys.register_key(env_key, tenant_id="default", plan="pro")

        return cls(key_manager=keys, jwt_handler=jwt_handler)

    async def verify(self, request: Any) -> AuthResult:
        """
        Extract and verify credentials from a request.
        Supports FastAPI Request objects and plain dicts.
        """
        path    = self._get_path(request)
        headers = self._get_headers(request)

        if path in self._public or path.startswith("/.well-known"):
            return AuthResult(authenticated=True, scheme=AuthScheme.NONE,
                              scopes=["read"])

        raw_key = (
            headers.get("x_truenorth_key", "")
            or headers.get(self.HEADER_API_KEY, "")
            or headers.get("x-truenorth-key", "")
        )
        if raw_key:
            return self._verify_api_key(raw_key)

        auth_header = (
            headers.get("authorization", "")
            or headers.get(self.HEADER_AUTH, "")
        )
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return self._verify_jwt(token)

        if not self._require:
            return AuthResult(authenticated=True, scheme=AuthScheme.NONE,
                              tenant_id="anonymous", scopes=["read"])

        return AuthResult(
            authenticated = False,
            reason        = "Authentication required. Provide X-TrueNorth-Key or Authorization: Bearer <token>",
        )

    def verify_sync(self, headers: Dict[str, str], path: str = "/") -> AuthResult:
        """Synchronous verify for non-async contexts."""
        import asyncio

        class _FakeRequest:
            pass

        req = _FakeRequest()
        req.headers = headers
        req.url = type("url", (), {"path": path})()
        try:
            return asyncio.get_event_loop().run_until_complete(self.verify(req))
        except RuntimeError:
            return asyncio.run(self.verify(req))

    def _verify_api_key(self, raw_key: str) -> AuthResult:
        meta = self._keys.validate(raw_key)
        if meta is None:
            return AuthResult(
                authenticated = False,
                scheme        = AuthScheme.API_KEY,
                reason        = "Invalid or revoked API key",
            )
        return AuthResult(
            authenticated = True,
            scheme        = AuthScheme.API_KEY,
            tenant_id     = meta["tenant_id"],
            plan          = meta.get("plan", "free"),
            scopes        = meta.get("scopes", ["read"]),
            is_test_key   = meta.get("is_test", False),
            api_key_id    = hashlib.sha256(raw_key.encode()).hexdigest()[:16],
        )

    def _verify_jwt(self, token: str) -> AuthResult:
        if self._jwt is None:
            return AuthResult(
                authenticated = False,
                scheme        = AuthScheme.JWT,
                reason        = "JWT authentication not configured",
            )
        payload = self._jwt.verify(token)
        if payload is None:
            return AuthResult(
                authenticated = False,
                scheme        = AuthScheme.JWT,
                reason        = "Invalid or expired JWT token",
            )
        return AuthResult(
            authenticated = True,
            scheme        = AuthScheme.JWT,
            tenant_id     = payload.get("tenant_id", ""),
            user_id       = payload.get("sub", ""),
            plan          = payload.get("plan", "free"),
            scopes        = payload.get("scopes", ["read"]),
        )

    @staticmethod
    def _get_path(request: Any) -> str:
        if hasattr(request, "url"):
            return getattr(request.url, "path", "/")
        if isinstance(request, dict):
            return request.get("path", "/")
        return "/"

    @staticmethod
    def _get_headers(request: Any) -> Dict[str, str]:
        if hasattr(request, "headers"):
            h = request.headers
            if hasattr(h, "items"):
                return {k.lower().replace("-", "_"): v
                        for k, v in h.items()}
            if isinstance(h, dict):
                return {k.lower().replace("-", "_"): v for k, v in h.items()}
        if isinstance(request, dict):
            return {k.lower().replace("-", "_"): v
                    for k, v in request.get("headers", {}).items()}
        return {}

    def create_api_key(self, tenant_id: str, **kwargs) -> tuple[str, str]:
        return self._keys.create_key(tenant_id, **kwargs)

    def revoke_api_key(self, raw_key: str) -> bool:
        return self._keys.revoke_key(raw_key)

    def issue_jwt(self, tenant_id: str, user_id: str, **kwargs) -> Optional[str]:
        if not self._jwt:
            return None
        return self._jwt.issue(tenant_id, user_id, **kwargs)
