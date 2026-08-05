"""
Loads goal YAML files, validates them against the JSON Schema,
supports inheritance (extends:) and environment variable substitution.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Union

import yaml

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent.parent.parent.parent.parent / \
               "specs" / "yaml-schema" / "goal.schema.json"

class YAMLLoaderError(Exception):
    """Raised when a goal YAML fails to load or validate."""

class YAMLLoader:
    """
    Loads, validates, and resolves goal YAML configs.

    Features:
      - JSON Schema validation (warns but doesn't block if schema not found)
      - extends: inheritance — child config merges on top of parent
      - ${ENV_VAR} substitution in string values
      - Caching — each file path is loaded once per process
    """

    _cache: Dict[str, dict] = {}

    @classmethod
    def load(cls, path: Union[str, Path], use_cache: bool = True) -> dict:
        path = Path(path).resolve()
        cache_key = str(path)

        if use_cache and cache_key in cls._cache:
            logger.debug("yaml_loader: cache hit for %s", path)
            return cls._cache[cache_key]

        if not path.exists():
            raise YAMLLoaderError(f"Goal YAML not found: {path}")

        logger.info("yaml_loader: loading %s", path)

        try:
            raw = path.read_text(encoding="utf-8")
            raw = cls._substitute_env_vars(raw)
            config: dict = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise YAMLLoaderError(f"Invalid YAML in {path}: {e}") from e

        if not isinstance(config, dict):
            raise YAMLLoaderError(f"{path} must be a YAML mapping at the top level")

        config = cls._resolve_inheritance(config, base_dir=path.parent)
        cls._validate(config, path)
        config = cls._normalise_fields(config)

        if use_cache:
            cls._cache[cache_key] = config

        return config

    @classmethod
    def load_from_string(cls, yaml_text: str) -> dict:
        """Load from a YAML string (useful for tests and API endpoints)."""
        try:
            config = yaml.safe_load(cls._substitute_env_vars(yaml_text))
        except yaml.YAMLError as e:
            raise YAMLLoaderError(f"Invalid YAML: {e}") from e
        if not isinstance(config, dict):
            raise YAMLLoaderError("YAML must be a mapping at the top level")
        return cls._normalise_fields(config)

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()

    @classmethod
    def _substitute_env_vars(cls, text: str) -> str:
        """Replace ${VAR_NAME} and ${VAR_NAME:default} with env values."""
        def replacer(m: re.Match) -> str:
            var, _, default = m.group(1).partition(":")
            return os.environ.get(var, default)
        return re.sub(r"\$\{([^}]+)\}", replacer, text)

    @classmethod
    def _resolve_inheritance(cls, config: dict, base_dir: Path) -> dict:
        """
        If config has an `extends:` key, load the parent YAML and deep-merge.
        Child values override parent values. Fields are merged by name.
        """
        extends = config.pop("extends", None)
        if not extends:
            return config

        parent_path = (base_dir / extends).resolve()
        logger.debug("yaml_loader: resolving extends %s", parent_path)
        parent = cls.load(parent_path)

        merged = cls._deep_merge(parent, config)
        return merged

    @classmethod
    def _deep_merge(cls, base: dict, override: dict) -> dict:
        """
        Deep merge override onto base. Lists with 'name' keys (fields) are
        merged by name rather than replaced.
        """
        result = dict(base)
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = cls._deep_merge(result[key], val)
            elif key == "fields" and isinstance(result.get("fields"), list) and isinstance(val, list):
                result["fields"] = cls._merge_field_lists(result["fields"], val)
            else:
                result[key] = val
        return result

    @classmethod
    def _merge_field_lists(cls, base: list, override: list) -> list:
        """Merge field lists by name — override entries replace base entries with same name."""
        base_map = {f["name"]: f for f in base if isinstance(f, dict) and "name" in f}
        for f in override:
            if isinstance(f, dict) and "name" in f:
                if f["name"] in base_map:
                    base_map[f["name"]] = {**base_map[f["name"]], **f}
                else:
                    base_map[f["name"]] = f
        return list(base_map.values())

    @classmethod
    def _validate(cls, config: dict, path: Path) -> None:
        """Validate against JSON Schema. Logs warnings, does not raise (schema file optional)."""
        try:
            import jsonschema
            schema_path = _SCHEMA_PATH
            if not schema_path.exists():
                schema_path = Path(__file__).parent.parent / "specs" / "yaml-schema" / "goal.schema.json"
            if not schema_path.exists():
                logger.debug("yaml_loader: no schema file found, skipping validation")
                return
            schema = json.loads(schema_path.read_text())
            jsonschema.validate(config, schema)
            logger.debug("yaml_loader: %s passed schema validation", path.name)
        except ImportError:
            logger.debug("yaml_loader: jsonschema not installed, skipping validation")
        except Exception as e:
            logger.warning("yaml_loader: schema validation warning for %s: %s", path.name, e)

    @classmethod
    def _normalise_fields(cls, config: dict) -> dict:
        """
        Ensure every field entry has the required keys with sensible defaults.
        Converts shorthand entries to full dicts.
        """
        raw_fields = config.get("fields", [])
        normalised = []
        for f in raw_fields:
            if isinstance(f, str):

                f = {"name": f}
            f.setdefault("required", True)
            f.setdefault("type", "text")
            f.setdefault("description", f.get("name", "").replace("_", " ").title())
            f.setdefault("question", f"What is your {f.get('name', 'information')}?")
            normalised.append(f)
        config["fields"] = normalised

        if "id" not in config:
            _raw_id = config.get("name", "unknown")

            import re as _re
            _slug = _re.sub(r"[^a-z0-9]+", "_", _raw_id.lower()).strip("_")
            config["id"] = _slug or "unknown"
        config.setdefault("persona", {"name": "TrueNorth", "tone": "friendly"})
        config.setdefault("output", {"format": "text", "template": ""})
        config.setdefault("budget", {})

        raw_mcp = config.get("mcp_servers", [])
        normalised_mcp = []
        for server in raw_mcp:
            if isinstance(server, str):

                normalised_mcp.append({"name": server, "builtin": True})
            elif isinstance(server, dict):
                if "name" not in server:
                    server["name"] = server.get("url", "unnamed").split("/")[-1]
                server.setdefault("builtin", False)
                normalised_mcp.append(server)
        config["mcp_servers"] = normalised_mcp

        return config

    @classmethod
    def list_fields(cls, config: dict) -> list[str]:
        """Return list of field names from a loaded config."""
        return [f["name"] for f in config.get("fields", [])]

    @classmethod
    def required_fields(cls, config: dict) -> list[str]:
        """Return list of required field names."""
        return [f["name"] for f in config.get("fields", []) if f.get("required", True)]
