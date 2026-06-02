"""
GoalRegistry — the marketplace for TrueNorth goal YAMLs.

"npm for AI agents" — developers publish goal YAMLs to the registry
and users install them with one command:

    $ truenorth install fitness-coach
    $ truenorth install medical-intake --version 2.1.0
    $ truenorth install @acmecorp/legal-intake

Analogous to npm/pip: goals are versioned, discoverable, and
composable. One developer's medical intake becomes another
developer's building block.

Registry architecture:
  - Goals are stored as versioned YAML files
  - Metadata: name, version, author, description, tags, sector, downloads
  - In-memory registry for testing (no HTTP)
  - HTTP registry for production (truenorth.ai/registry)
  - Local registry for private enterprise deployment

Goal YAML package format:
    name:        fitness-coach
    version:     1.3.0
    author:      "@priya_mehta"
    description: "Fitness intake + weekly plan generation"
    sector:      fitness
    tags:        [fitness, wellness, bmi, workout]
    license:     MIT

    fields:     [...]
    output:     {...}
    follow_up:  [...]

CLI commands (wired to cli/main.py):
    truenorth install <goal-name>         # install latest
    truenorth install <name>@<version>    # install specific version
    truenorth publish <goal.yaml>         # publish to registry
    truenorth search fitness              # search by keyword
    truenorth list                        # show installed goals
    truenorth info fitness-coach          # show goal metadata

Usage:
    registry = GoalRegistry()
    registry.publish(goal_yaml_string, author="@priya")
    goal_config = await registry.install("fitness-coach")
    results = registry.search("fitness")
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ─────────────────────────────────────────────────────────────────────────────
#  Goal metadata
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoalPackage:
    """
    A published goal package in the registry.
    Equivalent to a package.json / setup.cfg entry.
    """
    name:         str
    version:      str
    author:       str
    description:  str
    sector:       str              # fitness | medical | legal | hr | finance | other
    tags:         List[str]        = field(default_factory=list)
    license:      str              = "MIT"
    homepage:     str              = ""
    source_url:   str              = ""
    yaml_content: str              = ""    # full YAML string
    downloads:    int              = 0
    published_at: float            = field(default_factory=time.time)
    checksum:     str              = ""    # sha256 of yaml_content

    @property
    def full_name(self) -> str:
        return f"{self.name}@{self.version}"

    def to_dict(self, include_yaml: bool = False) -> dict:
        d = {
            "name":        self.name,
            "version":     self.version,
            "author":      self.author,
            "description": self.description,
            "sector":      self.sector,
            "tags":        self.tags,
            "license":     self.license,
            "downloads":   self.downloads,
            "published_at":self.published_at,
            "checksum":    self.checksum,
        }
        if include_yaml:
            d["yaml_content"] = self.yaml_content
        return d

    @classmethod
    def from_yaml_string(cls, yaml_str: str, author: str = "") -> "GoalPackage":
        """Parse a goal YAML string into a GoalPackage."""
        if not _HAS_YAML:
            raise ImportError("pyyaml required: pip install pyyaml")
        data = _yaml.safe_load(yaml_str)
        checksum = hashlib.sha256(yaml_str.encode()).hexdigest()
        return cls(
            name         = data.get("name", "unnamed"),
            version      = data.get("version", "1.0.0"),
            author       = data.get("author", author),
            description  = data.get("description", ""),
            sector       = data.get("sector", "other"),
            tags         = data.get("tags", []),
            license      = data.get("license", "MIT"),
            homepage     = data.get("homepage", ""),
            yaml_content = yaml_str,
            checksum     = checksum,
        )

    def to_goal_config(self) -> dict:
        """Parse the YAML content into a goal config dict."""
        if not _HAS_YAML:
            raise ImportError("pyyaml required: pip install pyyaml")
        return _yaml.safe_load(self.yaml_content)


# ─────────────────────────────────────────────────────────────────────────────
#  GoalRegistry
# ─────────────────────────────────────────────────────────────────────────────

class GoalRegistry:
    """
    Package registry for TrueNorth goal YAMLs.

    In-memory by default (for testing and local use).
    Set registry_url to use the remote registry at truenorth.ai.

    Usage:
        registry = GoalRegistry()

        # Publish a goal
        registry.publish(yaml_str, author="@dev")

        # Install a goal
        config = registry.install("fitness-coach")

        # Search
        results = registry.search("medical intake")

        # Local goal storage
        registry.set_install_dir("~/.truenorth/goals")
    """

    DEFAULT_REGISTRY_URL = "https://registry.truenorth.ai"
    INSTALL_DIR_DEFAULT  = "~/.truenorth/goals"

    def __init__(
        self,
        registry_url:  Optional[str] = None,
        install_dir:   Optional[str] = None,
        http_timeout:  float         = 10.0,
    ):
        self._url        = registry_url    # None = in-memory only
        self._install_dir = Path(install_dir or self.INSTALL_DIR_DEFAULT).expanduser()
        self._timeout    = http_timeout

        # In-memory store: {name: {version: GoalPackage}}
        self._store:     Dict[str, Dict[str, GoalPackage]] = {}
        self._installed: Dict[str, str] = {}   # name → version

        # Pre-seed with curated official goals
        self._seed_official_goals()

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(
        self,
        yaml_str:  str,
        author:    str  = "",
        overwrite: bool = False,
    ) -> GoalPackage:
        """
        Publish a goal YAML to the registry.
        Returns the GoalPackage. Raises if already exists and overwrite=False.
        """
        pkg = GoalPackage.from_yaml_string(yaml_str, author)
        self._validate_package(pkg)

        existing = self._store.get(pkg.name, {})
        if pkg.version in existing and not overwrite:
            raise ValueError(
                f"Goal '{pkg.name}@{pkg.version}' already exists. "
                f"Bump version or use overwrite=True."
            )

        if pkg.name not in self._store:
            self._store[pkg.name] = {}
        self._store[pkg.name][pkg.version] = pkg

        if self._url:
            self._remote_publish(pkg)

        return pkg

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    def install(
        self,
        name:      str,
        version:   str   = "latest",
        save_local: bool = True,
    ) -> dict:
        """
        Install a goal by name (and optional version).
        Returns the goal config dict, ready to pass to TrueNorthEngine.

        Parses name@version shorthand: "fitness-coach@1.2.0"
        """
        name, version = self._parse_name_version(name, version)

        pkg = self._resolve(name, version)
        if pkg is None:
            raise LookupError(
                f"Goal '{name}@{version}' not found in registry. "
                f"Try: truenorth search {name.split('/')[0]}"
            )

        pkg.downloads += 1
        self._installed[name] = pkg.version

        if save_local:
            self._save_local(pkg)

        return pkg.to_goal_config()

    def install_from_file(self, path: str) -> dict:
        """Install a goal directly from a local YAML file."""
        content = Path(path).read_text()
        pkg     = GoalPackage.from_yaml_string(content)
        self._installed[pkg.name] = pkg.version
        return pkg.to_goal_config()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query:    str           = "",
        sector:   Optional[str] = None,
        tag:      Optional[str] = None,
        limit:    int           = 10,
    ) -> List[dict]:
        """
        Search the registry. Returns list of package metadata dicts.
        Ranks by: exact name match > description match > tag match > downloads.
        """
        query_lower = query.lower()
        results     = []

        for name, versions in self._store.items():
            pkg = self._latest_version(versions)
            if not pkg:
                continue
            if sector and pkg.sector.lower() != sector.lower():
                continue

            if tag and tag.lower() not in [t.lower() for t in pkg.tags]:
                continue

            # Score
            score = 0
            if query_lower:
                if query_lower == pkg.name.lower():
                    score += 100
                elif query_lower in pkg.name.lower():
                    score += 50
                if query_lower in pkg.description.lower():
                    score += 30
                if any(query_lower in t.lower() for t in pkg.tags):
                    score += 20
                if score == 0:
                    continue 
            else:
                score = pkg.downloads  

            results.append((score + pkg.downloads * 0.01, pkg))

        results.sort(reverse=True)
        return [pkg.to_dict() for _, pkg in results[:limit]]

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self, name: str, version: str = "latest") -> Optional[dict]:
        """Return full metadata for a goal (including all available versions)."""
        name, _ = self._parse_name_version(name, version)
        versions = self._store.get(name)
        if not versions:
            return None
        pkg = self._resolve(name, version)
        if not pkg:
            return None
        d = pkg.to_dict()
        d["available_versions"] = sorted(versions.keys(), key=self._version_key, reverse=True)
        return d

    def list_installed(self) -> List[dict]:
        """Return metadata for all installed goals."""
        result = []
        for name, version in self._installed.items():
            pkg = self._resolve(name, version)
            if pkg:
                result.append(pkg.to_dict())
        return result

    def uninstall(self, name: str) -> bool:
        """Remove a goal from the installed list."""
        if name in self._installed:
            del self._installed[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _resolve(self, name: str, version: str) -> Optional[GoalPackage]:
        """Resolve name+version to a GoalPackage."""
        versions = self._store.get(name)
        if not versions:
            if self._url:
                return self._remote_fetch(name, version)
            return None

        if version == "latest":
            return self._latest_version(versions)
        return versions.get(version)

    @staticmethod
    def _latest_version(versions: Dict[str, GoalPackage]) -> Optional[GoalPackage]:
        if not versions:
            return None
        sorted_v = sorted(versions.keys(), key=GoalRegistry._version_key, reverse=True)
        return versions[sorted_v[0]]

    @staticmethod
    def _version_key(v: str) -> tuple:
        """Sort semantic versions correctly."""
        try:
            parts = [int(x) for x in re.split(r"[.\-]", v) if x.isdigit()]
            return tuple(parts)
        except (ValueError, TypeError):
            return (0,)

    @staticmethod
    def _parse_name_version(name: str, default_version: str) -> tuple[str, str]:
        """Parse 'fitness-coach@1.2.0' into ('fitness-coach', '1.2.0')."""
        if "@" in name and not name.startswith("@"):
            n, v = name.rsplit("@", 1)
            return n, v
        return name, default_version

    # ------------------------------------------------------------------
    # Local storage
    # ------------------------------------------------------------------

    def _save_local(self, pkg: GoalPackage) -> None:
        """Save installed goal YAML to ~/.truenorth/goals/."""
        try:
            self._install_dir.mkdir(parents=True, exist_ok=True)
            path = self._install_dir / f"{pkg.name}@{pkg.version}.yaml"
            path.write_text(pkg.yaml_content)
        except Exception:
            pass  

    # ------------------------------------------------------------------
    # Remote registry (HTTP)
    # ------------------------------------------------------------------

    def _remote_fetch(self, name: str, version: str) -> Optional[GoalPackage]:
        """Fetch a goal package from the remote registry."""
        try:
            import urllib.request
            import json
            ver_path = "latest" if version == "latest" else version
            url  = f"{self._url}/goals/{name}/{ver_path}"
            req  = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
            return GoalPackage(**data)
        except Exception:
            return None

    def _remote_publish(self, pkg: GoalPackage) -> None:
        """Publish to remote registry (best-effort)."""
        try:
            import urllib.request
            import json
            data    = json.dumps(pkg.to_dict(include_yaml=True)).encode()
            url     = f"{self._url}/goals"
            request = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=self._timeout)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_package(pkg: GoalPackage) -> None:
        """Basic sanity checks before publishing."""
        if not pkg.name:
            raise ValueError("Goal must have a name")
        if not re.match(r"^[a-z0-9][a-z0-9\-]{0,63}$", pkg.name):
            raise ValueError(
                f"Goal name '{pkg.name}' invalid. "
                f"Use lowercase letters, numbers, and hyphens only."
            )
        if not re.match(r"^\d+\.\d+\.\d+", pkg.version):
            raise ValueError(
                f"Version '{pkg.version}' must follow semver (e.g. 1.0.0)"
            )
        if len(pkg.yaml_content) < 50:
            raise ValueError("Goal YAML content too short")

    # ------------------------------------------------------------------
    # Official curated goals (seed data)
    # ------------------------------------------------------------------

    def _seed_official_goals(self) -> None:
        """Pre-load curated official goal packages."""
        _OFFICIAL = [
            {
                "name": "fitness-coach",
                "version": "1.3.0",
                "author": "@truenorth-official",
                "description": "Personalised fitness assessment and weekly plan. Collects goals, current fitness level, equipment access, and generates a structured weekly programme.",
                "sector": "fitness",
                "tags": ["fitness", "wellness", "bmi", "workout", "weight-loss"],
                "license": "MIT",
                "downloads": 12847,
            },
            {
                "name": "medical-intake",
                "version": "2.1.0",
                "author": "@truenorth-official",
                "description": "DPDP-compliant medical intake. Collects chief complaint, pain scale, medications, allergies, and medical history.",
                "sector": "medical",
                "tags": ["medical", "healthcare", "intake", "dpdp", "hipaa"],
                "license": "MIT",
                "downloads": 8934,
            },
            {
                "name": "legal-intake",
                "version": "1.0.2",
                "author": "@truenorth-official",
                "description": "Legal case intake for personal injury, contract disputes, and criminal matters. GDPR/DPDP compliant.",
                "sector": "legal",
                "tags": ["legal", "case-intake", "personal-injury", "gdpr"],
                "license": "MIT",
                "downloads": 5210,
            },
            {
                "name": "hr-screening",
                "version": "1.2.1",
                "author": "@truenorth-official",
                "description": "HR candidate screening. Collects experience, salary expectations, notice period, and role fit questions.",
                "sector": "hr",
                "tags": ["hr", "recruitment", "screening", "candidate"],
                "license": "MIT",
                "downloads": 7340,
            },
            {
                "name": "financial-plan",
                "version": "1.1.0",
                "author": "@truenorth-official",
                "description": "Personal financial planning intake. Income, expenses, risk tolerance, investment goals, and generates a structured savings plan.",
                "sector": "finance",
                "tags": ["finance", "investment", "savings", "planning", "kyc"],
                "license": "MIT",
                "downloads": 6190,
            },
            {
                "name": "nutrition-coach",
                "version": "1.0.0",
                "author": "@truenorth-official",
                "description": "Nutrition assessment and meal plan generation. Works with fitness-coach via state_transfer.",
                "sector": "fitness",
                "tags": ["nutrition", "diet", "meal-plan", "calories"],
                "license": "MIT",
                "downloads": 3820,
            },
        ]

        for meta in _OFFICIAL:
            yaml_stub = f"""\
name: {meta['name']}
version: {meta['version']}
author: "{meta['author']}"
description: "{meta['description']}"
sector: {meta['sector']}
tags: {meta['tags']}
license: {meta['license']}
# Full implementation available at https://registry.truenorth.ai/goals/{meta['name']}
fields: []
output:
  format: json
"""
            pkg = GoalPackage(
                yaml_content = yaml_stub,
                checksum     = hashlib.sha256(yaml_stub.encode()).hexdigest(),
                **meta,
            )
            if pkg.name not in self._store:
                self._store[pkg.name] = {}
            self._store[pkg.name][pkg.version] = pkg