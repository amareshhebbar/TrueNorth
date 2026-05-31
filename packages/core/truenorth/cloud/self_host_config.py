"""
Self-host configuration generator for TrueNorth.

Generates a production-ready docker-compose.yml and .env template
so anyone can self-host TrueNorth on their own infrastructure in
under 5 minutes.

    $ pip install truenorth
    $ truenorth self-host init --dir ./my-truenorth
    # Edit .env with your API keys
    $ cd my-truenorth && docker compose up -d

Generated stack:
  truenorth-api     — FastAPI server (port 8000)
  truenorth-worker  — Celery worker for background reminders
  postgres          — Session state + long-term memory
  redis             — Rate limiting + cost tracking + cache
  nginx             — Reverse proxy + TLS termination

Profiles:
  minimal   — api + postgres + redis (no nginx, no worker)
  standard  — full stack above
  enterprise— standard + monitoring (Prometheus + Grafana)

Usage:
    from truenorth.cloud.self_host_config import SelfHostConfig
    cfg = SelfHostConfig(profile="standard")
    cfg.generate(output_dir="./deploy")
    # Writes: docker-compose.yml, .env.template, nginx.conf, README.md
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class DeployProfile(str, Enum):
    MINIMAL    = "minimal"
    STANDARD   = "standard"
    ENTERPRISE = "enterprise"


@dataclass
class SelfHostConfig:
    """
    Configuration for a TrueNorth self-hosted deployment.

    Args:
        profile:        minimal | standard | enterprise
        port:           API server port (default 8000)
        domain:         your domain for nginx TLS config
        image_tag:      Docker image tag (default: latest)
        postgres_db:    Postgres database name
        with_monitoring: Include Prometheus + Grafana
    """
    profile:          DeployProfile = DeployProfile.STANDARD
    port:             int           = 8000
    domain:           str           = "localhost"
    image_tag:        str           = "latest"
    postgres_db:      str           = "truenorth"
    postgres_user:    str           = "truenorth"
    with_monitoring:  bool          = False
    with_worker:      bool          = True
    extra_services:   Dict[str, str] = field(default_factory=dict)

    def generate(self, output_dir: str = ".") -> List[str]:
        """
        Generate all deployment files in output_dir.
        Returns list of created file paths.
        """
        out   = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = []

        compose_path = out / "docker-compose.yml"
        compose_path.write_text(self._docker_compose())
        files.append(str(compose_path))

        env_path = out / ".env.template"
        env_path.write_text(self._env_template())
        files.append(str(env_path))

        nginx_path = out / "nginx.conf"
        nginx_path.write_text(self._nginx_conf())
        files.append(str(nginx_path))

        readme_path = out / "README.md"
        readme_path.write_text(self._readme())
        files.append(str(readme_path))

        if self.profile == DeployProfile.ENTERPRISE or self.with_monitoring:
            prom_path = out / "prometheus.yml"
            prom_path.write_text(self._prometheus_config())
            files.append(str(prom_path))

        return files

    # ------------------------------------------------------------------
    # docker-compose.yml
    # ------------------------------------------------------------------

    def _docker_compose(self) -> str:
        worker_svc = textwrap.dedent(f"""
  truenorth-worker:
    image: truenorthai/truenorth:{self.image_tag}
    command: celery -A truenorth.worker worker --loglevel=info -Q reminders,emails
    env_file: .env
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
""") if self.with_worker else ""

        monitoring_svcs = textwrap.dedent("""
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped
""") if (self.with_monitoring or self.profile == DeployProfile.ENTERPRISE) else ""

        monitoring_volumes = "\n  prometheus_data:\n  grafana_data:" \
            if (self.with_monitoring or self.profile == DeployProfile.ENTERPRISE) else ""

        nginx_svc = textwrap.dedent(f"""
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - ssl_certs:/etc/nginx/ssl
    depends_on:
      - truenorth-api
    restart: unless-stopped
""") if self.profile != DeployProfile.MINIMAL else ""

        nginx_volume = "\n  ssl_certs:" if self.profile != DeployProfile.MINIMAL else ""

        return textwrap.dedent(f"""\
# TrueNorth Self-Host — generated by `truenorth self-host init`
# Profile: {self.profile.value}
# Edit .env with your API keys, then: docker compose up -d

version: "3.9"

services:
  truenorth-api:
    image: truenorthai/truenorth:{self.image_tag}
    ports:
      - "{self.port}:8000"
    env_file: .env
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
{worker_svc}
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB:       {self.postgres_db}
      POSTGRES_USER:     {self.postgres_user}
      POSTGRES_PASSWORD: ${{POSTGRES_PASSWORD}}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U {self.postgres_user}"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${{REDIS_PASSWORD}} --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    restart: unless-stopped
{nginx_svc}{monitoring_svcs}
volumes:
  postgres_data:
  redis_data:{nginx_volume}{monitoring_volumes}
""")

    # ------------------------------------------------------------------
    # .env.template
    # ------------------------------------------------------------------

    def _env_template(self) -> str:
        return textwrap.dedent(f"""\
# TrueNorth Environment Configuration
# Copy this file to .env and fill in your values.
# NEVER commit .env to version control.

# ── Database ──────────────────────────────────────────────────────────────
DATABASE_URL=postgresql://{self.postgres_user}:CHANGE_ME@postgres:5432/{self.postgres_db}
POSTGRES_PASSWORD=CHANGE_ME

# ── Redis ─────────────────────────────────────────────────────────────────
REDIS_URL=redis://:CHANGE_ME@redis:6379/0
REDIS_PASSWORD=CHANGE_ME

# ── Security ──────────────────────────────────────────────────────────────
TRUENORTH_JWT_SECRET=CHANGE_ME_TO_64_RANDOM_CHARS
TRUENORTH_ENCRYPTION_KEY=CHANGE_ME_TO_32_RANDOM_CHARS

# ── LLM Providers (add the ones you use) ──────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...

# ── Routing defaults ──────────────────────────────────────────────────────
TRUENORTH_MODEL_EXTRACT=gemini-1.5-flash
TRUENORTH_MODEL_CONVERSE=claude-haiku-4-5-20251001
TRUENORTH_MODEL_OUTPUT=claude-sonnet-4-20250514
TRUENORTH_MODEL_VERIFY=claude-sonnet-4-20250514

# ── Plan and billing ──────────────────────────────────────────────────────
TRUENORTH_PLAN=pro
TRUENORTH_BUDGET_USD=10.00

# ── Initial admin API key (shown once, rotate after first login) ───────────
TRUENORTH_API_KEY=tn_live_CHANGE_ME

# ── WhatsApp (optional) ───────────────────────────────────────────────────
WA_VERIFY_TOKEN=
WA_ACCESS_TOKEN=
WA_PHONE_NUMBER_ID=

# ── Email reminders (optional) ────────────────────────────────────────────
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM=noreply@{self.domain}

# ── Compliance ────────────────────────────────────────────────────────────
COMPLIANCE_MODE=dpdp              # dpdp | gdpr | none
DATA_FIDUCIARY=Your Company Name

# ── Monitoring ────────────────────────────────────────────────────────────
SENTRY_DSN=
GRAFANA_PASSWORD=CHANGE_ME
""")

    # ------------------------------------------------------------------
    # nginx.conf
    # ------------------------------------------------------------------

    def _nginx_conf(self) -> str:
        return textwrap.dedent(f"""\
# TrueNorth nginx reverse proxy
# Place your TLS certificates at /etc/nginx/ssl/cert.pem and key.pem

upstream truenorth {{
    server truenorth-api:8000;
}}

server {{
    listen 80;
    server_name {self.domain};
    return 301 https://$host$request_uri;
}}

server {{
    listen 443 ssl http2;
    server_name {self.domain};

    ssl_certificate     /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Content-Type-Options    nosniff;
    add_header X-Frame-Options           DENY;

    # Rate limit zone (nginx-level, before TrueNorth rate limiter)
    limit_req_zone $binary_remote_addr zone=api:10m rate=30r/s;

    location / {{
        limit_req zone=api burst=50 nodelay;

        proxy_pass         http://truenorth;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_send_timeout 30s;
    }}

    # WebSocket support (for Studio live preview)
    location /ws {{
        proxy_pass         http://truenorth;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade    $http_upgrade;
        proxy_set_header   Connection "upgrade";
    }}
}}
""")

    # ------------------------------------------------------------------
    # Prometheus config
    # ------------------------------------------------------------------

    def _prometheus_config(self) -> str:
        return textwrap.dedent("""\
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: truenorth
    static_configs:
      - targets: ['truenorth-api:8000']
    metrics_path: /metrics
""")

    # ------------------------------------------------------------------
    # README.md
    # ------------------------------------------------------------------

    def _readme(self) -> str:
        return textwrap.dedent(f"""\
# TrueNorth Self-Host ({self.profile.value} profile)

Generated by `truenorth self-host init`. Deploy in 3 steps:

## Quick start

```bash
# 1. Configure your environment
cp .env.template .env
nano .env   # fill in API keys and passwords

# 2. Start the stack
docker compose up -d

# 3. Verify it's running
curl http://localhost:{self.port}/health
```

## Services

| Service | Port | Purpose |
|---------|------|---------|
| truenorth-api | {self.port} | Main API server |
| postgres | internal | Session + memory storage |
| redis | internal | Rate limiting + cost cache |
{"| nginx | 80, 443 | Reverse proxy + TLS |" if self.profile != DeployProfile.MINIMAL else ""}
{"| truenorth-worker | internal | Reminder delivery |" if self.with_worker else ""}

## First API call

```bash
# Create your first session
curl -X POST http://localhost:{self.port}/session \\
  -H "X-TrueNorth-Key: $TRUENORTH_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"goal_id": "fitness_plan"}}'
```

## Security checklist

- [ ] Changed all CHANGE_ME values in .env
- [ ] Rotated TRUENORTH_API_KEY after first login
- [ ] Enabled TLS (added SSL certificates)
- [ ] Set strong POSTGRES_PASSWORD and REDIS_PASSWORD
- [ ] Configured firewall to block direct Postgres/Redis access

## Upgrade

```bash
docker compose pull
docker compose up -d
```

## Logs

```bash
docker compose logs -f truenorth-api
docker compose logs -f truenorth-worker
```

## Support

Docs: https://docs.truenorth.ai/self-host
Issues: https://github.com/truenorth-ai/truenorth/issues
""")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI command handler (called by truenorth CLI)
# ─────────────────────────────────────────────────────────────────────────────

def cli_init(
    output_dir:  str           = ".",
    profile:     str           = "standard",
    port:        int           = 8000,
    domain:      str           = "localhost",
    image_tag:   str           = "latest",
    monitoring:  bool          = False,
) -> List[str]:
    """
    Entry point for `truenorth self-host init` CLI command.
    Returns list of generated file paths.
    """
    cfg = SelfHostConfig(
        profile         = DeployProfile(profile),
        port            = port,
        domain          = domain,
        image_tag       = image_tag,
        with_monitoring = monitoring,
    )
    return cfg.generate(output_dir)