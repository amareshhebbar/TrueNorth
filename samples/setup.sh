set -e
SAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SAMPLES_DIR")"

echo ""
echo "Samples dir : $SAMPLES_DIR"
echo "Repo root   : $ROOT_DIR"
echo ""

echo "── Python core ──────────────────────────────────────────"
cd "$ROOT_DIR/packages/core"
pip install -e . -q
pip install anthropic fastapi uvicorn flask httpx -q
echo "  ✓ done"
echo ""

echo "── Node.js SDK ──────────────────────────────────────────"
cd "$ROOT_DIR/packages/sdk-node"
npm install -q
npm run build 2>/dev/null || echo "  (build skipped)"
echo "  ✓ done"
echo ""

TSCONFIG='{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "types": ["node"]
  },
  "include": ["./*.ts", "./*.tsx"],
  "exclude": ["node_modules", "dist"]
}'

write_pkg() {
  local dir="$1" name="$2" deps="$3" main="${4:-app.ts}"
  mkdir -p "$dir"
  cat > "$dir/package.json" << PKG
{
  "name": "$name",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev":   "ts-node $main",
    "build": "tsc",
    "start": "node dist/$(basename $main .ts).js"
  },
  "dependencies": $deps,
  "devDependencies": {
    "@types/node": "^20.0.0",
    "ts-node":     "^10.9.0",
    "typescript":  "^5.3.0"
  }
}
PKG
  printf '%s' "$TSCONFIG" > "$dir/tsconfig.json"
}

echo "── Node.js package.json ─────────────────────────────────"
ND="$SAMPLES_DIR/nodejs"
E='{ "express": "^4.18.0", "axios": "^1.6.0", "@types/express": "^4.17.21" }'
P='{}'
TG='{ "node-telegram-bot-api": "^0.66.0", "@types/node-telegram-bot-api": "^0.66.0" }'

write_pkg "$ND/cliniqflow"            cliniqflow-ts            "$E"
write_pkg "$ND/cost-tracking"         cost-tracking-ts         "$P"
write_pkg "$ND/goal-chaining"         goal-chaining-ts         "$P"
write_pkg "$ND/hireflow"              hireflow-ts              "$E"
write_pkg "$ND/legal-aid-cli"         legal-aid-cli-ts         "$P"
write_pkg "$ND/multilingual-demo"     multilingual-demo-ts     "$P"
write_pkg "$ND/my-demo"               my-demo-ts               "$P"
write_pkg "$ND/restaurant-feedback"   restaurant-feedback-ts   "$E"
write_pkg "$ND/scholarship-web"       scholarship-web-ts       "$E"
write_pkg "$ND/telegram-career-coach" telegram-career-ts       "$TG" "bot.ts"

echo "  ✓ package.json + tsconfig.json written for all"
echo ""

echo "── npm install (all Node.js projects) ───────────────────"
for dir in "$ND"/*/; do
  [ -f "$dir/package.json" ] || continue
  name=$(basename "$dir")
  (cd "$dir" && npm install -q 2>/dev/null)
  echo "  ✓ $name"
done
echo ""

echo "── go.mod (all Go projects) ─────────────────────────────"
GD="$SAMPLES_DIR/go-lang"
SDK="$ROOT_DIR/packages/sdk-go"

write_gomod() {
  local dir="$1" name="$2" extra="${3:-}"
  mkdir -p "$dir"
  cat > "$dir/go.mod" << GOMOD
module github.com/truenorth-ai/$name

go 1.22

require (
	github.com/truenorth-ai/truenorth-go v0.1.0
$([ -n "$extra" ] && printf '\t%s\n' "$extra")
)

replace github.com/truenorth-ai/truenorth-go => $SDK
GOMOD
}

write_gomod "$GD/cliniqflow"          cliniqflow          "github.com/gin-gonic/gin v1.9.1"
write_gomod "$GD/cost-tracking"       cost-tracking
write_gomod "$GD/farm_advisory"       farm_advisory
write_gomod "$GD/goal-chaining"       goal-chaining
write_gomod "$GD/hireflow"            hireflow            "github.com/gin-gonic/gin v1.9.1"
write_gomod "$GD/hr-screener"         hr-screener         "github.com/gin-gonic/gin v1.9.1"
write_gomod "$GD/legal-aid-cli"       legal-aid-cli
write_gomod "$GD/multilingual-demo"   multilingual-demo
write_gomod "$GD/my-demo"             my-demo
write_gomod "$GD/restaurant-feedback" restaurant-feedback

echo "  ✓ go.mod written for all"
echo ""

echo "── go mod tidy (all Go projects) ────────────────────────"
for dir in "$GD"/*/; do
  [ -f "$dir/go.mod" ] || continue
  name=$(basename "$dir")
  (cd "$dir" && go mod tidy -q 2>/dev/null) && echo "  ✓ $name" || echo "  ⚠ $name"
done
echo ""

echo "══════════════════════════════════════════════════════════"
echo "  Done! Correct paths:"
echo "  Samples : $SAMPLES_DIR"
echo "  Root    : $ROOT_DIR"
echo "  Go SDK  : $SDK"
echo "══════════════════════════════════════════════════════════"
