//! truenorth — official Rust SDK
//!
//! # Install
//!
//! ```toml
//! [dependencies]
//! truenorth = "0.1"
//! tokio = { version = "1", features = ["full"] }
//! ```
//!
//! # Usage
//!
//! ```rust
//! use truenorth::TrueNorth;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let tn = TrueNorth::new("tn_live_...", "http://localhost:8000");
//!
//!     let session = tn.sessions().create("fitness-coach", None).await?;
//!     let result  = tn.sessions().message(&session.id, "I am 28").await?;
//!     let output  = tn.sessions().output(&session.id).await?;
//!
//!     println!("{:?}", output.content);
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use reqwest::{Client, StatusCode};
use thiserror::Error;

// ─── Error types ─────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum TrueNorthError {
    #[error("API error ({status_code}): {error_code} — {message}")]
    Api {
        status_code: u16,
        error_code:  String,
        message:     String,
    },
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, TrueNorthError>;

// ─── Response types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct Session {
    #[serde(rename = "session_id")]
    pub id:                String,
    #[serde(rename = "goal_id")]
    pub goal_id:           String,
    pub status:            String,
    #[serde(rename = "current_turn", default)]
    pub current_turn:      u32,
    #[serde(rename = "completion_pct")]
    pub completion_pct:    f64,
    #[serde(rename = "collected_fields", default)]
    pub collected_fields:  HashMap<String, serde_json::Value>,
    #[serde(rename = "missing_required", default)]
    pub missing_required:  Vec<String>,
    #[serde(rename = "total_cost_usd", default)]
    pub total_cost_usd:    f64,
    #[serde(rename = "is_complete", default)]
    pub is_complete:       bool,
    #[serde(rename = "agent_message", default)]
    pub agent_message:     String,
    #[serde(rename = "detected_language")]
    pub detected_language: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MessageResult {
    #[serde(rename = "session_id")]
    pub session_id:       String,
    pub turn:             u32,
    #[serde(default)]
    pub text:             String,
    #[serde(rename = "is_complete")]
    pub is_complete:      bool,
    #[serde(rename = "completion_pct")]
    pub completion_pct:   f64,
    #[serde(rename = "cost_usd")]
    pub cost_usd:         f64,
    #[serde(rename = "latency_ms")]
    pub latency_ms:       u32,
    pub output:           Option<Output>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Output {
    #[serde(rename = "session_id")]
    pub session_id:   String,
    #[serde(rename = "goal_id", default)]
    pub goal_id:      String,
    #[serde(default = "default_format")]
    pub format:       String,
    pub content:      Option<serde_json::Value>,
    #[serde(default)]
    pub fields:       HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub metadata:     HashMap<String, serde_json::Value>,
}

fn default_format() -> String { "json".to_string() }

#[derive(Debug, Clone, Deserialize)]
pub struct Goal {
    pub name:        String,
    pub version:     String,
    pub description: String,
    pub sector:      String,
    pub tags:        Vec<String>,
    pub downloads:   u32,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    error:   String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct OutputWrapper {
    output:     Option<Output>,
    session_id: String,
}

// ─── HTTP transport ───────────────────────────────────────────────────────────

#[derive(Clone)]
struct Transport {
    base_url: String,
    api_key:  String,
    client:   Client,
}

impl Transport {
    fn new(base_url: &str, api_key: &str, timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to build HTTP client");
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key:  api_key.to_string(),
            client,
        }
    }

    async fn get<T: for<'de> Deserialize<'de>>(&self, path: &str, params: Option<Vec<(&str, &str)>>) -> Result<T> {
        let mut url = format!("{}{}", self.base_url, path);
        if let Some(p) = params {
            let qs: Vec<String> = p.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
            url = format!("{}?{}", url, qs.join("&"));
        }
        let resp = self.client.get(&url)
            .header("X-TrueNorth-Key", &self.api_key)
            .header("Accept", "application/json")
            .send().await?;
        self.parse_response(resp).await
    }

    async fn post<T: for<'de> Deserialize<'de>>(&self, path: &str, body: &serde_json::Value) -> Result<T> {
        let resp = self.client.post(format!("{}{}", self.base_url, path))
            .header("X-TrueNorth-Key", &self.api_key)
            .header("Content-Type",    "application/json")
            .json(body)
            .send().await?;
        self.parse_response(resp).await
    }

    async fn delete(&self, path: &str) -> Result<()> {
        let resp = self.client.delete(format!("{}{}", self.base_url, path))
            .header("X-TrueNorth-Key", &self.api_key)
            .send().await?;
        if resp.status().is_success() || resp.status() == StatusCode::NO_CONTENT {
            return Ok(());
        }
        let status = resp.status().as_u16();
        let body: ApiError = resp.json().await.unwrap_or(ApiError { error: "unknown".into(), message: String::new() });
        Err(TrueNorthError::Api { status_code: status, error_code: body.error, message: body.message })
    }

    async fn parse_response<T: for<'de> Deserialize<'de>>(&self, resp: reqwest::Response) -> Result<T> {
        let status = resp.status();
        if !status.is_success() {
            let code = status.as_u16();
            let body: ApiError = resp.json().await.unwrap_or(ApiError { error: "http_error".into(), message: format!("HTTP {}", code) });
            return Err(TrueNorthError::Api { status_code: code, error_code: body.error, message: body.message });
        }
        Ok(resp.json::<T>().await?)
    }
}

// ─── Resource clients ─────────────────────────────────────────────────────────

pub struct SessionsClient {
    t: Transport,
}

impl SessionsClient {
    pub async fn create(&self, goal_id: &str, opts: Option<CreateSessionOptions>) -> Result<Session> {
        let mut body = serde_json::json!({ "goal_id": goal_id });
        if let Some(o) = opts {
            if let Some(uid)    = o.user_id    { body["user_id"]    = uid.into(); }
            if let Some(sid)    = o.session_id { body["session_id"] = sid.into(); }
            if let Some(budget) = o.budget_usd { body["budget_usd"] = budget.into(); }
            if let Some(lang)   = o.language   { body["language"]   = lang.into(); }
        }
        self.t.post("/sessions", &body).await
    }

    pub async fn message(&self, session_id: &str, text: &str) -> Result<MessageResult> {
        self.t.post(&format!("/sessions/{}/message", session_id),
            &serde_json::json!({ "text": text })).await
    }

    pub async fn get(&self, session_id: &str) -> Result<Session> {
        self.t.get(&format!("/sessions/{}", session_id), None).await
    }

    pub async fn output(&self, session_id: &str) -> Result<Output> {
        let wrapper: OutputWrapper = self.t.get(
            &format!("/sessions/{}/output", session_id), None
        ).await?;
        wrapper.output.ok_or_else(|| TrueNorthError::Api {
            status_code: 409,
            error_code:  "not_complete".into(),
            message:     "Session not yet complete".into(),
        })
    }

    pub async fn force_output(&self, session_id: &str) -> Result<Output> {
        let wrapper: OutputWrapper = self.t.post(
            &format!("/sessions/{}/force-output", session_id),
            &serde_json::json!({}),
        ).await?;
        wrapper.output.ok_or_else(|| TrueNorthError::Api {
            status_code: 500,
            error_code:  "no_output".into(),
            message:     "No output in response".into(),
        })
    }

    pub async fn end(&self, session_id: &str) -> Result<()> {
        self.t.delete(&format!("/sessions/{}", session_id)).await
    }
}

pub struct GoalsClient {
    t: Transport,
}

impl GoalsClient {
    pub async fn list(&self, query: Option<&str>, sector: Option<&str>) -> Result<Vec<Goal>> {
        let mut params = vec![];
        if let Some(q) = query  { params.push(("q",      q));      }
        if let Some(s) = sector { params.push(("sector", s));      }
        self.t.get("/goals", Some(params)).await
    }

    pub async fn install(&self, name: &str, version: &str) -> Result<Goal> {
        self.t.post(&format!("/goals/{}/install", name),
            &serde_json::json!({ "version": version })).await
    }
}

// ─── Options ─────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct CreateSessionOptions {
    pub user_id:    Option<String>,
    pub session_id: Option<String>,
    pub budget_usd: Option<f64>,
    pub language:   Option<String>,
    pub seed_fields: Option<HashMap<String, serde_json::Value>>,
}

pub struct TrueNorth {
    t: Transport,
}

impl TrueNorth {
    pub fn new(api_key: &str, base_url: &str) -> Self {
        Self { t: Transport::new(base_url, api_key, Duration::from_secs(60)) }
    }

    pub fn from_env() -> Self {
        let key     = std::env::var("TRUENORTH_API_KEY").unwrap_or_default();
        let base    = std::env::var("TRUENORTH_BASE_URL")
                        .unwrap_or_else(|_| "http://localhost:8000".into());
        Self::new(&key, &base)
    }

    pub fn sessions(&self) -> SessionsClient {
        SessionsClient { t: self.t.clone() }
    }

    pub fn goals(&self) -> GoalsClient {
        GoalsClient { t: self.t.clone() }
    }

    pub async fn health(&self) -> Result<serde_json::Value> {
        self.t.get("/health", None).await
    }
}


pub async fn run_session(
    goal_id:  &str,
    messages: &[&str],
    tn:       &TrueNorth,
) -> Result<Output> {
    let sessions = tn.sessions();
    let session  = sessions.create(goal_id, None).await?;
    for &msg in messages {
        let result = sessions.message(&session.id, msg).await?;
        if result.is_complete {
            if let Some(out) = result.output {
                return Ok(out);
            }
        }
    }
    sessions.force_output(&session.id).await
}