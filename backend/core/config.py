from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = "development"
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379/0"
    # Empty in dev → per-process random secret generated at startup (with a warning).
    # Production startup fails fast unless a strong secret is configured (main.py).
    JWT_SECRET: str = ""
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    HF_API_TOKEN: str | None = None
    HF_MODEL_URL: str | None = None
    # Managed inference: same HF-compatible HTTP contract; set for primary + optional chain
    HF_FALLBACK_MODEL_URL: str | None = None
    HF_FALLBACK_MODEL_NAME: str = "fallback"
    HF_CANARY_MODEL_URL: str | None = None
    HF_CANARY_MODEL_NAME: str = "canary"
    HF_FAST_MODEL_URL: str | None = None
    HF_FAST_MODEL_NAME: str = "fast"
    HF_QUALITY_MODEL_URL: str | None = None
    HF_QUALITY_MODEL_NAME: str = "quality"
    INFERENCE_TIMEOUT_SECONDS: float = 30.0
    INFERENCE_MAX_ATTEMPTS: int = 2
    GATEWAY_CB_COOLDOWN_SECONDS: int = 60
    GATEWAY_POLICY_VERSION: str = "1"
    # Canary: percent of users (hash-stable) routed to HF_CANARY_MODEL_URL when set
    CANARY_TRAFFIC_PERCENT: int = 0
    INFERENCE_PROVIDER: str = "stub"
    INFERENCE_MODEL_NAME: str = "emotional-slm-stub"
    CHAT_HISTORY_LIMIT: int = 16
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    # Stricter per-IP budget for credential endpoints (brute-force mitigation)
    LOGIN_RATE_LIMIT_PER_MINUTE: int = 10
    # Trust X-Forwarded-For for client IP behind a proxy/LB (only when deployed behind one)
    TRUST_PROXY_X_FORWARDED_FOR: bool = False
    # WebSocket message cap (chars) — WS payloads bypass pydantic body validation
    WS_MESSAGE_MAX_CHARS: int = 32000
    # Safety (Phase 2 — URLs optional)
    # fail_closed=True: when a configured safety provider errors, block (score 0) instead of passing.
    # Local heuristic backstops (regex) always run regardless of provider availability.
    SAFETY_FAIL_CLOSED: bool = True
    SAFETY_INPUT_THRESHOLD: float = 0.5
    SAFETY_OUTPUT_THRESHOLD: float = 0.5
    SAFETY_INPUT_HF_URL: str | None = None
    SAFETY_OUTPUT_HF_URL: str | None = None
    # Emotion — HF text-classification router (optional)
    EMOTION_HF_URL: str | None = None
    EMOTION_HF_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"
    EMOTION_MIN_CONFIDENCE: float = 0.60
    EMOTION_ENGLISH_ONLY: bool = True
    # Memory / embeddings (Phase 4)
    MEMORY_EMBEDDING_DIM: int = 384
    MEMORY_ENABLED: bool = False
    MEMORY_PI_REDACT: bool = True
    MEMORY_CHUNK_TTL_DAYS: int = 365
    HF_EMBEDDING_URL: str | None = None
    # Observability (Phase 5)
    SENTRY_DSN: str | None = None
    LOG_JSON: bool = False
    # First account with this email becomes staff (dev/bootstrap only)
    ADMIN_BOOTSTRAP_EMAIL: str | None = None
    # CORS: comma-separated origins (dev default: local frontend only; wildcard rejected in prod)
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    # Optional bearer token required to scrape /metrics (unset = open scrape endpoint)
    METRICS_BEARER_TOKEN: str | None = None
    # Emotion cache (PRD: Redis <20ms)
    EMOTION_CACHE_ENABLED: bool = True
    EMOTION_CACHE_TTL_SECONDS: int = 300
    # OAuth2 roadmap (PRD: JWT + OAuth2) — full flow not bundled; see /api/v1/auth/oauth/*
    GOOGLE_OAUTH_CLIENT_ID: str | None = None
    GOOGLE_OAUTH_CLIENT_SECRET: str | None = None
    # Crisis / governance
    CRISIS_HUMAN_REVIEW_ENABLED: bool = False
    # Token budgets (0 = disabled)
    MAX_USER_MESSAGE_TOKENS_ESTIMATE: int = 8000
    MAX_COMPLETION_TOKENS: int = 256
    MIN_COMPLETION_TOKENS: int = 64
    INFERENCE_TEMPERATURE: float = 0.7
    INFERENCE_TOP_P: float = 0.9
    INFERENCE_REPETITION_PENALTY: float = 1.08
    TOKEN_BUDGET_PER_USER_DAILY: int = 0
    # Idempotency
    IDEMPOTENCY_TTL_SECONDS: int = 86400
    # Model / prompt registry (governance)
    PROMPT_VERSION: str = "1"
    SAFETY_RULESET_VERSION: str = "1"
    # Degraded inference
    DEGRADED_MODE: bool = False
    # Local PyTorch artifact tier (opt-in): serve slm-training/best_model.pt in-process
    LOCAL_TORCH_MODEL_ENABLED: bool = False
    LOCAL_TORCH_MODEL_PATH: str | None = None
    # Locally-served fine-tune tier (opt-in): URL of slm-ft/serve_model.py /generate host
    SERVED_MODEL_URL: str | None = None
    SERVE_MODEL_TOKEN: str | None = None
    # 24GB-oriented routing: send short/simple turns to fast model when configured
    ROUTER_SIMPLE_TURN_MAX_CHARS: int = 220
    ROUTER_SIMPLE_TURN_MAX_LINES: int = 3
    ROUTER_COMPLEX_TURN_MIN_CHARS: int = 900
    ROUTER_COMPLEX_TURN_MIN_LINES: int = 12
    # Safety multi-label thresholds
    SAFETY_SELF_HARM_THRESHOLD: float = 0.50
    SAFETY_VIOLENCE_THRESHOLD: float = 0.60
    SAFETY_ABUSE_THRESHOLD: float = 0.60
    SAFETY_MEDICAL_EMERGENCY_THRESHOLD: float = 0.55
    SAFETY_PROMPT_INJECTION_THRESHOLD: float = 0.70
    # Retrieval blend: semantic similarity + recency
    MEMORY_RECENCY_WEIGHT: float = 0.25
    MEMORY_SIMILARITY_WEIGHT: float = 0.75


settings = Settings()
