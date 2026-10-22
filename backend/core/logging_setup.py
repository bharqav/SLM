"""JSON logging when LOG_JSON=true (PRD: structured logs for ELK-style ingestion)."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from backend.core.request_context import request_id_ctx


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(*, use_json: bool) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    if use_json:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(levelname)s %(name)s [%(request_id)s] %(message)s"),
        )

        class RequestIdFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                record.request_id = request_id_ctx.get()
                return True

        handler.addFilter(RequestIdFilter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
