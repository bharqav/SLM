# Background jobs

Chat requests should stay **low-latency**; heavy work belongs here.

## Queues (suggested)

| Queue | Work |
|-------|------|
| `embed` | Chunk text → embedding → `memory_chunks` |
| `safety` | Re-run classifiers on stored messages |
| `analytics` | Rollups, funnels |
| `eval` | Offline eval batches |

## Example: RQ

```bash
pip install rq
rq worker embed safety --url redis://localhost:6379/0
```

Enqueue from API after `commit()`:

```python
from redis import Redis
from rq import Queue
q = Queue("embed", connection=Redis.from_url(settings.REDIS_URL))
q.enqueue("backend.jobs.tasks.embed_memory_job", str(user_id), str(conversation_id), text[:2000])
```

## Temporal / Celery

For longer workflows (multi-step, retries, schedules), prefer **Temporal** or **Celery** with Redis/Rabbit broker and explicit idempotency keys.
