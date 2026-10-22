from collections.abc import AsyncIterator

from uuid import UUID



from backend.services.model_gateway import gateway as model_gateway





def estimate_tokens(text: str) -> int:

    return max(1, len(text) // 4)





def inference_meta() -> tuple[str, str]:

    t = model_gateway.get_last_gateway_trace()

    if t:

        return t.provider, t.model_id

    from backend.core.config import settings



    if settings.HF_API_TOKEN and settings.HF_MODEL_URL:

        return "huggingface", settings.INFERENCE_MODEL_NAME

    return settings.INFERENCE_PROVIDER, settings.INFERENCE_MODEL_NAME





async def complete_stream(prompt: str, user_id: UUID | None = None) -> AsyncIterator[str]:

    async for chunk in model_gateway.generate_stream(prompt, user_id=user_id):

        yield chunk





async def complete_full(prompt: str, user_id: UUID | None = None) -> str:

    res = await model_gateway.generate_full(prompt, user_id=user_id)

    return res.text.strip()


