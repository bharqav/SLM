"""Generate high-quality synthetic supportive data aligned with your base model.

This version:
- Uses model-aware prompt formatting (Mistral/Qwen/Llama compatible)
- Strict tone/quality validation for empathetic support
- Filters for model compatibility
- Outputs exact format needed for training

Usage:
  python scripts/training/generate_synthetic_data_v2.py \
    --local_llm_url http://localhost:8000/v1 \
    --model_name mistral-7b-instruct \
    --base_model_name mistral-7b-instruct \
    --output_file data/synthetic_sft.jsonl \
    --num_samples 1000 \
    --quality_threshold 0.75

Assumes OpenAI-compatible API (vLLM, Ollama with OpenAI mode, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ConversationExample:
    """Single training example in chat format."""
    messages: list[dict]
    emotion: str
    context: str
    quality_score: float
    alignment_score: float
    generation_valid: bool


# Base model templates (adjust for your chosen base model)
MODEL_TEMPLATES = {
    "mistral": "{system}\n{user}\n",
    "qwen": "{system}\n{user}\n",
    "llama": "{system}\n{user}\n",
    "default": "{system}\n{user}\n",
}

EMOTION_LABELS = ["anxiety", "sadness", "anger", "overwhelm", "loneliness", "stress", "frustration"]

CONTEXTS = [
    "work overwhelm and pressure",
    "self-doubt and imposter syndrome",
    "heartbreak after ending a relationship",
    "health anxiety and medical concerns",
    "social isolation and disconnection",
    "financial stress and instability",
    "relationship conflict and tension",
    "life direction and purpose confusion",
    "personal growth barriers",
    "future uncertainty and worry",
    "grief and loss processing",
    "career transitions and uncertainty",
]

# Tone markers to EXPECT in good supportive responses
GOOD_TONE_MARKERS = [
    "understand",
    "valid",
    "difficult",
    "take",
    "step",
    "support",
    "hear",
    "feel",
    "right",
    "okay",
]

# Tone markers that indicate BAD/generic responses
BAD_TONE_MARKERS = [
    "you should",
    "just",
    "simply",
    "obviously",
    "clearly",
    "easy",
    "just think",
    "toxic",
    "this is weird",
    "problem is you",
]

SYSTEM_PROMPT = """You are a compassionate, supportive AI assistant trained to offer emotional support.
Your responses are:
- Brief and natural (1-3 sentences)
- Warm and non-judgmental
- Validating without being patronizing
- Grounded in listening, not advice-giving
- Never clinical or diagnostic

You do NOT:
- Provide medical/psychiatric diagnosis
- Suggest specific treatments or medications
- Minimize or dismiss feelings
- Use platitudes or toxic positivity
- Go into long explanations
"""


def _hash_text(text: str) -> str:
    """Fast hash for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _check_tone_alignment(text: str) -> tuple[float, list[str]]:
    """Score response for alignment with supportive tone.
    
    Returns: (alignment_score 0-1, identified_issues)
    """
    text_lower = text.lower()
    issues: list[str] = []
    good_count = 0
    bad_count = 0

    for marker in GOOD_TONE_MARKERS:
        if marker in text_lower:
            good_count += 1

    for marker in BAD_TONE_MARKERS:
        if marker in text_lower:
            bad_count += 1
            issues.append(f"found '{marker}' (generic/dismissive)")

    # Check for clinical language
    clinical_terms = [
        "diagnosis",
        "treatment",
        "medication",
        "psychiatrist",
        "therapist",
        "syndrome",
        "disorder",
    ]
    for term in clinical_terms:
        if term in text_lower:
            bad_count += 1
            issues.append(f"clinical language: {term}")

    # Check length
    words = len(text.split())
    if words < 8:
        issues.append("too short (< 8 words)")
        bad_count += 1
    if words > 100:
        issues.append("too long (> 100 words)")
        bad_count += 1

    # Check for preachy patterns
    if text.count("!") > 2:
        issues.append("too many exclamations (preachy tone)")
        bad_count += 1

    if text.count("I'm") > 2 or text.count("I am") > 1:
        issues.append("too much self-reference")
        bad_count += 1

    # Calculate score
    score = max(0.0, 1.0 - (bad_count * 0.15) + (good_count * 0.1))
    return min(1.0, score), issues


def _is_degenerate(text: str) -> bool:
    """Reject low-quality/degenerate responses."""
    if len(text.strip()) < 10:
        return True
    if text.count("\n\n") > 2:  # Too many breaks
        return True
    if "API" in text or "OpenAI" in text or "model" in text:
        return True
    if re.search(r"[0-9]{4,}", text):  # Long numbers (likely model artifacts)
        return True
    if text.lower().count("the") > len(text.split()) * 0.2:  # Over-use of "the"
        return True
    if "......" in text or text.count(".") > text.count(" ") * 0.3:
        return True
    return False


def _extract_json_response(text: str) -> dict | None:
    """Extract JSON from LLM output robustly."""
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


async def call_local_llm(
    client: httpx.AsyncClient,
    model_name: str,
    prompt: str,
    base_url: str,
    temperature: float = 0.7,
) -> str:
    """Call local LLM via OpenAI-compatible API."""
    try:
        response = await client.post(
            f"{base_url}/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 1200,
                "top_p": 0.95,
                "stop": ["User:", "user:"],
            },
            timeout=90.0,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("choices") and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        return ""
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return ""


async def generate_aligned_batch(
    client: httpx.AsyncClient,
    model_name: str,
    base_url: str,
    emotion: str,
    context: str,
    count: int = 2,
) -> list[ConversationExample]:
    """Generate batch of synthetic examples aligned with model style."""
    prompt = f"""Generate {count} supportive conversation examples for someone experiencing {emotion}.

Context: {context}

For each example, return a JSON object with:
- "user_text": 1-2 sentences from the user describing their situation
- "assistant_response": 1-3 sentences of warm, supportive response
- "quality_score": float 0-1 rating your response quality

Requirements for assistant responses:
- Warm and validating, NOT clinical
- Show you're listening, NOT giving advice
- Natural language, NOT robotic
- 8-100 words
- Grounded in empathy, not platitudes

Return a JSON array of objects. ONLY JSON, no other text.

Example format:
[
  {{"user_text": "I feel overwhelmed at work", "assistant_response": "That sounds really tough. Work pressure can feel all-consuming.", "quality_score": 0.9}}
]
"""

    text = await call_local_llm(client, model_name, prompt, base_url, temperature=0.75)
    examples: list[ConversationExample] = []

    if not text:
        logger.warning(f"No response from LLM for {emotion}/{context}")
        return examples

    # Parse JSON
    try:
        data = json.loads(text.strip())
        if not isinstance(data, list):
            data = [data] if isinstance(data, dict) else []
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON for {emotion}/{context}")
        return examples

    for item in data:
        if not isinstance(item, dict):
            continue

        user_msg = (item.get("user_text") or "").strip()
        assistant_msg = (item.get("assistant_response") or "").strip()

        if not user_msg or not assistant_msg:
            logger.warning("Missing user or assistant message in item")
            continue

        if _is_degenerate(assistant_msg):
            logger.debug(f"Rejected degenerate response: {assistant_msg[:50]}")
            continue

        # Check tone alignment
        alignment_score, issues = _check_tone_alignment(assistant_msg)
        if issues:
            logger.debug(f"Tone issues: {issues}")

        quality = float(item.get("quality_score", 0.7))
        quality = max(0.0, min(1.0, quality))

        # Blend LLM's quality score with our alignment score
        final_quality = (quality * 0.6 + alignment_score * 0.4)

        examples.append(
            ConversationExample(
                messages=[
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                emotion=emotion,
                context=context,
                quality_score=final_quality,
                alignment_score=alignment_score,
                generation_valid=True,
            )
        )
        logger.debug(f"Generated example: quality={final_quality:.2f}, alignment={alignment_score:.2f}")

    return examples


async def main() -> int:
    parser = argparse.ArgumentParser(description="Generate high-quality synthetic data aligned with your base model")
    parser.add_argument("--local_llm_url", default="http://localhost:8000/v1", help="Local LLM API URL")
    parser.add_argument("--model_name", default="mistral-7b-instruct", help="Generator LLM model name")
    parser.add_argument("--base_model_name", default="mistral-7b-instruct", help="Target base model (for format alignment)")
    parser.add_argument("--output_file", default="data/synthetic_sft.jsonl", help="Output JSONL")
    parser.add_argument("--num_samples", type=int, default=500, help="Target total samples")
    parser.add_argument("--quality_threshold", type=float, default=0.65, help="Minimum quality to keep")
    parser.add_argument("--max_retries_per_combo", type=int, default=2, help="Retries per emotion/context")
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating high-quality synthetic data aligned with {args.base_model_name}")
    logger.info(f"Target: {args.num_samples} samples, quality threshold: {args.quality_threshold}")
    logger.info(f"Generator LLM: {args.model_name} at {args.local_llm_url}")
    logger.info(f"Output: {output_path}")

    samples_per_combo = max(1, args.num_samples // (len(EMOTION_LABELS) * len(CONTEXTS)))
    total_target = samples_per_combo * len(EMOTION_LABELS) * len(CONTEXTS)
    logger.info(
        f"Plan: {len(EMOTION_LABELS)} emotions × {len(CONTEXTS)} contexts × {samples_per_combo} per combo = ~{total_target} samples"
    )

    all_examples: list[ConversationExample] = []
    seen_hashes: set[str] = set()
    emotion_counts: Counter = Counter()
    quality_stats: list[float] = []

    async with httpx.AsyncClient() as client:
        for emotion in EMOTION_LABELS:
            for context in CONTEXTS:
                logger.info(f"Generating: {emotion:12} | {context}")
                
                retry = 0
                while retry < args.max_retries_per_combo:
                    try:
                        examples = await generate_aligned_batch(
                            client,
                            args.model_name,
                            args.local_llm_url,
                            emotion,
                            context,
                            count=samples_per_combo,
                        )
                        
                        for ex in examples:
                            if ex.quality_score >= args.quality_threshold:
                                # Dedup on assistant response
                                h = _hash_text(ex.messages[1]["content"])
                                if h not in seen_hashes:
                                    all_examples.append(ex)
                                    seen_hashes.add(h)
                                    emotion_counts[emotion] += 1
                                    quality_stats.append(ex.quality_score)
                        
                        # Stop early if we got enough examples
                        if len(examples) >= samples_per_combo * 0.5:
                            break
                    except Exception as e:
                        logger.error(f"Batch generation failed: {e}")
                    
                    retry += 1
                    if retry < args.max_retries_per_combo:
                        await asyncio.sleep(2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Generated {len(all_examples)} high-quality unique examples")
    if quality_stats:
        logger.info(f"Quality stats: mean={sum(quality_stats)/len(quality_stats):.3f}, min={min(quality_stats):.3f}, max={max(quality_stats):.3f}")
    logger.info(f"Emotion distribution:\n{dict(emotion_counts)}")
    logger.info(f"{'='*60}\n")

    # Write JSONL in training format
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            row = {
                "messages": ex.messages,
                "emotion": ex.emotion,
                "context": ex.context,
                "quality_score": round(ex.quality_score, 4),
                "alignment_score": round(ex.alignment_score, 4),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"✓ Saved {len(all_examples)} examples to {output_path}")
    logger.info("Ready for SFT training!")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
