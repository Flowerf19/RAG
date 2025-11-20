"""
Prompt-building utilities for the Query Enhancement Module.
"""

from __future__ import annotations

from typing import Mapping

LANGUAGE_NAME_OVERRIDES = {
    "vi": "Vietnamese",
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "th": "Thai",
    "id": "Indonesian",
    "pt": "Portuguese",
}


def _describe_language(language_code: str | None) -> str:
    if not language_code:
        return "Unknown"
    normalized = language_code.strip().lower()
    if not normalized or normalized == "unknown":
        return "Unknown"
    return LANGUAGE_NAME_OVERRIDES.get(normalized, normalized)


def build_prompt(
    user_query: str,
    language_requirements: Mapping[str, int],
    *,
    additional_instructions: str | None = None,
    input_language: str | None = "unknown",
) -> str:
    """
    Build the instruction prompt supplied to the LLM for query enhancement.
    """
    total_variants = sum(max(count, 0) for count in language_requirements.values())
    if total_variants <= 0:
        total_variants = 1

    language_lines = []
    for lang_code, count in language_requirements.items():
        if count <= 0:
            continue

        if lang_code.lower() == "vi":
            description = "Vietnamese"
        elif lang_code.lower() == "en":
            description = "English"
        else:
            description = lang_code
        language_lines.append(f"- Generate {count} variants in {description}.")

    language_section = "\n".join(language_lines) if language_lines else "- Use the language that best improves recall."

    extras = additional_instructions.strip() if additional_instructions else ""
    language_name = _describe_language(input_language)
    context_label = (
        "the original language context"
        if language_name == "Unknown"
        else f"the {language_name} context"
    )

    prompt = f"""
You are a retrieval optimisation assistant. Given a user's search query, produce paraphrased
or translated alternatives that improve recall for a hybrid FAISS + BM25 retrieval system.

Base query:
\"\"\"{user_query.strip()}\"\"\"
Original query language: {language_name}.

Requirements:
- Produce exactly {total_variants} alternative queries.
- Maintain the original intent; avoid hallucinating new topics.
- Keep outputs concise (max 25 words each).
- Return results as a JSON array of strings, without additional commentary.
- Respect the cultural and semantic nuances of {context_label} when composing each variant.
{language_section}
{extras}
"""
    return "\n".join(line.rstrip() for line in prompt.strip().splitlines())
