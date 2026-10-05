import os
import json
from typing import Dict, Any


def _gemini_client():
    """
    Create a Gemini client when GEMINI_API_KEY is configured.
    Returns None when AI augmentation is unavailable.
    """
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return None

    from google import genai
    return genai.Client(api_key=api_key)


def generate_ai_profile(text_corpus: str) -> Dict[str, Any]:
    """
    Generate an AI linguistic profile based only on observable
    linguistic characteristics.

    AI augmentation is optional. Classical Persona analysis must
    continue when Gemini is unavailable.
    """
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return {
            "status": "NOT_CONFIGURED",
            "provider": "gemini",
            "message": (
                "GEMINI_API_KEY not set. "
                "Classical analysis continues without AI augmentation."
            ),
        }

    try:
        from google.genai import types

        client = _gemini_client()

        prompt = f"""
Analyze the following text corpus and identify OBSERVABLE LINGUISTIC
characteristics.

Do NOT provide:
- psychological diagnosis
- personality types
- statements about motivation
- claims about real-world identity
- claims that the author is a particular person

Focus strictly on observable writing characteristics.

Return a JSON object with this schema:

- communication_style: string
- formality: string
- technical_language: list of strings
- recurring_terms: list of strings
- slang: list of strings
- language_hints: list of strings
- writing_patterns: list of strings
- observable_opsec_language: list of strings
- topic_preferences: list of strings
- summary: string

TEXT CORPUS:
{text_corpus[:30000]}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        if not response.text:
            raise ValueError("Empty response from Gemini.")

        result = json.loads(response.text)

        result["status"] = "SUCCESS"
        result["provider"] = "gemini"

        return result

    except Exception as exc:
        return {
            "status": "ERROR",
            "provider": "gemini",
            "message": str(exc),
        }


def compare_ai_profiles(
    profile_a: Dict[str, Any],
    profile_b: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two already-generated AI linguistic profiles.

    This does NOT produce an identity probability.

    It identifies observable linguistic characteristics shared by
    the two profiles and returns an explainable AI comparison.
    """

    ai_a = profile_a.get("ai_profile") or {}
    ai_b = profile_b.get("ai_profile") or {}

    status_a = ai_a.get("status")
    status_b = ai_b.get("status")

    if status_a != "SUCCESS" or status_b != "SUCCESS":
        return {
            "status": "NOT_AVAILABLE",
            "provider": "gemini",
            "message": (
                "AI linguistic comparison requires successful "
                "AI profiles for both personas."
            ),
            "reference_status": status_a or "UNKNOWN",
            "candidate_status": status_b or "UNKNOWN",
            "shared_characteristics": [],
            "explanation": (
                "AI comparison was not used because one or both "
                "linguistic profiles were unavailable."
            ),
        }

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return {
            "status": "NOT_CONFIGURED",
            "provider": "gemini",
            "message": "GEMINI_API_KEY not configured.",
            "shared_characteristics": [],
            "explanation": (
                "AI linguistic comparison is unavailable. "
                "Classical and semantic analysis remain independent."
            ),
        }

    try:
        from google.genai import types

        client = _gemini_client()

        comparison_input = {
            "reference": {
                "communication_style": ai_a.get("communication_style"),
                "formality": ai_a.get("formality"),
                "technical_language": ai_a.get("technical_language", []),
                "recurring_terms": ai_a.get("recurring_terms", []),
                "slang": ai_a.get("slang", []),
                "language_hints": ai_a.get("language_hints", []),
                "writing_patterns": ai_a.get("writing_patterns", []),
                "observable_opsec_language": ai_a.get(
                    "observable_opsec_language", []
                ),
                "topic_preferences": ai_a.get(
                    "topic_preferences", []
                ),
            },
            "candidate": {
                "communication_style": ai_b.get("communication_style"),
                "formality": ai_b.get("formality"),
                "technical_language": ai_b.get("technical_language", []),
                "recurring_terms": ai_b.get("recurring_terms", []),
                "slang": ai_b.get("slang", []),
                "language_hints": ai_b.get("language_hints", []),
                "writing_patterns": ai_b.get("writing_patterns", []),
                "observable_opsec_language": ai_b.get(
                    "observable_opsec_language", []
                ),
                "topic_preferences": ai_b.get(
                    "topic_preferences", []
                ),
            },
        }

        prompt = f"""
Compare these two AI-generated linguistic profiles.

Identify ONLY observable similarities and differences.

Do NOT:
- identify a real person
- claim the accounts belong to the same person
- provide an identity probability
- make psychological claims
- infer motivation

Return JSON with:

- shared_characteristics: list of strings
- differing_characteristics: list of strings
- shared_technical_language: list of strings
- shared_writing_patterns: list of strings
- shared_opsec_language: list of strings
- communication_style_match: boolean
- formality_match: boolean
- explanation: string

Profiles:

{json.dumps(comparison_input, ensure_ascii=False)}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        if not response.text:
            raise ValueError("Empty AI comparison response.")

        result = json.loads(response.text)

        result["status"] = "SUCCESS"
        result["provider"] = "gemini"

        return result

    except Exception as exc:
        return {
            "status": "ERROR",
            "provider": "gemini",
            "message": str(exc),
            "shared_characteristics": [],
            "explanation": (
                "AI linguistic comparison failed. "
                "Other Persona signals remain independent."
            ),
        }
