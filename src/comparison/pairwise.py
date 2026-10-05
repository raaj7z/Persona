import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def vector_similarity(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0

    keys = sorted(set(a) | set(b))

    va = np.array(
        [float(a.get(k, 0)) for k in keys]
    ).reshape(1, -1)

    vb = np.array(
        [float(b.get(k, 0)) for k in keys]
    ).reshape(1, -1)

    if not np.any(va) or not np.any(vb):
        return 0.0

    return float(
        cosine_similarity(va, vb)[0][0]
    )


def _find_shared_features(
    profile_a: dict,
    profile_b: dict,
) -> dict:
    """Extract shared observable features."""

    shared = {
        "character_ngrams": [],
        "function_words": [],
        "punctuation": [],
        "lexical_patterns": [],
        "hinglish_phrases": [],
    }

    style_a = profile_a.get("stylometry", {})
    style_b = profile_b.get("stylometry", {})

    ngrams_a = set(
        style_a.get("character_ngrams", {}).keys()
    )
    ngrams_b = set(
        style_b.get("character_ngrams", {}).keys()
    )

    shared["character_ngrams"] = sorted(
        list(ngrams_a.intersection(ngrams_b))
    )[:15]

    fw_a = set(
        style_a.get("function_words", {}).keys()
    )
    fw_b = set(
        style_b.get("function_words", {}).keys()
    )

    shared["function_words"] = sorted(
        list(fw_a.intersection(fw_b))
    )[:15]

    punc_a = set(
        style_a.get("punctuation", {}).keys()
    )
    punc_b = set(
        style_b.get("punctuation", {}).keys()
    )

    shared["punctuation"] = sorted(
        list(punc_a.intersection(punc_b))
    )[:10]

    lex_a = set(
        style_a.get("lexical", {}).keys()
    )
    lex_b = set(
        style_b.get("lexical", {}).keys()
    )

    shared["lexical_patterns"] = sorted(
        list(lex_a.intersection(lex_b))
    )[:15]

    return shared


def compare_profiles(
    profile_a: dict,
    profile_b: dict,
) -> dict:

    # ------------------------------------------------------------
    # 1. Classical stylometry
    # ------------------------------------------------------------

    style_a_fw = profile_a.get(
        "stylometry", {}
    ).get(
        "function_words", {}
    )

    style_b_fw = profile_b.get(
        "stylometry", {}
    ).get(
        "function_words", {}
    )

    fw_sim = vector_similarity(
        style_a_fw,
        style_b_fw,
    )

    style_a_ng = profile_a.get(
        "stylometry", {}
    ).get(
        "character_ngrams", {}
    )

    style_b_ng = profile_b.get(
        "stylometry", {}
    ).get(
        "character_ngrams", {}
    )

    ng_sim = vector_similarity(
        style_a_ng,
        style_b_ng,
    )

    style_sim = (
        fw_sim * 0.5 +
        ng_sim * 0.5
    ) if (fw_sim or ng_sim) else 0.0

    # ------------------------------------------------------------
    # 2. Semantic embedding similarity
    # ------------------------------------------------------------

    sem_a = (
        profile_a.get("semantic", {}).get(
            "corpus_centroid"
        )
        or
        profile_a.get("semantic", {}).get(
            "centroid_embedding",
            [],
        )
    )

    sem_b = (
        profile_b.get("semantic", {}).get(
            "corpus_centroid"
        )
        or
        profile_b.get("semantic", {}).get(
            "centroid_embedding",
            [],
        )
    )

    semantic_sim = 0.0

    if sem_a and sem_b:
        va = np.array(sem_a).reshape(1, -1)
        vb = np.array(sem_b).reshape(1, -1)

        if np.any(va) and np.any(vb):
            semantic_sim = float(
                cosine_similarity(
                    va,
                    vb,
                )[0][0]
            )

    # ------------------------------------------------------------
    # 3. Topic similarity
    # ------------------------------------------------------------

    from ..semantic.topics import topic_overlap
    from ..fusion.engine import fuse_signals

    topics_a = profile_a.get(
        "semantic", {}
    ).get(
        "topics",
        {},
    )

    topics_b = profile_b.get(
        "semantic", {}
    ).get(
        "topics",
        {},
    )

    topic_overlap_res = topic_overlap(
        topics_a,
        topics_b,
    )

    topic_sim = topic_overlap_res.get(
        "topic_similarity",
        0.0,
    )

    # ------------------------------------------------------------
    # 4. Behaviour + temporal association
    # ------------------------------------------------------------

    beh_a = profile_a.get(
        "behavior", {}
    ).get(
        "temporal", {}
    ).get(
        "posting_hours",
        {},
    )

    beh_b = profile_b.get(
        "behavior", {}
    ).get(
        "temporal", {}
    ).get(
        "posting_hours",
        {},
    )

    beh_sim = (
        vector_similarity(
            beh_a,
            beh_b,
        )
        if beh_a and beh_b
        else 0.0
    )

    temp_a = profile_a.get(
        "behavior", {}
    ).get(
        "temporal",
        {},
    )

    temp_b = profile_b.get(
        "behavior", {}
    ).get(
        "temporal",
        {},
    )

    temporal_sim = 0.0

    if temp_a and temp_b:
        hours_a = set(
            temp_a.get(
                "posting_hours",
                {},
            ).keys()
        )

        hours_b = set(
            temp_b.get(
                "posting_hours",
                {},
            ).keys()
        )

        if hours_a and hours_b:
            shared_h = hours_a.intersection(
                hours_b
            )

            union_h = hours_a.union(
                hours_b
            )

            temporal_sim = float(
                len(shared_h) /
                len(union_h)
            ) if union_h else 0.0

    # ------------------------------------------------------------
    # 5. Language
    # ------------------------------------------------------------

    lang_a = (
        profile_a.get(
            "corpus_quality",
            {},
        ).get(
            "language"
        )
        or "en"
    )

    lang_b = (
        profile_b.get(
            "corpus_quality",
            {},
        ).get(
            "language"
        )
        or "en"
    )

    lang_sim = (
        1.0
        if lang_a == lang_b
        else 0.5
    )

    # ------------------------------------------------------------
    # 6. Shared classical features
    # ------------------------------------------------------------

    shared_features = _find_shared_features(
        profile_a,
        profile_b,
    )

    # ------------------------------------------------------------
    # 7. Existing explainable fusion
    # ------------------------------------------------------------

    signals = {
        "style_similarity": style_sim,
        "semantic_similarity": semantic_sim,
        "behavioral_association": beh_sim,
        "topic_similarity": topic_sim,
        "temporal_association": temporal_sim,
    }

    fusion_result = fuse_signals(
        signals
    )

    assessment = (
        "INVESTIGATIVE LEAD"
        if fusion_result.get(
            "combined_confidence",
            0,
        ) >= 0.5
        else "LOW SIMILARITY SIGNAL"
    )

    # ------------------------------------------------------------
    # 8. AI linguistic comparison
    #
    # AI remains an explainable evidence layer.
    # It does NOT replace the validated signal fusion.
    # ------------------------------------------------------------

    try:
        from ..stylometry.ai_profiler import (
            compare_ai_profiles,
        )

        ai_analysis = compare_ai_profiles(
            profile_a,
            profile_b,
        )

    except Exception as exc:
        ai_analysis = {
            "status": "ERROR",
            "provider": "gemini",
            "message": str(exc),
            "shared_characteristics": [],
            "explanation": (
                "AI linguistic comparison was unavailable."
            ),
        }

    return {
        "persona_a": profile_a.get(
            "persona_id",
            "Reference",
        ),

        "persona_b": profile_b.get(
            "persona_id",
            "Candidate",
        ),

        "investigation_id": (
            profile_a.get("investigation_id")
            or profile_b.get("investigation_id")
        ),

        "style_similarity": style_sim,

        "semantic_similarity": semantic_sim,

        "topic_similarity": topic_sim,

        "behavioral_association": beh_sim,

        "temporal_association": temporal_sim,

        "language_analysis": {
            "reference_language": lang_a,
            "candidate_language": lang_b,
            "language_match": lang_a == lang_b,
            "language_similarity": lang_sim,
        },

        "shared_features": shared_features,

        "signals": signals,

        "fusion": fusion_result,

        "ai_analysis": ai_analysis,

        # Preserve profiles so Platform can display
        # the actual AI-generated linguistic evidence.
        "reference_profile": profile_a,

        "candidate_profile": profile_b,

        "assessment": assessment,

        "explanation": (
            "Persona Comparison evaluated classical "
            "stylometric signals, semantic embeddings, "
            "posting behaviour, temporal association, "
            "topic overlap, and AI-generated observable "
            "linguistic analysis."
        ),

        "evidence_ids": [],

        "limitations": [
            "AI linguistic analysis describes observable writing characteristics and is not an identity probability.",
            "Similarity signals represent investigative leads and not definitive identity proof.",
            "Temporal and stylometric overlap does not independently establish that two accounts belong to the same person.",
        ],

        "analysis_version": "4.1.0",
    }
