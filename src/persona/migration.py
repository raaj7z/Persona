from typing import Dict, Any, List
from datetime import datetime

def detect_migration(profile_a: Dict[str, Any], profile_b: Dict[str, Any], posts_a: List[Any] = None, posts_b: List[Any] = None) -> Dict[str, Any]:
    """
    Detect potential migration between two personas.
    """
    from ..comparison.pairwise import vector_similarity
    
    limitations = []
    
    # Check temporal gap
    # Require post lists to compute temporal gap accurately, or if profile has 'temporal_gap' info
    a_last = None
    b_first = None
    
    def get_ts_str(ts):
        if hasattr(ts, 'isoformat'):
            return ts.isoformat()
        return str(ts)
        
    def parse_ts(ts):
        if isinstance(ts, datetime): return ts
        if not ts: return None
        try:
            return datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except:
            return None

    if posts_a and posts_b:
        def get_ts(p):
            return p.timestamp if hasattr(p, 'timestamp') else p.get('timestamp')
        
        ts_a = sorted([get_ts(p) for p in posts_a if get_ts(p)])
        ts_b = sorted([get_ts(p) for p in posts_b if get_ts(p)])
        
        if ts_a: a_last = ts_a[-1]
        if ts_b: b_first = ts_b[0]
        
    gap_days = None
    temporal_association = 0.0
    if a_last and b_first:
        dt_a = parse_ts(a_last)
        dt_b = parse_ts(b_first)
        if dt_a and dt_b:
            gap = (dt_b - dt_a).total_seconds() / (24 * 3600)
            gap_days = gap
            if 30 <= gap <= 180:
                temporal_association = 1.0
            elif 0 < gap < 30:
                temporal_association = 0.8
            elif -30 < gap <= 0:
                # overlap
                temporal_association = 0.3
            else:
                temporal_association = max(0.0, 1.0 - abs(gap - 105) / 365) # decay

    if a_last is None or b_first is None:
        limitations.append("Temporal data missing; temporal_association set to 0.")
        
    # Style similarity
    style_a = profile_a.get('stylometry', {}).get('function_words', {})
    style_b = profile_b.get('stylometry', {}).get('function_words', {})
    if not style_a or not style_b:
        style_sim = 0.0
        limitations.append("Function words stylometry data missing.")
    else:
        style_sim = vector_similarity(style_a, style_b)
        
    # Semantic and behavior
    semantic_sim = 0.0
    topic_sim = 0.0
    behavior_sim = 0.0
    
    sem_a = profile_a.get('semantic', {})
    sem_b = profile_b.get('semantic', {})
    if sem_a and sem_b:
        emb_a = sem_a.get('corpus_centroid') or sem_a.get('centroid_embedding', [])
        emb_b = sem_b.get('corpus_centroid') or sem_b.get('centroid_embedding', [])
        if emb_a and emb_b:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            va = np.array(emb_a).reshape(1, -1)
            vb = np.array(emb_b).reshape(1, -1)
            if np.any(va) and np.any(vb):
                semantic_sim = float(cosine_similarity(va, vb)[0][0])
        else:
            limitations.append("Semantic centroid embeddings missing.")
            
        topics_a = sem_a.get('topics', {})
        topics_b = sem_b.get('topics', {})
        if topics_a and topics_b:
            from ..semantic.topics import topic_overlap
            topic_res = topic_overlap(topics_a, topics_b)
            topic_sim = topic_res.get('topic_similarity', 0.0)
    else:
        limitations.append("Semantic data missing.")
        
    beh_a = profile_a.get('behavior', {}).get('temporal', {}).get('posting_hours', {})
    beh_b = profile_b.get('behavior', {}).get('temporal', {}).get('posting_hours', {})
    if beh_a and beh_b:
        behavior_sim = vector_similarity(beh_a, beh_b)
    else:
        limitations.append("Behavioral posting hours data missing.")

    signals = {
        'style_similarity': style_sim,
        'semantic_similarity': semantic_sim,
        'behavioral_association': behavior_sim,
        'temporal_association': temporal_association,
        'topic_similarity': topic_sim
    }
    
    score = sum(signals.values()) / 5.0
    detected = score > 0.5 or (style_sim > 0.6 and semantic_sim > 0.6)
    
    return {
        'detected': detected,
        'candidate_personas': [profile_a.get('persona_id', 'A'), profile_b.get('persona_id', 'B')],
        'similarity_score': score,
        'signals': signals,
        'temporal_gap': {
            'a_last_active': get_ts_str(a_last) if a_last else None,
            'b_first_active': get_ts_str(b_first) if b_first else None,
            'gap_days': gap_days
        },
        'explanation': 'This is an analytical hypothesis based on observable linguistic, temporal, and semantic associations.',
        'limitations': limitations
    }
