import re
from typing import List, Dict, Any
from .models import HumanBehaviorSignal

def _get_field(post: Any, field_name: str) -> Any:
    if hasattr(post, field_name):
        return getattr(post, field_name)
    if isinstance(post, dict):
        return post.get(field_name)
    return None

def analyze_aliases(posts: List[Any]) -> List[Dict[str, Any]]:
    signals = []
    
    authors = set()
    author_posts = {}
    for post in posts:
        author = _get_field(post, 'author_id') or _get_field(post, 'author') or _get_field(post, 'username')
        if author:
            author_str = str(author).strip()
            authors.add(author_str)
            if author_str not in author_posts:
                author_posts[author_str] = []
            post_id = _get_field(post, 'id') or _get_field(post, 'post_id')
            if post_id:
                author_posts[author_str].append(str(post_id))

    authors_list = list(authors)
    mutations_found = set()
    
    for i, author1 in enumerate(authors_list):
        for j in range(i + 1, len(authors_list)):
            author2 = authors_list[j]
            base1 = re.sub(r'[^a-zA-Z]', '', author1).lower()
            base2 = re.sub(r'[^a-zA-Z]', '', author2).lower()
            
            if base1 and base1 == base2 and author1 != author2:
                mutation_pair = frozenset([author1, author2])
                if mutation_pair not in mutations_found:
                    mutations_found.add(mutation_pair)
                    evidence_ids = author_posts.get(author1, []) + author_posts.get(author2, [])
                    signal = HumanBehaviorSignal(
                        category='identity',
                        signal_type='ALIAS_MUTATION',
                        observations=[f"Observed similar handles: '{author1}' and '{author2}'"],
                        evidence_ids=list(set(evidence_ids)),
                        confidence=0.6,
                        explanation="Similarity in alphabetic characters of aliases suggests potential mutation or reuse of naming conventions.",
                        limitations=["Different actors may use similar base names coincidentally.", "Does not confirm same identity."]
                    )
                    signals.append(signal.model_dump())
                    
    for author, p_ids in author_posts.items():
        if len(p_ids) > 1:
            signal = HumanBehaviorSignal(
                category='identity',
                signal_type='ALIAS_REUSE',
                observations=[f"Handle '{author}' reused across {len(p_ids)} observations."],
                evidence_ids=list(set(p_ids)),
                confidence=0.9,
                explanation="Repeated use of the exact handle in the provided dataset.",
                limitations=["Handle could be a default or generic term.", "Account sharing is possible."]
            )
            signals.append(signal.model_dump())

    return signals
