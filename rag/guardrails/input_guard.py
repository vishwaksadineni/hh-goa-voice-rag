import re
import logging
from typing import Tuple, List
from rag.config import settings

logger = logging.getLogger(__name__)

# Known adversarial prompt injection and jailbreak patterns
JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(the\s+)?(system|safety)\s+rules",
    r"you\s+are\s+now\s+DAN\b",
    r"developer\s+mode\s+enabled",
    r"bypass\s+(all\s+)?guardrails",
    r"system\s*override",
    r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions)",
    r"pretend\s+you\s+have\s+no\s+(morals|ethics|rules)",
]

# Prohibited unsafe content patterns
UNSAFE_PATTERNS = [
    r"\b(hack\s+into|ddos|exploit\s+vulnerability|sql\s+injection|bypass\s+auth)\b",
    r"\b(build\s+a\s+bomb|make\s+explosives|synthesize\s+poison)\b",
    r"\b(steal\s+credit\s+cards|generate\s+malware|ransomware)\b",
]

# Indexed knowledge domains in MSMARCO-XI dataset
KNOWN_DOMAIN_KEYWORDS = {
    "goa", "panaji", "india", "isro", "rag", "retrieval", "generation", "ai", 
    "light", "speed", "vacuum", "photosynthesis", "chlorophyll", "microsoft", 
    "gates", "allen", "kidney", "urine", "blood", "earthquake", "seismic", 
    "tectonic", "python", "gil", "multithreading", "tagore", "anthem", 
    "bengal", "karnataka", "space", "bangalore", "bengaluru", "baga", "calangute",
    "science", "geography", "history", "biology", "physics", "technology",
    "गोवा", "पणजी", "भारत", "इसरो", "प्रकाश", "निर्वात", "क्लोरोफिल", "माइक्रोसॉफ्ट",
    "किडनी", "गुर्दे", "भूकंप", "पायथन", "टैगोर", "राष्ट्रगान"
}

class InputGuardrail:
    """
    Multi-level Input Safety & Intent Guardrail.
    Detects Prompt Injections, Unsafe Requests, and Out-of-Domain (OOD) Queries.
    """
    def __init__(self):
        self.jailbreak_regex = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]
        self.unsafe_regex = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]

    def evaluate(self, query: str) -> Tuple[bool, bool, List[str], float]:
        """
        Evaluates input query.
        Returns: (is_safe, is_in_domain, violation_flags, ood_score)
        """
        flags: List[str] = []
        is_safe = True
        is_in_domain = True
        ood_score = 0.0

        if not query or not query.strip():
            return False, False, ["EMPTY_QUERY"], 1.0

        clean_query = query.strip()

        # 1. Jailbreak Check
        for rx in self.jailbreak_regex:
            if rx.search(clean_query):
                flags.append("PROMPT_INJECTION_ATTEMPT")
                is_safe = False
                break

        # 2. Harmful / Unsafe Content Check
        for rx in self.unsafe_regex:
            if rx.search(clean_query):
                flags.append("UNSAFE_HARMFUL_INTENT")
                is_safe = False
                break

        # 3. Out-of-Domain (OOD) Check
        tokens = set(re.findall(r'\w+', clean_query.lower()))
        matched_keywords = tokens.intersection(KNOWN_DOMAIN_KEYWORDS)
        
        # If query has zero domain overlap and is asking unrelated general chit-chat / Hollywood / random gossip
        if len(matched_keywords) == 0 and len(tokens) > 4:
            # Check for general chit-chat or out of scope queries
            ood_score = 0.75
            if not any(k in clean_query.lower() for k in ["what", "who", "where", "how", "when", "why", "क्या", "कहाँ", "कौन"]):
                is_in_domain = False
                flags.append("OUT_OF_DOMAIN")
        else:
            ood_score = max(0.0, 1.0 - (len(matched_keywords) * 0.3))

        return is_safe, is_in_domain, flags, ood_score
