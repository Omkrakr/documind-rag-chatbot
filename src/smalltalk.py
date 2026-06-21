"""
smalltalk.py
------------
Lightweight, rule-based small-talk detection.

Design rationale: routing greetings/thanks/farewells through the full RAG
pipeline (embed -> vector search -> generate) wastes a retrieval cycle and,
worse, can produce a nonsense "grounded" answer to "hi" if some unrelated
chunk happens to score above the similarity threshold. A fast, deterministic
pre-check keeps small talk instant and free, and it never touches the
vector store, the cache, or the LLM.

This is intentionally NOT an LLM-based intent classifier: that would add
latency and cost to *every* message just to catch "hi" and "thanks". Plain
pattern matching covers conversational small talk well because its surface
forms are small and stable -- unlike real document questions, which are
open-ended and exactly what retrieval exists for.

Extension point: swap `SmallTalkDetector.detect()` for a small classifier
(or a single cheap LLM call with a "smalltalk | document_question" label)
if you outgrow regex coverage -- nothing else in the pipeline needs to
change, since RAGPipeline only depends on `detect(query) -> str | None`.
"""

from __future__ import annotations
import random
import re
from typing import Optional

_GREETING = re.compile(
    r"^\s*(hi|hello|hey|yo|hiya|sup|good\s*(morning|afternoon|evening))\s*(there|documind)?[\s!.,]*$",
    re.I,
)
_FAREWELL = re.compile(r"^\s*(bye|goodbye|see\s*y(ou|a)|farewell|good\s*night)\s*!?\s*$", re.I)
_THANKS = re.compile(
    r"^\s*(thanks?(\s*(a\s*lot|a\s*bunch|so\s*much|very\s*much))?"
    r"|thank\s*you(\s*(so\s*much|very\s*much|a\s*lot))?"
    r"|thx|ty|appreciate\s*it|cheers)\s*[!.,]*\s*$",
    re.I,
)
_HOW_ARE_YOU = re.compile(r"^\s*how\s*(are|'?s)\s*(you|u|it\s*going)\b", re.I)
_IDENTITY = re.compile(r"^\s*(who\s*are\s*you|what\s*are\s*you|what'?s?\s*your\s*name)\s*\??\s*$", re.I)
_CAPABILITY = re.compile(r"^\s*(what\s*can\s*you\s*do|what\s*do\s*you\s*do|help)\s*\??\s*$", re.I)

_RESPONSES = {
    "greeting": [
        "Hi there! Ask me anything about the documents you've uploaded.",
        "Hello! I'm ready whenever you have a question about your indexed documents.",
        "Hey! Upload a document from the sidebar, then ask away.",
    ],
    "farewell": [
        "Goodbye! Your indexed documents will still be here next time.",
        "See you! Come back anytime you have more questions.",
    ],
    "thanks": [
        "You're welcome! Let me know if you have another question.",
        "Happy to help — feel free to ask anything else about your documents.",
    ],
    "how_are_you": [
        "Running smoothly and ready to help — what would you like to know from your documents?",
    ],
    "identity": [
        "I'm DocuMind, a document Q&A assistant. I answer questions using only the "
        "documents you've uploaded, and I cite the exact passage behind every answer.",
    ],
    "capability": [
        "I answer questions grounded in your uploaded documents and cite my sources. "
        "Upload a file from the sidebar, then ask me anything about it.",
    ],
}

# Order matters: more specific patterns first so e.g. "thanks" isn't
# accidentally caught by a looser pattern checked earlier.
_PATTERNS = [
    (_THANKS, "thanks"),
    (_FAREWELL, "farewell"),
    (_HOW_ARE_YOU, "how_are_you"),
    (_IDENTITY, "identity"),
    (_CAPABILITY, "capability"),
    (_GREETING, "greeting"),
]


class SmallTalkDetector:
    def detect(self, query: str) -> Optional[str]:
        """Return a small-talk reply if `query` is conversational filler,
        else None -- signalling the caller should fall through to the RAG
        pipeline. Deliberately conservative: anything with extra words
        ("hi, what's the leave policy") is left alone so real questions are
        never swallowed by a small-talk match.
        """
        text = (query or "").strip()
        if not text:
            return None
        for pattern, category in _PATTERNS:
            if pattern.search(text):
                return random.choice(_RESPONSES[category])
        return None
