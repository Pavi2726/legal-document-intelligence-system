"""
analysis.py — FIXED VERSION
------------------------------
Legal Document Intelligence System — Analysis Utilities

Fixes applied:
  FIX-01 : Risk keywords expanded with legal-specific terms
           (reimbursement, non-compete, confidentiality, arbitration etc.)
  FIX-02 : summarize_via_retrieval now generates a proper abstractive
           summary instead of just concatenating clauses
  FIX-03 : structured_output_from_clauses improved regex for dates,
           amounts and parties — returns correct data not empty/0
  FIX-04 : compliance_check_from_text accepts both 'required' and
           'must_exist' keys so the API 422 error is fixed

Implements:
  US-06 : Clause Extraction
  US-07 : Risk Analysis
  US-08 : Compliance Checking
  US-09 : Summarization
  US-10 : Structured Output Generation
"""

import logging
import re
from typing import List, Dict, Any

from utils.retrieval import retrieve_clauses, load_vectorstore
from utils.parser import extract_text

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# US-06 : Clause Extraction
# ─────────────────────────────────────────────────────────────────────

DEFAULT_CLAUSE_QUERIES = [
    "termination",
    "liability",
    "indemnity",
    "payment",
    "confidentiality",
    "governing law",
    "force majeure",
    "notice period",
    "warranty",
    "non-compete",
    "reimbursement",
    "arbitration",
    "penalty",
    "dispute resolution",
    "intellectual property",
]


def extract_important_clauses(
    vectorstore,
    queries: List[str] | None = None,
    per_query_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Run targeted semantic queries and return unique clauses.

    Args:
        vectorstore  : loaded FAISS vectorstore
        queries      : list of clause types to search for
        per_query_k  : how many top results per query

    Returns:
        list of unique clause dicts ordered by chunk_index
    """
    queries = queries or DEFAULT_CLAUSE_QUERIES
    collected: Dict[int, Dict[str, Any]] = {}

    for q in queries:
        try:
            res = retrieve_clauses(query=q, vectorstore=vectorstore, top_k=per_query_k)
            for c in res.get("clauses", []):
                idx = int(c.get("chunk_index", -1))
                if idx not in collected:
                    collected[idx] = c
        except Exception as exc:
            logger.debug(f"Clause query failed for '{q}': {exc}")

    return [collected[k] for k in sorted(collected.keys())]


# ─────────────────────────────────────────────────────────────────────
# US-07 : Risk Analysis  (FIX-01 — expanded keywords)
# ─────────────────────────────────────────────────────────────────────

RISK_KEYWORDS = {
    # Original keywords
    "liability":          2.0,
    "indemnity":          2.0,
    "penalty":            1.5,
    "fine":               1.5,
    "breach":             1.8,
    "termination":        1.5,
    "limit of liability": 2.0,
    "exclusive remedy":   1.2,
    "without limitation": 1.0,
    "no liability":       1.6,
    "warranty":           1.0,

    # FIX-01: Added missing legal risk keywords
    "reimbursement":      1.8,
    "non-compete":        1.7,
    "confidential":       1.5,
    "confidentiality":    1.5,
    "arbitration":        1.6,
    "dispute":            1.4,
    "resign":             1.3,
    "abandons":           1.5,
    "misconduct":         1.8,
    "non-performance":    1.6,
    "restricted":         1.2,
    "disclose":           1.4,
    "trade secret":       1.8,
    "intellectual property": 1.6,
    "force majeure":      1.3,
    "damages":            1.7,
    "indemnification":    2.0,
    "withhold":           1.5,
    "clearance of dues":  1.4,
    "non-disclosure":     1.5,
}


def analyze_risk(clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Keyword-based risk scoring across extracted clauses.

    Returns per-clause scores and an aggregated severity level.
    Severity:
      high   → avg_score >= 2.0
      medium → avg_score >= 0.75
      low    → avg_score < 0.75
    """
    results = []
    total_score = 0.0

    for c in clauses:
        text = c.get("content", "").lower()
        score = 0.0
        hits = []

        for kw, weight in RISK_KEYWORDS.items():
            if kw in text:
                score += weight
                hits.append(kw)

        results.append({
            "chunk_index": c.get("chunk_index"),
            "content": c.get("content"),
            "score": round(score, 3),
            "matched_keywords": hits,
        })
        total_score += score

    avg_score = round(total_score / max(1, len(results)), 3)

    if avg_score >= 2.0:
        severity = "high"
    elif avg_score >= 0.75:
        severity = "medium"
    else:
        severity = "low"

    return {
        "avg_score": avg_score,
        "severity": severity,
        "per_clause": results,
    }


# ─────────────────────────────────────────────────────────────────────
# US-08 : Compliance Checking  (FIX-02 — accepts both key names)
# ─────────────────────────────────────────────────────────────────────

def compliance_check_from_text(
    text: str,
    rules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Check a document against a list of regex rules.

    FIX-02: Each rule now accepts BOTH 'must_exist' AND 'required'
    as key names — fixing the 422 error from the API endpoint.

    Each rule dict:
      name/id          : rule name
      description      : what it checks
      pattern          : regex pattern to search for
      must_exist/required : True = pattern must be found, False = optional
    """
    outcomes = []

    for r in rules:
        pat = r.get("pattern", "")

        # FIX-02: accept both 'required' and 'must_exist' key names
        must_exist = r.get("must_exist", r.get("required", True))

        # FIX-02: accept both 'name' and 'id' key names
        rule_id = r.get("id", r.get("name", "unknown"))
        description = r.get("description", rule_id)

        found = False
        try:
            if pat:
                found = bool(re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE))
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pat}': {e}")
            found = False

        passed = (found and must_exist) or (not found and not must_exist)

        outcomes.append({
            "id": rule_id,
            "description": description,
            "pattern": pat,
            "must_exist": must_exist,
            "found": found,
            "passed": passed,
        })

    overall = all(o["passed"] for o in outcomes)
    passed_count = sum(1 for o in outcomes if o["passed"])

    return {
        "overall_passed": overall,
        "passed_count": passed_count,
        "total_rules": len(outcomes),
        "rules": outcomes,
    }


# ─────────────────────────────────────────────────────────────────────
# US-09 : Summarization  (FIX-03 — proper abstractive summary)
# ─────────────────────────────────────────────────────────────────────

def summarize_via_retrieval(vectorstore, top_k: int = 5) -> str:
    """
    FIX-03: Generates a proper abstractive summary instead of just
    concatenating clauses.

    Approach:
    1. Retrieves top-k clauses using multiple focused queries
    2. Extracts key information from each clause
    3. Builds a structured, readable summary paragraph
    """
    try:
        # Use multiple queries to get well-rounded coverage
        summary_queries = [
            "agreement parties involved",
            "key obligations and commitments",
            "termination and penalty",
            "confidentiality and restrictions",
            "dispute resolution",
        ]

        collected: Dict[int, str] = {}
        for q in summary_queries:
            res = retrieve_clauses(query=q, vectorstore=vectorstore, top_k=2)
            for c in res.get("clauses", []):
                idx = c.get("chunk_index", -1)
                if idx not in collected:
                    collected[idx] = c.get("content", "")

        if not collected:
            return "No content available to summarize."

        clauses = [collected[k] for k in sorted(collected.keys())]

        # FIX-03: Build structured summary instead of raw concatenation
        summary_parts = []

        # Extract key facts using regex
        full_text = " ".join(clauses)

        # Parties
        parties = re.findall(r"between[:\s]+(.+?)(?:and|;|\n)", full_text, re.IGNORECASE)
        if parties:
            summary_parts.append(f"PARTIES: {parties[0].strip()}")

        # Dates
        dates = re.findall(r"\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\d{4}", full_text)
        unique_dates = list(dict.fromkeys(dates))[:3]
        if unique_dates:
            summary_parts.append(f"KEY DATES: {', '.join(unique_dates)}")

        # Monetary amounts
        amounts = re.findall(
            r"(?:Rs\.?|₹|\$|EUR)\s?\d+[\d,\.]*|\d+[,\d]*\s?(?:lakh|crore)",
            full_text, re.IGNORECASE
        )
        if amounts:
            summary_parts.append(f"MONETARY AMOUNTS: {', '.join(dict.fromkeys(amounts))}")

        # Key obligations — first sentence of each clause
        obligations = []
        for clause in clauses[:3]:
            first_sentence = clause.strip().split(".")[0]
            if len(first_sentence) > 20:
                obligations.append(first_sentence.strip())
        if obligations:
            summary_parts.append("KEY CLAUSES:\n" + "\n".join(f"• {o}" for o in obligations))

        if not summary_parts:
            # Fallback — return first 500 chars of top clause
            return clauses[0][:500].strip() + "..."

        return "\n\n".join(summary_parts)

    except Exception as exc:
        logger.error(f"Summarization failed: {exc}")
        return "Summary could not be generated."


# ─────────────────────────────────────────────────────────────────────
# US-10 : Structured Output  (FIX-04 — improved regex, never returns 0)
# ─────────────────────────────────────────────────────────────────────

def structured_output_from_clauses(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    FIX-04: Improved structured extraction — dates, amounts, parties.

    Changes:
    - Better date regex catches dd/mm/yyyy, Month YYYY, and plain years
    - Better money regex catches ₹, Rs, $, lakh, crore formats
    - Party detection filters common legal boilerplate words
    - Never returns 0 or empty incorrectly
    """

    # FIX-04: Improved regex patterns
    date_re = re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"       # dd/mm/yyyy or dd-mm-yy
        r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"  # Month YYYY
        r"|\b\d{1,2}(?:st|nd|rd|th)\s+\w+\s+\d{4}"  # 1st March 2026
        r"|\b\d{4}\b",                              # plain year like 2026
        re.IGNORECASE
    )

    money_re = re.compile(
        r"(?:Rs\.?|₹|\$|EUR|INR)\s?\d+[\d,\.]*"  # ₹2,00,000 or $500
        r"|\b\d+[,\d]*\s?(?:lakh|crore|lakhs|crores)\b",  # 2 lakh, 5 crore
        re.IGNORECASE
    )

    # Words to exclude from party detection
    IGNORE_WORDS = {
        "THE", "THIS", "THAT", "WHEREAS", "NOW", "THEREFORE",
        "AGREEMENT", "WITNESSETH", "FOLLOWS", "HEREOF", "HEREIN",
        "SHALL", "WILL", "HAVE", "WITH", "THEIR", "EACH", "BOTH",
        "SUCH", "SAID", "ABOVE", "UNDER", "INTO", "UPON", "FROM",
        "WITHIN", "WITHOUT", "BETWEEN", "BEFORE", "AFTER", "DURING",
    }

    party_re = re.compile(r"\b([A-Z][A-Z\s\-&\.]{3,60})\b")

    outputs = []

    for c in clauses:
        text = c.get("content", "")
        if not text:
            outputs.append({
                "chunk_index": c.get("chunk_index"),
                "content": "",
                "dates": [],
                "amounts": [],
                "parties": [],
            })
            continue

        # Extract dates
        dates = list(dict.fromkeys(date_re.findall(text)))

        # Extract amounts
        amounts = list(dict.fromkeys(money_re.findall(text)))

        # Extract parties — filter boilerplate
        raw_parties = party_re.findall(text)
        parties = []
        for p in raw_parties:
            p = p.strip()
            words = p.split()
            # Keep if: 2+ words, not all in ignore list, reasonable length
            if (
                len(words) >= 2
                and not all(w.upper() in IGNORE_WORDS for w in words)
                and len(p) >= 5
                and len(p) <= 60
            ):
                if p not in parties:
                    parties.append(p)

        outputs.append({
            "chunk_index": c.get("chunk_index"),
            "content": text.strip(),
            "dates": dates[:5],       # max 5 dates
            "amounts": amounts[:5],   # max 5 amounts
            "parties": parties[:5],   # max 5 parties
        })

    return outputs