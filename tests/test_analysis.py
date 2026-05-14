import sys
from pathlib import Path

# Add project root to sys.path so tests can import `utils` when run from pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from utils.analysis import analyze_risk, compliance_check_from_text, structured_output_from_clauses


def test_analyze_risk_empty():
    res = analyze_risk([])
    assert res["avg_score"] == 0.0
    assert res["severity"] == "low"


def test_compliance_check_basic():
    text = "This agreement is governed by the laws of Wonderland. The governing law is Wonderland."
    rules = [
        {"id": "gov_law", "description": "governing law exists", "pattern": "governing law", "must_exist": True},
        {"id": "has_term", "description": "termination clause exists", "pattern": "termination", "must_exist": False},
    ]
    out = compliance_check_from_text(text, rules)
    assert out["overall_passed"] is True
    assert len(out["rules"]) == 2


def test_structured_output_simple():
    clauses = [
        {"chunk_index": 0, "content": "Payment of $1,000 is due by 01/01/2024 to ACME CORP."},
        {"chunk_index": 1, "content": "No liability for indirect damages."},
    ]
    out = structured_output_from_clauses(clauses)
    assert isinstance(out, list)
    assert len(out) == 2
    # first clause should contain a date and at least one amount
    assert isinstance(out[0]["dates"], list)
    assert len(out[0]["dates"]) >= 1
    assert isinstance(out[0]["amounts"], list)
    assert len(out[0]["amounts"]) >= 1
