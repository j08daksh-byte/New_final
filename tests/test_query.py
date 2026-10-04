import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdm.graph import build_demo_fleet_graph
from pdm.query import GraphQueryLayer

def get_test_graph():
    preds = [
        {
            "id": 1, 
            "pred": 120, 
            "risk_category": {"pessimistic": "LOW"}, 
            "uncertainty": "Normal", 
            "lo": 100, 
            "hi": 140
        },
        {
            "id": 2, 
            "pred": 15, 
            "risk_category": {"pessimistic": "CRITICAL", "material_change": True}, 
            "uncertainty": "High", 
            "lo": 5, 
            "hi": 25
        },
        {
            "id": 3, 
            "pred": 45, 
            "risk_category": {"pessimistic": "HIGH"}, 
            "uncertainty": "Normal", 
            "lo": 40, 
            "hi": 50
        }
    ]
    return build_demo_fleet_graph(preds)

def test_query_engine():
    q = GraphQueryLayer(get_test_graph())
    eng = q.query_engine("ENG-002")
    
    assert eng["id"] == "ENG-002"
    assert eng["rul"] == 15
    assert eng["risk_status"] == "CRITICAL"
    assert eng["uncertainty"] == "High"
    assert eng["lo"] == 5
    assert eng["hi"] == 25
    assert eng["parent_aircraft"] == "Aircraft 001"
    assert eng["upstream_path"] == ["Aircraft 001", "Simulated Demo Fleet"]

def test_query_aircraft():
    q = GraphQueryLayer(get_test_graph())
    ac = q.query_aircraft("AC-001")
    
    assert ac["id"] == "AC-001"
    assert ac["aircraft_risk_status"] == "CRITICAL"
    assert ac["aircraft_risk_score"] == 15
    assert len(ac["engines"]) == 2
    
    # Engine 1 and 2
    ids = [e["id"] for e in ac["engines"]]
    assert "ENG-001" in ids
    assert "ENG-002" in ids

def test_query_fleet():
    q = GraphQueryLayer(get_test_graph())
    fleet = q.query_fleet()
    
    # Risk-ranked engines: 15 (ENG-002), 45 (ENG-003), 120 (ENG-001)
    ranked = fleet["risk_ranked_engines"]
    assert len(ranked) == 3
    assert ranked[0]["id"] == "ENG-002"
    assert ranked[1]["id"] == "ENG-003"
    assert ranked[2]["id"] == "ENG-001"
    
    affected_ac = fleet["affected_aircraft_ids"]
    assert len(affected_ac) == 2
    assert "AC-001" in affected_ac  # via ENG-002 (CRITICAL)
    assert "AC-002" in affected_ac  # via ENG-003 (HIGH)
    
    contributors = fleet["fleet_risk_contributors"]
    assert len(contributors) == 2
