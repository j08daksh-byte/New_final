import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdm.graph import MaintenanceGraph, Node, build_demo_fleet_graph

def test_graph_add_and_retrieve():
    g = MaintenanceGraph()
    g.add_node(Node(id="FLEET-1", type="FLEET", name="Fleet 1"))
    g.add_node(Node(id="AC-1", type="AIRCRAFT", name="Aircraft 1", parent_id="FLEET-1"))
    
    assert g.get_node("AC-1").name == "Aircraft 1"
    assert g.get_parent("AC-1").id == "FLEET-1"
    
    children = g.get_children("FLEET-1")
    assert len(children) == 1
    assert children[0].id == "AC-1"

def test_graph_upstream_affected():
    g = MaintenanceGraph()
    g.add_node(Node(id="FLEET", type="FLEET", name="Fleet"))
    g.add_node(Node(id="AC", type="AIRCRAFT", name="AC", parent_id="FLEET"))
    g.add_node(Node(id="ENG", type="ENGINE", name="ENG", parent_id="AC"))
    g.add_node(Node(id="COMP", type="COMPONENT", name="COMP", parent_id="ENG"))
    
    affected = g.get_upstream_affected("COMP")
    ids = [n.id for n in affected]
    assert ids == ["ENG", "AC", "FLEET"]

def test_graph_risk_propagation():
    g = MaintenanceGraph()
    g.add_node(Node(id="AC-1", type="AIRCRAFT", name="AC"))
    g.add_node(Node(id="ENG-1", type="ENGINE", name="E1", parent_id="AC-1", risk_status="LOW", risk_score=100))
    g.add_node(Node(id="ENG-2", type="ENGINE", name="E2", parent_id="AC-1", risk_status="CRITICAL", risk_score=10))
    
    g.propagate_risk("AC-1")
    ac = g.get_node("AC-1")
    assert ac.risk_status == "CRITICAL"
    assert ac.risk_score == 10

def test_build_demo_fleet_graph():
    # Mock data strictly labelled as test fixtures
    preds = [
        {"id": 1, "pred": 120, "risk_category": {"pessimistic": "LOW"}},
        {"id": 2, "pred": 15, "risk_category": {"pessimistic": "CRITICAL"}},
        {"id": 3, "pred": 45, "risk_category": {"pessimistic": "HIGH"}}
    ]
    
    g = build_demo_fleet_graph(preds)
    
    # 1 fleet + 2 aircraft + 3 engines + 3 components = 9 nodes
    assert len(g.nodes) == 9
    
    ac1 = g.get_node("AC-001")
    assert ac1 is not None
    # Aircraft 1 inherits from E2 which is CRITICAL
    assert ac1.risk_status == "CRITICAL"
    assert ac1.risk_score == 15
    
    # E3 is on Aircraft 2 (since it groups by 2)
    ac2 = g.get_node("AC-002")
    assert ac2.risk_status == "HIGH"
    assert ac2.risk_score == 45
    
    contributors = g.get_fleet_risk_contributors()
    assert len(contributors) == 2  # Unit 2 and Unit 3
    ids = {c.id for c in contributors}
    assert "ENG-002" in ids
    assert "ENG-003" in ids

def test_build_demo_fleet_graph_preserves_all_engines():
    # Ensure no engines are silently dropped
    preds = [{"id": i, "pred": 100, "risk_category": {"pessimistic": "LOW"}} for i in range(1, 101)]
    g = build_demo_fleet_graph(preds)
    
    # 1 fleet + 50 aircraft + 100 engines + 100 components = 251 nodes
    assert len(g.nodes) == 251
    
    engines = [n for n in g.nodes.values() if n.type == "ENGINE"]
    assert len(engines) == 100
    
    components = [n for n in g.nodes.values() if n.type == "COMPONENT"]
    assert len(components) == 100
    
    aircraft = [n for n in g.nodes.values() if n.type == "AIRCRAFT"]
    assert len(aircraft) == 50
