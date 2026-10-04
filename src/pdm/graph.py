from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Node:
    id: str
    type: str  # 'FLEET', 'AIRCRAFT', 'ENGINE', 'COMPONENT'
    name: str
    risk_score: Optional[float] = None  # e.g., RUL
    risk_status: str = "UNKNOWN"  # e.g., 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class MaintenanceGraph:
    """A lightweight directed acyclic graph mapping aerospace components to fleet impact."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        if node.parent_id and node.parent_id in self.nodes:
            # Avoid duplicates
            if node.id not in self.nodes[node.parent_id].children_ids:
                self.nodes[node.parent_id].children_ids.append(node.id)

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_parent(self, node_id: str) -> Optional[Node]:
        node = self.get_node(node_id)
        if node and node.parent_id:
            return self.get_node(node.parent_id)
        return None

    def get_children(self, node_id: str) -> List[Node]:
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def get_upstream_affected(self, node_id: str) -> List[Node]:
        """Returns all ancestors of a node, tracing impact up to the fleet."""
        affected = []
        current = self.get_parent(node_id)
        while current:
            affected.append(current)
            current = self.get_parent(current.id)
        return affected

    def get_fleet_risk_contributors(self) -> List[Node]:
        """Returns all engines that contribute to fleet risk (e.g. CRITICAL or HIGH status)."""
        return [
            n for n in self.nodes.values()
            if n.type == "ENGINE" and n.risk_status in ("CRITICAL", "HIGH", "MATERIAL_CHANGE")
        ]

    def propagate_risk(self, node_id: str):
        """
        Simple aggregate: Aircraft inherits the worst-case risk of its engines.
        Fleet inherits the count of degraded aircraft.
        """
        node = self.get_node(node_id)
        if not node or not node.children_ids:
            return
            
        # Recursive propagation from children
        for cid in node.children_ids:
            self.propagate_risk(cid)
            
        children = self.get_children(node_id)
        
        # Determine the worst status among children
        status_priority = {"CRITICAL": 4, "MATERIAL_CHANGE": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
        worst_status = "UNKNOWN"
        min_score = None
        
        for c in children:
            if status_priority.get(c.risk_status, 0) > status_priority.get(worst_status, 0):
                worst_status = c.risk_status
            if c.risk_score is not None:
                if min_score is None or c.risk_score < min_score:
                    min_score = c.risk_score
                    
        node.risk_status = worst_status
        if min_score is not None:
            node.risk_score = min_score


def build_demo_fleet_graph(engine_predictions: List[Dict]) -> MaintenanceGraph:
    """
    Builds a deterministic demonstration fleet connecting C-MAPSS engines to simulated aircraft.
    Groups every 2 engines into a single twin-engine aircraft. Supports any number of engines.
    """
    g = MaintenanceGraph()
    
    # Fleet Root
    g.add_node(Node(id="FLEET-01", type="FLEET", name="Simulated Demo Fleet"))
    
    # Process all engines dynamically
    # Expects engine_predictions to be a list of dictionaries containing 'id', 'pred', 'risk_category'
    for idx in range(0, len(engine_predictions), 2):
        ac_num = (idx // 2) + 1
        ac_id = f"AC-{ac_num:03d}"
        g.add_node(Node(id=ac_id, type="AIRCRAFT", name=f"Aircraft {ac_num:03d}", parent_id="FLEET-01"))
        
        # Take up to 2 engines for this aircraft
        engines_for_ac = engine_predictions[idx:idx+2]
        
        for engine in engines_for_ac:
            e_unit = engine["id"]
            e_id = f"ENG-{e_unit:03d}"
            
            # Real risk status comes from the material_change override logic (the pessimistic band)
            # Fallback to plain band if risk_category is absent (for mock tests)
            status = "UNKNOWN"
            if "risk_category" in engine:
                status = engine["risk_category"]["pessimistic"]
            elif "band" in engine:
                status = engine["band"]
                
            pred_rul = engine.get("pred")
            
            # Attach Engine to Aircraft
            g.add_node(Node(
                id=e_id, 
                type="ENGINE", 
                name=f"Engine Unit {e_unit}", 
                risk_score=pred_rul,
                risk_status=status,
                parent_id=ac_id,
                metadata=engine
            ))
            
            # Since C-MAPSS FD001 provides a single monolithic RUL without specific component isolation,
            # Data Honesty Rule: We must NOT invent fake failing components.
            g.add_node(Node(
                id=f"COMP-{e_unit:03d}-SYS",
                type="COMPONENT",
                name="Engine Degradation Risk (General)",
                risk_score=pred_rul,
                risk_status=status,
                parent_id=e_id
            ))

    # Propagate up to the fleet
    g.propagate_risk("FLEET-01")
    return g
