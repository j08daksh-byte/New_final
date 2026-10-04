from typing import Any, Dict
from pdm.graph import MaintenanceGraph


class GraphQueryLayer:
    """Read-only query layer bridging the impact graph with fleet decision requirements."""

    def __init__(self, graph: MaintenanceGraph):
        self.graph = graph

    def query_engine(self, engine_id: str) -> Dict[str, Any]:
        """Query engine-specific degradation and its path to fleet impact."""
        engine = self.graph.get_node(engine_id)
        if not engine or engine.type != "ENGINE":
            return {}

        parent = self.graph.get_parent(engine_id)
        path = [n.name for n in self.graph.get_upstream_affected(engine_id)]

        return {
            "id": engine.id,
            "name": engine.name,
            "rul": engine.risk_score,
            "risk_status": engine.risk_status,
            "uncertainty": engine.metadata.get("uncertainty"),
            "lo": engine.metadata.get("lo"),
            "hi": engine.metadata.get("hi"),
            "parent_aircraft": parent.name if parent else None,
            "upstream_path": path,
        }

    def query_aircraft(self, aircraft_id: str) -> Dict[str, Any]:
        """Query aircraft-level risk aggregating its sub-systems (engines)."""
        ac = self.graph.get_node(aircraft_id)
        if not ac or ac.type != "AIRCRAFT":
            return {}

        engines = self.graph.get_children(aircraft_id)
        engine_details = [
            {"id": e.id, "rul": e.risk_score, "risk_status": e.risk_status}
            for e in engines
            if e.type == "ENGINE"
        ]

        return {
            "id": ac.id,
            "name": ac.name,
            "aircraft_risk_status": ac.risk_status,
            "aircraft_risk_score": ac.risk_score,
            "engines": engine_details,
        }

    def query_fleet(self) -> Dict[str, Any]:
        """Query macro-level fleet readiness metrics."""
        contributors = self.graph.get_fleet_risk_contributors()

        all_engines = [n for n in self.graph.nodes.values() if n.type == "ENGINE"]
        # Rank by RUL (ascending)
        ranked_engines = sorted(
            all_engines, key=lambda e: (e.risk_score if e.risk_score is not None else float("inf"))
        )

        affected_ac = set()
        for e in contributors:
            p = self.graph.get_parent(e.id)
            if p and p.type == "AIRCRAFT":
                affected_ac.add(p.id)

        return {
            "risk_ranked_engines": [
                {"id": e.id, "rul": e.risk_score, "risk_status": e.risk_status} for e in ranked_engines
            ],
            "affected_aircraft_ids": sorted(list(affected_ac)),
            "fleet_risk_contributors": [
                {"id": c.id, "rul": c.risk_score, "risk_status": c.risk_status} for c in contributors
            ],
        }
