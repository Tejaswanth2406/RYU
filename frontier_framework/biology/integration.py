from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BiologicalConcept:
    """A biological mechanism with a direct computational analogy."""
    name: str
    biological_function: str
    computational_application: str


class BiologicalLifecycle:
    """
    Taxonomy of life-like mechanisms to integrate into the frontier framework.

    The project already models agents, populations, and civilizations. This class maps
    biological functions to computational roles so that the architecture can evolve in
    a more faithful life-like manner without requiring a separate simulation engine.
    """

    TAXONOMY: tuple[BiologicalConcept, ...] = (
        BiologicalConcept("cell_formation", "Creation of new cells", "Entity initialization"),
        BiologicalConcept("cell_differentiation", "Cells acquire specialized functions", "Dynamic specialization"),
        BiologicalConcept("cell_specialization", "Cells perform particular tasks", "Role assignment"),
        BiologicalConcept("membrane", "Controls exchange with environment", "Access control / boundary"),
        BiologicalConcept("cytoplasm", "Medium for cellular reactions", "Shared execution environment"),
        BiologicalConcept("nucleus", "Stores genetic information", "Central information storage"),
        BiologicalConcept("mitochondria", "Energy production", "Energy management"),
        BiologicalConcept("ribosomes", "Protein synthesis", "Instruction execution"),
        BiologicalConcept("metabolism", "Transforms nutrients into usable energy", "Resource utilization"),
        BiologicalConcept("reproduction", "Creates offspring and propagates lineage", "Replication and inheritance"),
        BiologicalConcept("immune_response", "Detects and eliminates threats", "Threat detection and containment"),
        BiologicalConcept("adaptation", "Changes under environmental pressure", "Learning under stress"),
        BiologicalConcept("homeostasis", "Maintains internal stability", "Feedback control and balancing"),
        BiologicalConcept("apoptosis", "Programmed cell death", "Selective pruning and cleanup"),
        BiologicalConcept("evolution", "Species diversify over generations", "Selection pressure and lineage growth"),
    )

    def __init__(self, concepts: Iterable[BiologicalConcept] | None = None):
        self.concepts = tuple(concepts) if concepts is not None else self.TAXONOMY

    def get(self, name: str) -> BiologicalConcept | None:
        for concept in self.concepts:
            if concept.name == name:
                return concept
        return None

    def stage_summary(self) -> dict[str, str]:
        return {
            concept.name: concept.computational_application
            for concept in self.concepts
        }

    def summarize(self) -> list[dict[str, str]]:
        return [
            {
                "name": concept.name,
                "biological_function": concept.biological_function,
                "computational_application": concept.computational_application,
            }
            for concept in self.concepts
        ]
