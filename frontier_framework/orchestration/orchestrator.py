"""
Frontier Framework — Cognitive Orchestrator
The bridge between the evolutionary ecosystem and your cognitive AI system.
Routes tasks to the right species, manages mitosis for hard problems,
aggregates results from multiple agents, and feeds fitness signals back.
"""

from __future__ import annotations
import time
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass, field

from frontier_framework.evolution.genome import Specialization
from frontier_framework.evolution.population import Population


@dataclass
class CognitiveTask:
    """A unit of work routed through the ecosystem."""
    content: str
    required_specialization: Optional[str] = None
    priority: float = 0.5               # 0=low, 1=critical
    use_mitosis: bool = False           # spawn explorer branches
    mitosis_branches: int = 5
    timeout_cycles: int = 3
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class CognitiveResult:
    """Aggregated output from one or more agents."""
    task: str
    agent_id: str
    specialization: str
    confidence: float
    output: str
    generation: int
    energy_spent: float
    consensus_score: float = 0.0        # agreement across mitosis branches
    mitosis_survivors: int = 0
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task":               self.task,
            "agent_id":           self.agent_id,
            "specialization":     self.specialization,
            "confidence":         round(self.confidence, 3),
            "output":             self.output,
            "generation":         self.generation,
            "energy_spent":       round(self.energy_spent, 3),
            "consensus_score":    round(self.consensus_score, 3),
            "mitosis_survivors":  self.mitosis_survivors,
            "latency_ms":         round(self.latency_ms, 2),
        }


class CognitiveOrchestrator:
    """
    Routes cognitive tasks through the living ecosystem.

    Integration pattern:
        orchestrator = CognitiveOrchestrator(population)
        result = orchestrator.process(CognitiveTask("Solve X", use_mitosis=True))

    You can attach an external LLM callback so agents delegate actual reasoning
    to your AI system — the framework handles lifecycle, routing, and evolution.
    """

    def __init__(
        self,
        population: Population,
        llm_callback: Optional[Callable[[str, str], str]] = None,
    ):
        self.population = population
        self.llm_callback = llm_callback          # optional: fn(task, specialization) → str
        self.processed: list[CognitiveResult] = []
        self.routing_log: list[dict] = []

    # ─── Main Entry Point ────────────────────────────────────────────────────

    def process(self, task: CognitiveTask) -> CognitiveResult:
        """Route a task through the ecosystem and return the best result."""
        t0 = time.time()

        if task.use_mitosis or task.priority >= 0.85:
            result = self._process_via_mitosis(task)
        else:
            result = self._process_via_best_agent(task)

        result.latency_ms = (time.time() - t0) * 1000
        self.processed.append(result)
        self._log_routing(task, result)
        return result

    # ─── Routing Strategies ──────────────────────────────────────────────────

    def _process_via_best_agent(self, task: CognitiveTask) -> CognitiveResult:
        """Route to the highest-fitness agent of the required specialization."""
        candidates = [
            a for a in self.population.agents
            if a.alive and (
                task.required_specialization is None or
                a.genome.specialization == task.required_specialization
            )
        ]

        if not candidates:
            # Fallback: any living agent
            candidates = [a for a in self.population.agents if a.alive]

        if not candidates:
            return self._dead_result(task)

        agent = max(candidates, key=lambda a: a.genome.fitness_score)
        energy_before = agent.energy
        think_result = agent.think(task.content)

        # LLM delegation
        output = think_result.get("output", "")
        if self.llm_callback:
            output = self.llm_callback(task.content, agent.genome.specialization)

        return CognitiveResult(
            task=task.content,
            agent_id=agent.id,
            specialization=agent.genome.specialization,
            confidence=think_result.get("confidence", 0.0),
            output=output,
            generation=agent.genome.generation,
            energy_spent=round(energy_before - agent.energy, 3),
        )

    def _process_via_mitosis(self, task: CognitiveTask) -> CognitiveResult:
        """Cognitive mitosis — branch, explore, recombine."""
        mitosis_result = self.population.cognitive_mitosis(
            problem=task.content,
            branches=task.mitosis_branches,
        )

        survivors = mitosis_result.get("explorers", [])
        best_conf = mitosis_result.get("best_confidence", 0.0)

        # Consensus: agreement among survivors
        if survivors:
            confs = [s.get("genome", {}).get("fitness_score", 0) for s in survivors]
            variance = max(confs) - min(confs) if len(confs) > 1 else 0
            consensus = round(1.0 - variance, 3)
        else:
            consensus = 0.0

        spec = mitosis_result.get("best_specialization") or Specialization.SYNTHESIZER.value
        output = f"[Mitosis:{mitosis_result['branches_spawned']}→{mitosis_result['survivors']}] {task.content}"
        if self.llm_callback:
            output = self.llm_callback(task.content, spec)

        return CognitiveResult(
            task=task.content,
            agent_id="mitosis-cluster",
            specialization=spec,
            confidence=best_conf,
            output=output,
            generation=max((s.get("genome", {}).get("generation", 0) for s in survivors), default=0),
            energy_spent=0.0,
            consensus_score=consensus,
            mitosis_survivors=len(survivors),
        )

    def _dead_result(self, task: CognitiveTask) -> CognitiveResult:
        return CognitiveResult(
            task=task.content,
            agent_id="none",
            specialization="none",
            confidence=0.0,
            output="[ecosystem exhausted — no living agents]",
            generation=0,
            energy_spent=0.0,
        )

    # ─── Specialization Routing Helpers ──────────────────────────────────────

    def route_to_species(self, task_content: str, spec: str) -> Optional[CognitiveResult]:
        """Directly target a specialization niche."""
        return self.process(CognitiveTask(task_content, required_specialization=spec))

    def run_pipeline(self, task_content: str) -> list[CognitiveResult]:
        """
        Multi-stage pipeline: planner → researcher → synthesizer → verification.
        Models the full cognitive workflow through species niches.
        """
        pipeline = [
            Specialization.PLANNER.value,
            Specialization.RESEARCHER.value,
            Specialization.CREATIVITY.value,
            Specialization.SYNTHESIZER.value,
            Specialization.VERIFICATION.value,
        ]
        results = []
        context = task_content

        for spec in pipeline:
            result = self.route_to_species(context, spec)
            if result:
                results.append(result)
                context = f"{context} | [{spec}]: {result.output}"

        return results

    # ─── Metrics ─────────────────────────────────────────────────────────────

    def metrics(self) -> dict:
        if not self.processed:
            return {"total": 0}

        confs = [r.confidence for r in self.processed]
        mitosis_tasks = [r for r in self.processed if r.mitosis_survivors > 0]
        return {
            "total_tasks":      len(self.processed),
            "avg_confidence":   round(sum(confs) / len(confs), 3),
            "max_confidence":   round(max(confs), 3),
            "mitosis_tasks":    len(mitosis_tasks),
            "pipeline_tasks":   len([r for r in self.processed if r.specialization == "synthesizer"]),
            "avg_latency_ms":   round(sum(r.latency_ms for r in self.processed) / len(self.processed), 2),
            "recent":           [r.to_dict() for r in self.processed[-5:]],
        }

    def _log_routing(self, task: CognitiveTask, result: CognitiveResult):
        self.routing_log.append({
            "ts":             round(time.time(), 3),
            "task":           task.content[:80],
            "routed_to":      result.agent_id,
            "specialization": result.specialization,
            "confidence":     result.confidence,
            "used_mitosis":   task.use_mitosis,
        })
        if len(self.routing_log) > 200:
            self.routing_log = self.routing_log[-200:]