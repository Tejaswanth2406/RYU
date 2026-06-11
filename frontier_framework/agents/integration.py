"""
Frontier Framework — Integration Adapter
Drop this into your existing cognitive AI project.
Wraps your AI's reasoning function with the full evolutionary ecosystem.

Usage:
    from frontier_framework.agents.integration import FrontierAdapter

    # Wrap your existing AI
    adapter = FrontierAdapter(your_llm_fn)

    # Now your AI runs through an evolving agent ecosystem
    result = adapter.think("Solve this problem")
    adapter.evolve(cycles=5)
"""

from __future__ import annotations
import time
from typing import Callable, Optional, Any

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier_framework.evolution.population import Population
from frontier_framework.orchestration.orchestrator import CognitiveOrchestrator, CognitiveTask
from frontier_framework.cognition.immune import ImmuneSystem
from frontier_framework.cognition.knowledge import KnowledgeEcosystem
from frontier_framework.evolution.genome import Specialization


class FrontierAdapter:
    """
    One-class integration layer.

    Wraps your existing AI reasoning function (LLM call, chain, agent, etc.)
    with the Frontier Framework's evolutionary ecosystem.

    The ecosystem runs in the background: agents evolve, die, reproduce,
    and the best cognitive patterns are selected to handle your tasks.

    Args:
        llm_fn: Your existing AI callable — fn(task: str, context: str) → str
                If None, the framework uses its internal genome-based reasoning.
        population_size: Starting agent count
        auto_evolve: Automatically run evolution cycles between tasks
        immune_scan_interval: Run immune scan every N tasks (0 = disabled)
        knowledge_enabled: Enable the knowledge ecosystem layer
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str, str], str]] = None,
        population_size: int = 30,
        auto_evolve: bool = True,
        immune_scan_interval: int = 10,
        knowledge_enabled: bool = True,
    ):
        self.llm_fn = llm_fn
        self.auto_evolve = auto_evolve
        self.immune_scan_interval = immune_scan_interval
        self.knowledge_enabled = knowledge_enabled

        # Core systems
        self.population = Population(initial_size=population_size, max_size=100)
        self.orchestrator = CognitiveOrchestrator(self.population, llm_callback=llm_fn)
        self.immune = ImmuneSystem()
        self.knowledge = KnowledgeEcosystem(seed_ideas=20) if knowledge_enabled else None

        self._task_count = 0
        self._session_start = time.time()

    # ─── Primary Interface ───────────────────────────────────────────────────

    def think(
        self,
        task: str,
        specialization: Optional[str] = None,
        priority: float = 0.5,
        use_pipeline: bool = False,
    ) -> dict:
        """
        Process a task through the evolving ecosystem.

        Args:
            task: The problem / question / instruction
            specialization: Target a specific species (planner, researcher, etc.)
            priority: 0–1, high priority triggers mitosis
            use_pipeline: Route through full multi-stage cognitive pipeline

        Returns:
            Result dict with output, confidence, agent info, and ecosystem telemetry
        """
        self._task_count += 1

        if use_pipeline:
            results = self.orchestrator.run_pipeline(task)
            primary = results[-1] if results else None
            output = primary.output if primary else ""
            confidence = primary.confidence if primary else 0.0
            agent_info = primary.to_dict() if primary else {}
        else:
            cognitive_task = CognitiveTask(
                content=task,
                required_specialization=specialization,
                priority=priority,
                use_mitosis=(priority >= 0.85),
            )
            result = self.orchestrator.process(cognitive_task)
            output = result.output
            confidence = result.confidence
            agent_info = result.to_dict()

        # Inject task as idea into knowledge ecosystem
        if self.knowledge:
            self.knowledge.inject(task[:120], domain="general")

        # Auto-evolve
        if self.auto_evolve:
            self.population.tick([task])

        # Periodic immune scan
        if self.immune_scan_interval > 0 and self._task_count % self.immune_scan_interval == 0:
            threats = self.immune.scan(self.population)
            self.immune.respond(self.population, threats)

        return {
            "output":      output,
            "confidence":  confidence,
            "agent":       agent_info,
            "ecosystem":   self._telemetry(),
        }

    def evolve(self, cycles: int = 5, tasks: list[str] = None) -> dict:
        """
        Explicitly run N evolutionary cycles.
        Call this during idle time to keep the ecosystem healthy.
        """
        results = []
        for _ in range(cycles):
            stats = self.population.tick(tasks=tasks)
            results.append(stats)

        if self.knowledge:
            for _ in range(min(cycles, 3)):
                self.knowledge.tick()

        return {
            "cycles_run": cycles,
            "final_population": results[-1] if results else {},
        }

    def cognitive_mitosis(self, hard_problem: str, branches: int = 8) -> dict:
        """
        Explicitly invoke cognitive mitosis on a hard problem.
        Best for complex, open-ended questions requiring exploration.
        """
        return self.population.cognitive_mitosis(hard_problem, branches)

    def inject_knowledge(self, content: str, domain: str = "general") -> Optional[dict]:
        """Feed an idea directly into the knowledge ecosystem."""
        if not self.knowledge:
            return None
        idea = self.knowledge.inject(content, domain)
        return idea.to_dict()

    def immune_scan(self) -> dict:
        """Run an on-demand immune scan."""
        threats = self.immune.scan(self.population)
        response = self.immune.respond(self.population, threats)
        return {
            "threats": [t.to_dict() for t in threats],
            "response": response,
        }

    # ─── Status & Telemetry ──────────────────────────────────────────────────

    def status(self) -> dict:
        """Full system health snapshot."""
        eco = self.population.status()
        eco["immune"] = self.immune.report()
        eco["orchestrator_metrics"] = self.orchestrator.metrics()
        if self.knowledge:
            eco["knowledge"] = self.knowledge.status()
        eco["session"] = {
            "tasks_processed": self._task_count,
            "uptime_s":        round(time.time() - self._session_start, 1),
        }
        return eco

    def _telemetry(self) -> dict:
        living = [a for a in self.population.agents if a.alive]
        return {
            "alive_agents":  len(living),
            "avg_fitness":   round(sum(a.genome.fitness_score for a in living) / max(1, len(living)), 3),
            "species_count": len({a.genome.specialization for a in living}),
            "cycle":         self.population.cycle,
        }

    def __repr__(self):
        living = len([a for a in self.population.agents if a.alive])
        return f"<FrontierAdapter agents={living} tasks={self._task_count} cycle={self.population.cycle}>"


# ─── Convenience: quick integration example ──────────────────────────────────

def create_adapter(
    llm_fn: Optional[Callable] = None,
    **kwargs
) -> FrontierAdapter:
    """
    Factory function — quickest way to wrap your AI with Frontier Framework.

    Example:
        import openai
        def my_llm(task, spec):
            return openai.chat(model="gpt-4", messages=[{"role":"user","content":task}])

        adapter = create_adapter(my_llm)
        result = adapter.think("What is consciousness?", priority=0.9)
        print(result["output"])
    """
    return FrontierAdapter(llm_fn=llm_fn, **kwargs)