"""
Frontier Framework — Agent Swarm (Stage 4: Internal Ecosystems)
Multi-agent swarm intelligence.
Decomposes tasks into subtasks, assigns to specialist agents in parallel,
aggregates results via consensus voting, feeds fitness signals back.
Different AI species occupy different niches — some compete, some symbiose.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import defaultdict

from frontier_framework.evolution.genome import AgentCell, Specialization
from frontier_framework.evolution.population import Population
from frontier_framework.memory.memory import DistributedMemory


@dataclass
class SwarmTask:
    """A decomposed unit of work assigned to a swarm agent."""
    id: str                         = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str                    = ""
    required_spec: Optional[str]    = None
    priority: float                 = 0.5
    parent_task_id: Optional[str]   = None
    assigned_agent_id: Optional[str]= None
    status: str                     = "pending"     # pending | running | done | failed
    result: Optional[dict]          = None
    created_at: float               = field(default_factory=time.time)
    completed_at: Optional[float]   = None

    def complete(self, result: dict):
        self.result = result
        self.status = "done"
        self.completed_at = time.time()

    def fail(self, reason: str = ""):
        self.status = "failed"
        self.result = {"error": reason}
        self.completed_at = time.time()

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "content":      self.content[:80],
            "required_spec":self.required_spec,
            "priority":     self.priority,
            "status":       self.status,
            "agent_id":     self.assigned_agent_id,
            "result":       self.result,
            "latency_ms":   round((self.completed_at - self.created_at) * 1000, 2) if self.completed_at else None,
        }


@dataclass
class SwarmResult:
    """Aggregated output from a swarm run."""
    original_task: str
    subtasks: list[SwarmTask]
    consensus_output: str
    confidence: float
    species_used: list[str]
    total_agents: int
    agreement_score: float
    latency_ms: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "original_task":   self.original_task,
            "consensus_output":self.consensus_output,
            "confidence":      round(self.confidence, 3),
            "agreement_score": round(self.agreement_score, 3),
            "species_used":    self.species_used,
            "total_agents":    self.total_agents,
            "subtasks_count":  len(self.subtasks),
            "subtasks":        [s.to_dict() for s in self.subtasks],
            "latency_ms":      round(self.latency_ms, 2),
        }


class AgentSwarm:
    """
    Coordinates a colony of specialist agents to collectively solve problems.

    Execution modes:
    ─ PARALLEL   : All subtasks run simultaneously across different species
    ─ SEQUENTIAL : Pipeline — each stage feeds the next
    ─ TOURNAMENT : Multiple agents compete on same task; best wins
    ─ CONSENSUS  : All agents tackle the same task; majority vote

    Symbiosis patterns:
    ─ Planner + Researcher → Synthesizer (common pipeline)
    ─ Creativity + Verification (check/balance pair)
    ─ Engineer + Defense (build + secure)
    """

    PIPELINE_SPECS = [
        Specialization.PLANNER.value,
        Specialization.RESEARCHER.value,
        Specialization.CREATIVITY.value,
        Specialization.ENGINEER.value,
        Specialization.SYNTHESIZER.value,
        Specialization.VERIFICATION.value,
    ]

    def __init__(
        self,
        population: Population,
        memory: Optional[DistributedMemory] = None,
        llm_callback: Optional[Callable[[str, str], str]] = None,
        max_agents: int = 20,
        consensus_threshold: float = 0.6,
    ):
        self.population = population
        self.memory = memory or DistributedMemory()
        self.llm_callback = llm_callback
        self.max_agents = max_agents
        self.consensus_threshold = consensus_threshold
        self.task_history: list[SwarmTask] = []
        self.results_history: list[SwarmResult] = []
        self.run_count = 0

    # ── Entry Points ──────────────────────────────────────────────────────────

    def run_parallel(self, task: str, specs: list[str] = None) -> SwarmResult:
        """
        Parallel swarm: assign task to N specialist agents simultaneously.
        Each species processes from its own cognitive niche.
        """
        t0 = time.time()
        self.run_count += 1
        specs = specs or self.PIPELINE_SPECS

        subtasks = []
        for spec in specs:
            agent = self._best_agent_of(spec)
            if not agent:
                continue

            st = SwarmTask(content=task, required_spec=spec, assigned_agent_id=agent.id)
            think_result = agent.think(task)

            if think_result.get("status") == "died":
                st.fail("agent died during execution")
            else:
                output = self._delegate(task, spec)
                st.complete({
                    "output": output,
                    "confidence": think_result.get("confidence", 0.0),
                    "specialization": spec,
                    "agent_id": agent.id,
                    "generation": agent.genome.generation,
                })

            subtasks.append(st)
            self.task_history.append(st)

        result = self._aggregate(task, subtasks, time.time() - t0)
        self.results_history.append(result)
        self._store_to_memory(task, result)
        return result

    def run_tournament(self, task: str, contenders: int = 5) -> SwarmResult:
        """
        Tournament: multiple agents compete on same task. Best wins.
        Natural selection within a single reasoning step.
        """
        t0 = time.time()
        self.run_count += 1

        living = [a for a in self.population.agents if a.alive]
        if not living:
            return self._empty_result(task)

        import random
        candidates = random.sample(living, min(contenders, len(living)))
        subtasks = []

        for agent in candidates:
            st = SwarmTask(content=task, assigned_agent_id=agent.id,
                           required_spec=agent.genome.specialization)
            result = agent.think(task)
            if result.get("status") == "died":
                st.fail("agent died")
            else:
                output = self._delegate(task, agent.genome.specialization)
                st.complete({
                    "output": output,
                    "confidence": result.get("confidence", 0),
                    "specialization": agent.genome.specialization,
                    "agent_id": agent.id,
                    "generation": agent.genome.generation,
                })
            subtasks.append(st)
            self.task_history.append(st)

        # Winner: highest confidence
        winner = max(
            [s for s in subtasks if s.status == "done"],
            key=lambda s: s.result.get("confidence", 0),
            default=None
        )

        if winner:
            winner_agent = next((a for a in self.population.agents if a.id == winner.assigned_agent_id), None)
            if winner_agent:
                winner_agent.genome.fitness_score = min(
                    1.0, winner_agent.genome.fitness_score + 0.05
                )

        result = self._aggregate(task, subtasks, time.time() - t0, mode="tournament")
        self.results_history.append(result)
        return result

    def run_consensus(self, task: str, n_agents: int = 6) -> SwarmResult:
        """
        Consensus: N agents vote on a task. Confidence-weighted majority.
        Agreement score measures swarm coherence.
        """
        t0 = time.time()
        self.run_count += 1

        living = [a for a in self.population.agents if a.alive]
        if not living:
            return self._empty_result(task)

        import random
        panel = sorted(
            random.sample(living, min(n_agents, len(living))),
            key=lambda a: a.genome.fitness_score,
            reverse=True
        )

        subtasks = []
        confidences = []

        for agent in panel:
            st = SwarmTask(content=task, assigned_agent_id=agent.id,
                           required_spec=agent.genome.specialization)
            result = agent.think(task)
            conf = result.get("confidence", 0)
            confidences.append(conf)
            output = self._delegate(task, agent.genome.specialization)
            st.complete({
                "output": output,
                "confidence": conf,
                "specialization": agent.genome.specialization,
                "agent_id": agent.id,
            })
            subtasks.append(st)
            self.task_history.append(st)

        result = self._aggregate(task, subtasks, time.time() - t0, mode="consensus")
        self.results_history.append(result)
        return result

    def decompose_and_run(self, task: str) -> SwarmResult:
        """
        Decompose a complex task into subtasks, route each to best species.
        Planner decomposes → specialists execute → synthesizer integrates.
        """
        t0 = time.time()
        self.run_count += 1

        # Stage 1: Planner decomposes
        planner = self._best_agent_of(Specialization.PLANNER.value)
        subtask_contents = self._decompose(task, planner)

        # Stage 2: Route subtasks to specialists
        all_subtasks = []
        for sub_content in subtask_contents:
            spec = self._infer_specialization(sub_content)
            agent = self._best_agent_of(spec)
            if not agent:
                continue

            st = SwarmTask(content=sub_content, required_spec=spec,
                           assigned_agent_id=agent.id, parent_task_id=task[:20])
            think_result = agent.think(sub_content)
            output = self._delegate(sub_content, spec)
            st.complete({
                "output": output,
                "confidence": think_result.get("confidence", 0),
                "specialization": spec,
                "agent_id": agent.id,
            })
            all_subtasks.append(st)
            self.task_history.append(st)

        # Stage 3: Synthesizer integrates
        synthesizer = self._best_agent_of(Specialization.SYNTHESIZER.value)
        synthesis_input = " | ".join(
            s.result.get("output", "") for s in all_subtasks if s.status == "done"
        )
        final_output = self._delegate(f"Synthesize: {synthesis_input[:200]}", "synthesizer")

        result = self._aggregate(task, all_subtasks, time.time() - t0,
                                 mode="decompose", override_output=final_output)
        self.results_history.append(result)
        self._store_to_memory(task, result)
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _best_agent_of(self, spec: str) -> Optional[AgentCell]:
        candidates = [a for a in self.population.agents if a.alive and a.genome.specialization == spec]
        if not candidates:
            candidates = [a for a in self.population.agents if a.alive]
        return max(candidates, key=lambda a: a.genome.fitness_score) if candidates else None

    def _delegate(self, task: str, spec: str) -> str:
        if self.llm_callback:
            try:
                return self.llm_callback(task, spec)
            except Exception as e:
                return f"[llm_error: {e}]"
        return f"[{spec}] processed: {task[:60]}"

    def _decompose(self, task: str, planner: Optional[AgentCell]) -> list[str]:
        """Simple decomposition — in production, delegate to LLM."""
        if self.llm_callback and planner:
            prompt = f"Decompose this task into 3-5 specific subtasks. Return each on a new line:\n{task}"
            try:
                raw = self.llm_callback(prompt, "planner")
                lines = [l.strip("- •").strip() for l in raw.split("\n") if l.strip()]
                return lines[:5] if lines else [task]
            except Exception:
                pass
        # Fallback: create subtasks by role
        return [
            f"[plan] {task}",
            f"[research] {task}",
            f"[implement] {task}",
            f"[verify] {task}",
        ]

    def _infer_specialization(self, content: str) -> str:
        """Heuristic specialization routing from task content."""
        c = content.lower()
        if any(w in c for w in ["plan", "strategy", "goal", "decompose"]):
            return Specialization.PLANNER.value
        if any(w in c for w in ["research", "find", "analyze", "data"]):
            return Specialization.RESEARCHER.value
        if any(w in c for w in ["create", "invent", "design", "novel"]):
            return Specialization.CREATIVITY.value
        if any(w in c for w in ["verify", "check", "test", "validate"]):
            return Specialization.VERIFICATION.value
        if any(w in c for w in ["implement", "code", "build", "optimize"]):
            return Specialization.ENGINEER.value
        if any(w in c for w in ["synthesize", "combine", "integrate"]):
            return Specialization.SYNTHESIZER.value
        if any(w in c for w in ["secure", "defend", "protect"]):
            return Specialization.DEFENSE.value
        return Specialization.SYNTHESIZER.value

    def _aggregate(
        self,
        task: str,
        subtasks: list[SwarmTask],
        elapsed: float,
        mode: str = "parallel",
        override_output: str = None,
    ) -> SwarmResult:
        done = [s for s in subtasks if s.status == "done"]
        if not done:
            return self._empty_result(task)

        confidences = [s.result.get("confidence", 0) for s in done]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        variance = max(confidences) - min(confidences) if len(confidences) > 1 else 0
        agreement = round(1.0 - variance, 3)

        if override_output:
            consensus = override_output
        elif mode == "tournament":
            best = max(done, key=lambda s: s.result.get("confidence", 0))
            consensus = best.result.get("output", "")
        else:
            # Weight outputs by confidence
            consensus = max(done, key=lambda s: s.result.get("confidence", 0)).result.get("output", "")

        return SwarmResult(
            original_task=task,
            subtasks=subtasks,
            consensus_output=consensus,
            confidence=round(avg_conf, 3),
            species_used=list({s.required_spec for s in done if s.required_spec}),
            total_agents=len(subtasks),
            agreement_score=agreement,
            latency_ms=round(elapsed * 1000, 2),
            metadata={"mode": mode},
        )

    def _empty_result(self, task: str) -> SwarmResult:
        return SwarmResult(
            original_task=task, subtasks=[], consensus_output="[no agents available]",
            confidence=0, species_used=[], total_agents=0, agreement_score=0, latency_ms=0,
        )

    def _store_to_memory(self, task: str, result: SwarmResult):
        if self.memory:
            self.memory.remember(
                content=f"[swarm] {task[:60]} → conf={result.confidence:.2f} agree={result.agreement_score:.2f}",
                memory_type="episodic",
                tags=["swarm", "task"] + result.species_used,
                importance=result.confidence,
            )

    def metrics(self) -> dict:
        if not self.results_history:
            return {"total_runs": 0}
        confs = [r.confidence for r in self.results_history]
        agrees = [r.agreement_score for r in self.results_history]
        return {
            "total_runs":        self.run_count,
            "avg_confidence":    round(sum(confs) / len(confs), 3),
            "avg_agreement":     round(sum(agrees) / len(agrees), 3),
            "total_subtasks":    len(self.task_history),
            "failed_subtasks":   len([s for s in self.task_history if s.status == "failed"]),
            "species_frequency": self._species_freq(),
        }

    def _species_freq(self) -> dict:
        freq = defaultdict(int)
        for s in self.task_history:
            if s.required_spec:
                freq[s.required_spec] += 1
        return dict(freq)