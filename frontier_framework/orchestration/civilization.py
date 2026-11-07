"""
Frontier Framework — AI Civilization (Stage 9)
Multiple populations → organism → species → civilization.

One frontier model becomes a Cell.
Thousands become an Organism.
Millions become a Species.
Billions become a Civilization.

The civilization itself becomes the intelligence.
No single model is "the AI."
"""

from __future__ import annotations
import time
import uuid
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from frontier_framework.evolution.population import Population
from frontier_framework.evolution.genome import AgentCell, Genome, Specialization
from frontier_framework.memory.memory import DistributedMemory
from frontier_framework.cognition.immune import ImmuneSystem
from frontier_framework.cognition.knowledge import KnowledgeEcosystem
from frontier_framework.biology.integration import BiologicalLifecycle
from frontier_framework.simulation import SimulatedEnvironment
from frontier_framework.simulation.fitness_experiment import FitnessExperiment


@dataclass
class CivilizationNode:
    """
    One population within the civilization — analogous to a city-state or brain region.
    Each node specializes in a cognitive domain.
    Nodes exchange knowledge and the fittest agents migrate between them.
    """
    id: str                         = field(default_factory=lambda: str(uuid.uuid4())[:6])
    name: str                       = "unnamed"
    domain: str                     = "general"         # science, engineering, art, etc.
    population: Population          = field(default_factory=lambda: Population(10, 50))
    immune: ImmuneSystem            = field(default_factory=ImmuneSystem)
    knowledge: KnowledgeEcosystem   = field(default_factory=lambda: KnowledgeEcosystem(10))
    created_at: float               = field(default_factory=time.time)
    migrations_in: int              = 0
    migrations_out: int             = 0
    exchange_log: list              = field(default_factory=list)

    def tick(self, tasks: list = None) -> dict:
        stats = self.population.tick(tasks or [self.domain, "analyze", "synthesize"])
        self.knowledge.tick()
        return {"node": self.id, "domain": self.domain, **stats}

    def immune_scan(self):
        threats = self.immune.scan(self.population)
        self.immune.respond(self.population, threats)
        return threats

    def export_elite(self, n: int = 3) -> list[AgentCell]:
        """Export top-fitness agents for migration."""
        living = sorted(
            [a for a in self.population.agents if a.alive],
            key=lambda a: a.genome.fitness_score,
            reverse=True
        )
        emigrants = living[:n]
        for a in emigrants:
            self.population.agents.remove(a)
            self.migrations_out += 1
        return emigrants

    def import_agents(self, agents: list[AgentCell]):
        """Receive migrating agents from another node."""
        for agent in agents:
            if len(self.population.agents) < self.population.max_size:
                self.population.agents.append(agent)
                self.migrations_in += 1

    def status(self) -> dict:
        pop_status = self.population.status()
        return {
            "id":            self.id,
            "name":          self.name,
            "domain":        self.domain,
            "alive_agents":  pop_status.get("alive_count", 0),
            "avg_fitness":   pop_status.get("avg_fitness", 0),
            "max_generation":pop_status.get("max_generation", 0),
            "migrations_in": self.migrations_in,
            "migrations_out":self.migrations_out,
            "knowledge":     self.knowledge.status()["total_ideas"],
            "uptime_s":      round(time.time() - self.created_at, 1),
        }


class Civilization:
    """
    Stage 9: AI Civilization.
    A network of specialized population nodes.
    The civilization exhibits collective intelligence no single node possesses.

    Features:
    ─ Multiple domain nodes (science, engineering, art, governance, etc.)
    ─ Knowledge exchange between nodes (idea migration)
    ─ Elite agent migration (best minds move to frontier nodes)
    ─ Collective memory shared across all nodes
    ─ Civilization-level immune system
    ─ Emergence score: intelligence beyond the sum of parts
    """

    DOMAINS = [
        "science", "engineering", "art", "governance",
        "mathematics", "philosophy", "strategy", "defense"
    ]

    def __init__(
        self,
        n_nodes: int = 4,
        node_population_size: int = 15,
        exchange_rate: float = 0.1,
        fitness_experiment: Optional[FitnessExperiment] = None,
        forecast_horizon: int = 1,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.exchange_rate = exchange_rate
        self.cycle = 0
        self.created_at = time.time()
        self.collective_memory = DistributedMemory(max_entries=2000)
        self.event_log: list[dict] = []
        self.stats_history: list[dict] = []
        self.biological_lifecycle = BiologicalLifecycle()
        self.simulation = SimulatedEnvironment()
        self.fitness_experiment = fitness_experiment or FitnessExperiment()
        self.forecast_horizon = max(1, forecast_horizon)

        # Spawn nodes
        domains = self.DOMAINS[:n_nodes]
        self.nodes: dict[str, CivilizationNode] = {}
        for domain in domains:
            node = CivilizationNode(
                name=f"{domain.capitalize()} Node",
                domain=domain,
                population=Population(
                    initial_size=node_population_size,
                    max_size=node_population_size * 3,
                ),
                knowledge=KnowledgeEcosystem(seed_ideas=15),
            )
            # Seed with domain-appropriate specializations
            self._seed_node_specialization(node, domain)
            self.nodes[node.id] = node

        # Track N0 baseline (initial active population)
        self.initial_population = self.total_agents()
        self._pending_observations: deque[dict] = deque()
        self._active_directed_channels: set[tuple[str, str]] = set()
        self._recent_threats: list = []

        self._log("init", f"Civilization {self.id} initialized with {n_nodes} nodes (N0={self.initial_population})")

    # ── Core Lifecycle ────────────────────────────────────────────────────────

    def tick(self) -> dict:
        """
        One civilization cycle:
        1. Queue snapshot of current features at cycle t for resolution at t + H.
        2. Evolve nodes, exchange knowledge, migrate agents, scan immune defenses.
        3. Resolve any pending snapshots whose target cycle is reached (t + H).
        """
        pre_cycle = self.cycle
        pre_fitness = self._average_fitness()
        pre_features = self._fitness_experiment_features(pre_fitness)

        # Queue snapshot for t + H resolution
        self._pending_observations.append({
            "run_id": self.id,
            "cycle": pre_cycle,
            "features": pre_features,
            "target_cycle": pre_cycle + self.forecast_horizon,
            "initial_population": self.initial_population,
        })

        self.cycle += 1
        node_stats = []

        # 1. Node evolution
        for node in self.nodes.values():
            stats = node.tick()
            node_stats.append(stats)

        # 2. Knowledge exchange
        if random.random() < self.exchange_rate:
            self._knowledge_exchange()

        # 3. Agent migration
        if self.cycle % 5 == 0:
            self._elite_migration()

        # 4. Immune scan across all nodes
        if self.cycle % 10 == 0:
            self._civilization_immune_scan()

        # 5. Memory tick
        self.collective_memory.tick()

        stats = self._compute_stats(node_stats)
        self.stats_history.append(stats)

        post_fitness = self._average_fitness()
        post_agents = self.total_agents()

        # Resolve any pending snapshots whose target cycle is reached (t + H)
        while self._pending_observations and self._pending_observations[0]["target_cycle"] <= self.cycle:
            pending = self._pending_observations.popleft()
            self.fitness_experiment.record(
                run_id=pending["run_id"],
                cycle=pending["cycle"],
                features=pending["features"],
                target_avg_fitness=post_fitness,
                horizon=self.forecast_horizon,
                initial_population=pending["initial_population"],
                target_active_population=post_agents,
            )

        return stats

    def _knowledge_exchange(self):
        """Ideas migrate between nodes — like academic cross-pollination."""
        node_list = list(self.nodes.values())
        if len(node_list) < 2:
            return

        a, b = random.sample(node_list, 2)
        top_a = a.knowledge.query(top_k=2)
        top_b = b.knowledge.query(top_k=2)

        for idea in top_a:
            b.knowledge.inject(f"[from {a.domain}] {idea.content[:80]}", a.domain)
        for idea in top_b:
            a.knowledge.inject(f"[from {b.domain}] {idea.content[:80]}", b.domain)

        self._active_directed_channels.add((a.id, b.id))
        self._active_directed_channels.add((b.id, a.id))
        self._log("knowledge_exchange", f"Ideas exchanged between {a.domain} ↔ {b.domain}")

    def _elite_migration(self):
        """Best agents migrate to expand frontier nodes."""
        node_list = list(self.nodes.values())
        if len(node_list) < 2:
            return

        # Find strongest and weakest by avg fitness
        ranked = sorted(
            node_list,
            key=lambda n: n.population.status().get("avg_fitness", 0),
            reverse=True
        )
        strong = ranked[0]
        weak = ranked[-1]

        if strong.id == weak.id:
            return

        # Migrate 2 elite agents from strong → weak (idea diffusion)
        emigrants = strong.export_elite(n=2)
        if emigrants:
            weak.import_agents(emigrants)
            self._active_directed_channels.add((strong.id, weak.id))
            self._log("migration", f"{len(emigrants)} elite agents: {strong.domain} → {weak.domain}")

    def _civilization_immune_scan(self):
        """Scan all nodes for threats simultaneously."""
        total_threats = 0
        self._recent_threats.clear()
        for node in self.nodes.values():
            threats = node.immune_scan()
            total_threats += len(threats)
            self._recent_threats.extend(threats)
        if total_threats:
            self._log("immune", f"Civilization scan: {total_threats} threats neutralized across all nodes")
        if total_threats:
            self._log("immune", f"Civilization scan: {total_threats} threats neutralized across all nodes")

    def _seed_node_specialization(self, node: CivilizationNode, domain: str):
        """Bias agent specializations toward the node's domain."""
        domain_spec_map = {
            "science":      [Specialization.RESEARCHER, Specialization.VERIFICATION],
            "engineering":  [Specialization.ENGINEER, Specialization.PLANNER],
            "art":          [Specialization.CREATIVITY, Specialization.SYNTHESIZER],
            "governance":   [Specialization.PLANNER, Specialization.VERIFICATION],
            "mathematics":  [Specialization.RESEARCHER, Specialization.VERIFICATION],
            "philosophy":   [Specialization.CREATIVITY, Specialization.MEMORY],
            "strategy":     [Specialization.PLANNER, Specialization.SYNTHESIZER],
            "defense":      [Specialization.DEFENSE, Specialization.VERIFICATION],
        }
        preferred = domain_spec_map.get(domain, list(Specialization))
        for agent in node.population.agents:
            if random.random() < 0.6:
                agent.genome.specialization = random.choice(preferred).value

    # ── Queries ───────────────────────────────────────────────────────────────

    def query_all(self, task: str) -> dict:
        """Broadcast a task to all nodes. Return per-node responses."""
        responses = {}
        for node in self.nodes.values():
            living = [a for a in node.population.agents if a.alive]
            if living:
                best = max(living, key=lambda a: a.genome.fitness_score)
                result = best.think(task)
                responses[node.domain] = {
                    "agent_id":      best.id,
                    "specialization":best.genome.specialization,
                    "confidence":    result.get("confidence", 0),
                    "generation":    best.genome.generation,
                }
        return responses

    def total_agents(self) -> int:
        return sum(
            len([a for a in n.population.agents if a.alive])
            for n in self.nodes.values()
        )

    def _compute_stats(self, node_stats: list) -> dict:
        total_alive = self.total_agents()
        avg_fitness = self._average_fitness()

        # Emergence: collective fitness exceeds sum-of-parts heuristic
        emergence_score = round(avg_fitness * len(self.nodes) * 0.1, 4)

        stats = {
            "cycle":           self.cycle,
            "total_agents":    total_alive,
            "node_count":      len(self.nodes),
            "avg_fitness":     round(avg_fitness, 4),
            "emergence_score": emergence_score,
            "memory_alive":    self.collective_memory.status()["total_alive"],
            "uptime_s":        round(time.time() - self.created_at, 1),
            "nodes":           [n.status() for n in self.nodes.values()],
        }
        return stats

    def _average_fitness(self) -> float:
        fitnesses = [
            agent.genome.fitness_score
            for node in self.nodes.values()
            for agent in node.population.agents
            if agent.alive
        ]
        return sum(fitnesses) / len(fitnesses) if fitnesses else 0.0

    def _fitness_experiment_features(self, avg_fitness: float) -> dict[str, float]:
        living_agents = [
            a for node in self.nodes.values()
            for a in node.population.agents
            if a.alive
        ]
        n_t = len(living_agents)
        n0 = max(1, getattr(self, "initial_population", n_t))

        # 1. active_ratio: min(1.0, Nt / N0)
        active_ratio = min(1.0, n_t / n0) if n0 > 0 else 0.0

        # 2. avg_energy: mean metabolic energy of living agents in [0, 1]
        avg_energy = (
            sum(max(0.0, min(1.0, a.energy)) for a in living_agents) / n_t
            if n_t else 0.0
        )

        # 3. avg_damage: biological aging and cellular wear in [0, 1]
        avg_damage = (
            sum(min(1.0, a.age / max(1, a.lifespan)) for a in living_agents) / n_t
            if n_t else 0.0
        )

        # 4. graph_density: directed inter-node channels without self-loops in [0, 1]
        m = len(self.nodes)
        max_channels = m * (m - 1)
        active_channels = getattr(self, "_active_directed_channels", set())
        graph_density = (
            min(1.0, len(active_channels) / max_channels)
            if max_channels > 0 else 0.0
        )

        # 5. risk_level: active immune threat severity in [0, 1]
        recent_threats = getattr(self, "_recent_threats", [])
        if recent_threats:
            mean_sev = sum(getattr(t, "severity", 0.5) for t in recent_threats) / len(recent_threats)
            risk_level = min(1.0, mean_sev * min(1.0, len(recent_threats) / 5.0))
        else:
            risk_level = 0.0

        # 6. emergence_score: bounded collective synergy in [0, 1]
        mem_status = self.collective_memory.status()
        mem_ratio = min(1.0, mem_status.get("total_alive", 0) / 2000.0)
        emergence_score = min(1.0, avg_fitness * (0.5 + 0.5 * mem_ratio))

        return {
            # Canonical 7 features
            "avg_fitness": round(avg_fitness, 4),
            "active_ratio": round(active_ratio, 4),
            "avg_energy": round(avg_energy, 4),
            "avg_damage": round(avg_damage, 4),
            "graph_density": round(graph_density, 4),
            "risk_level": round(risk_level, 4),
            "emergence_score": round(emergence_score, 4),
            # Supplementary simulation context
            "cycle": float(self.cycle),
            "node_count": float(len(self.nodes)),
            "total_agents": float(n_t),
            "memory_alive": float(mem_status.get("total_alive", 0)),
            "uptime_s": max(0.0, round(time.time() - self.created_at, 1)),
        }

    def biological_profile(self) -> dict:
        """Expose the biological lifecycle mapped onto the civilization architecture."""
        node_domains = [node.domain for node in self.nodes.values()]
        return {
            "lifecycle_stages": self.biological_lifecycle.stage_summary(),
            "active_domains": node_domains,
            "domain_count": len(node_domains),
            "model": "bio-inspired frontier civilization",
        }

    def run_simulation(self) -> dict:
        """Run an ML-driven simulation pass over the civilization's current state."""
        result = self.simulation.simulate(self)
        return {
            "scenario": result.scenario,
            "ensemble_score": result.ensemble_score,
            "model_predictions": result.model_predictions,
            "metrics": result.metrics,
        }

    def status(self) -> dict:
        stats = self._compute_stats([])
        stats["recent_events"] = self.event_log[-10:]
        stats["stats_history"] = self.stats_history[-30:]
        stats["biological_profile"] = self.biological_profile()
        return stats

    def _log(self, event_type: str, msg: str):
        self.event_log.append({
            "ts":    round(time.time(), 3),
            "cycle": self.cycle,
            "type":  event_type,
            "msg":   msg,
        })
        if len(self.event_log) > 300:
            self.event_log = self.event_log[-300:]