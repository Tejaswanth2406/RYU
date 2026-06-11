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
from dataclasses import dataclass, field
from typing import Optional

from frontier_framework.evolution.population import Population
from frontier_framework.evolution.genome import AgentCell, Genome, Specialization
from frontier_framework.memory.memory import DistributedMemory
from frontier_framework.cognition.immune import ImmuneSystem
from frontier_framework.cognition.knowledge import KnowledgeEcosystem


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
    ):
        self.id = str(uuid.uuid4())[:8]
        self.exchange_rate = exchange_rate
        self.cycle = 0
        self.created_at = time.time()
        self.collective_memory = DistributedMemory(max_entries=2000)
        self.event_log: list[dict] = []
        self.stats_history: list[dict] = []

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

        self._log("init", f"Civilization {self.id} initialized with {n_nodes} nodes")

    # ── Core Lifecycle ────────────────────────────────────────────────────────

    def tick(self) -> dict:
        """
        One civilization cycle:
        1. Each node evolves independently
        2. Knowledge exchange between connected nodes
        3. Elite migration (best agents move between nodes)
        4. Collective memory decay
        5. Compute civilization-level emergence score
        """
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
            self._log("migration", f"{len(emigrants)} elite agents: {strong.domain} → {weak.domain}")

    def _civilization_immune_scan(self):
        """Scan all nodes for threats simultaneously."""
        total_threats = 0
        for node in self.nodes.values():
            threats = node.immune_scan()
            total_threats += len(threats)
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
        all_fitnesses = []
        for n in self.nodes.values():
            all_fitnesses.extend(a.genome.fitness_score for a in n.population.agents if a.alive)

        avg_fitness = sum(all_fitnesses) / max(1, len(all_fitnesses))

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

    def status(self) -> dict:
        stats = self._compute_stats([])
        stats["recent_events"] = self.event_log[-10:]
        stats["stats_history"] = self.stats_history[-30:]
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