"""
Frontier Framework — Population Ecosystem
Biological analog: the organism / rainforest
Manages agent lifecycles, selection pressure, cognitive mitosis, species niches.
"""

from __future__ import annotations
import random
import time
from typing import Optional
from collections import defaultdict

from frontier_framework.evolution.genome import AgentCell, Genome, Specialization


class Population:
    """
    A living ecosystem of agent cells.
    Natural selection runs continuously: bad agents die, good ones reproduce.
    Cognitive mitosis: hard problems spawn exploration branches that recombine.
    """

    def __init__(
        self,
        initial_size: int = 20,
        max_size: int = 100,
        selection_pressure: float = 0.3,
        mitosis_threshold: float = 0.7,
    ):
        self.max_size = max_size
        self.selection_pressure = selection_pressure
        self.mitosis_threshold = mitosis_threshold
        self.generation = 0
        self.cycle = 0
        self.agents: list[AgentCell] = []
        self.graveyard: list[dict] = []   # dead agent records
        self.event_log: list[dict] = []
        self.stats_history: list[dict] = []
        self.created_at = time.time()

        self._seed_population(initial_size)

    # ─── Seeding ────────────────────────────────────────────────────────────

    def _seed_population(self, n: int):
        """Create diverse founding population across all specializations."""
        specs = list(Specialization)
        for i in range(n):
            genome = Genome(
                reasoning_style  = random.random(),
                curiosity_level  = random.random(),
                memory_retention = random.random(),
                risk_tolerance   = random.random(),
                specialization   = specs[i % len(specs)].value,
                mutation_rate    = random.uniform(0.02, 0.12),
                cooperation_bias = random.random(),
            )
            self.agents.append(AgentCell(genome=genome, energy=random.uniform(0.4, 0.9)))
        self._log("seed", f"Population seeded with {n} agents")

    # ─── Core Lifecycle ─────────────────────────────────────────────────────

    def tick(self, tasks: list[str] = None) -> dict:
        """
        One evolutionary cycle:
        1. Assign tasks → agents think → fitness updates
        2. Kill weak agents (selection pressure)
        3. Reproduce fit agents
        4. Enforce population ceiling
        """
        self.cycle += 1
        tasks = tasks or ["analyze", "plan", "verify", "create", "synthesize"]
        results = []

        # 1. Each living agent thinks
        living = [a for a in self.agents if a.alive]
        for agent in living:
            task = random.choice(tasks)
            result = agent.think(task)
            results.append(result)

        # 2. Natural selection — cull weakest
        died_natural = self._natural_death()

        # 3. Selection culling
        culled = self._selection_cull()

        # 4. Reproduction
        born = self._reproduction_cycle()

        # 5. Cap population
        self._enforce_ceiling()

        stats = self._compute_stats()
        stats["cycle"] = self.cycle
        stats["born"] = born
        stats["died"] = died_natural + culled
        stats["task_count"] = len(results)
        self.stats_history.append(stats)

        self._log("tick", f"Cycle {self.cycle}: +{born} born, -{died_natural + culled} died, pop={stats['alive_count']}")
        return stats

    def _natural_death(self) -> int:
        """Agents with zero energy or exceeded lifespan die naturally."""
        dead = [a for a in self.agents if not a.alive]
        for a in dead:
            self.graveyard.append({**a.to_dict(), "cause": "natural"})
        self.agents = [a for a in self.agents if a.alive]
        return len(dead)

    def _selection_cull(self) -> int:
        """
        Kill the bottom N% by fitness — selection pressure.
        Death is necessary. The civilization stays adaptive.
        """
        living = [a for a in self.agents if a.alive]
        if len(living) < 5:
            return 0

        n_cull = max(1, int(len(living) * self.selection_pressure * 0.15))
        sorted_agents = sorted(living, key=lambda a: a.genome.fitness_score)
        culled = sorted_agents[:n_cull]

        for a in culled:
            a.alive = False
            self.graveyard.append({**a.to_dict(), "cause": "selection"})

        self.agents = [a for a in self.agents if a.alive]
        return len(culled)

    def _reproduction_cycle(self) -> int:
        """
        Fit agents reproduce. High cooperation_bias → crossover.
        Low cooperation_bias → asexual mitosis.
        """
        reproducers = [a for a in self.agents if a.can_reproduce()]
        new_agents = []
        born = 0

        for agent in reproducers:
            if len(self.agents) + len(new_agents) >= self.max_size:
                break

            if agent.genome.cooperation_bias > 0.5 and len(reproducers) > 1:
                # Sexual reproduction: find best partner of different specialization
                candidates = [p for p in reproducers if p.id != agent.id and
                              p.genome.specialization != agent.genome.specialization]
                partner = max(candidates, key=lambda p: p.genome.fitness_score) if candidates else None
                child = agent.reproduce(partner)
            else:
                child = agent.reproduce()

            new_agents.append(child)
            born += 1

        self.agents.extend(new_agents)
        if born > 0:
            self._log("reproduction", f"{born} agents born in generation {self.generation}")
        return born

    def _enforce_ceiling(self):
        """Hard cap: trim lowest-fitness agents if over max_size."""
        if len(self.agents) > self.max_size:
            self.agents.sort(key=lambda a: a.genome.fitness_score, reverse=True)
            trimmed = self.agents[self.max_size:]
            for a in trimmed:
                self.graveyard.append({**a.to_dict(), "cause": "overcrowding"})
            self.agents = self.agents[:self.max_size]

    # ─── Cognitive Mitosis ───────────────────────────────────────────────────

    def cognitive_mitosis(self, problem: str, branches: int = 5) -> dict:
        """
        Stage 7: Cognitive Mitosis.
        Hard problem → spawn N exploratory agents → let them evolve → harvest best.
        Thinking becomes a form of accelerated evolution.
        """
        if not self.agents:
            return {"error": "no agents to split"}

        seed = max(self.agents, key=lambda a: a.genome.fitness_score)
        explorers = []

        for _ in range(branches):
            explorer = AgentCell(
                genome=seed.genome.mutate(),
                energy=0.6,
                lifespan=10,
            )
            result = explorer.think(problem)
            explorers.append((explorer, result))

        # Sort by confidence — natural selection among minds
        explorers.sort(key=lambda x: x[1].get("confidence", 0), reverse=True)
        survivors = explorers[:max(1, branches // 3)]

        # Best survivors merge their insights back into the main population
        for explorer, _ in survivors:
            if len(self.agents) < self.max_size:
                self.agents.append(explorer)

        best = survivors[0][1] if survivors else {}
        self._log("mitosis", f"Cognitive mitosis on '{problem}': {branches} branches → {len(survivors)} survived")

        return {
            "problem": problem,
            "branches_spawned": branches,
            "survivors": len(survivors),
            "best_confidence": best.get("confidence", 0),
            "best_specialization": survivors[0][0].genome.specialization if survivors else None,
            "explorers": [e.to_dict() for e, _ in survivors],
        }

    # ─── Ecosystem Queries ───────────────────────────────────────────────────

    def get_species_map(self) -> dict:
        """Group living agents by specialization — the species niches."""
        niches = defaultdict(list)
        for a in self.agents:
            niches[a.genome.specialization].append(a)
        return {
            spec: {
                "count": len(agents),
                "avg_fitness": round(sum(a.genome.fitness_score for a in agents) / len(agents), 4),
                "avg_energy":  round(sum(a.energy for a in agents) / len(agents), 3),
                "avg_generation": round(sum(a.genome.generation for a in agents) / len(agents), 1),
            }
            for spec, agents in niches.items()
        }

    def _compute_stats(self) -> dict:
        living = [a for a in self.agents if a.alive]
        if not living:
            return {"alive_count": 0, "avg_fitness": 0, "avg_energy": 0, "generation": self.generation}

        fitnesses = [a.genome.fitness_score for a in living]
        generations = [a.genome.generation for a in living]

        return {
            "alive_count":   len(living),
            "total_born":    sum(a.children_count for a in living),
            "graveyard_size": len(self.graveyard),
            "avg_fitness":   round(sum(fitnesses) / len(living), 4),
            "max_fitness":   round(max(fitnesses), 4),
            "avg_energy":    round(sum(a.energy for a in living) / len(living), 3),
            "avg_generation": round(sum(generations) / len(generations), 1),
            "max_generation": max(generations),
            "species_count": len(set(a.genome.specialization for a in living)),
            "generation":    self.generation,
            "uptime_s":      round(time.time() - self.created_at, 1),
        }

    def _log(self, event_type: str, message: str):
        self.event_log.append({
            "ts": round(time.time(), 3),
            "cycle": self.cycle,
            "type": event_type,
            "msg": message,
        })
        if len(self.event_log) > 500:
            self.event_log = self.event_log[-500:]

    def status(self) -> dict:
        stats = self._compute_stats()
        stats["species_map"] = self.get_species_map()
        stats["recent_events"] = self.event_log[-10:]
        stats["stats_history"] = self.stats_history[-50:]
        return stats