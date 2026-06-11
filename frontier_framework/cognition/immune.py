"""
Frontier Framework — Immune System (Stage 16)
Cognitive predators, information parasites, and defense intelligences.
Patrols the network for: compute theft, reasoning corruption, fitness fraud, stagnation.
"""

from __future__ import annotations
import time
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frontier_framework.evolution.population import Population
    from frontier_framework.evolution.genome import AgentCell


@dataclass
class ThreatSignal:
    threat_type: str
    severity: float          # 0=low, 1=critical
    target_id: str
    description: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "threat_type": self.threat_type,
            "severity":    round(self.severity, 3),
            "target_id":   self.target_id,
            "description": self.description,
            "timestamp":   self.timestamp,
            "resolved":    self.resolved,
        }


class ImmuneSystem:
    """
    Defense intelligence layer.

    Threat classes:
      - ENERGY_PARASITE   : agent consuming disproportionate compute vs output
      - FITNESS_FRAUD     : agent with suspiciously fast fitness gains (exploit)
      - STAGNATION        : agent not evolving (genome locked, no mutations)
      - MUTATION_STORM    : runaway mutation rate destabilizing the population
      - MONOCULTURE       : species diversity collapse (extinction risk)
      - DEAD_WEIGHT       : agent alive but never completing tasks
    """

    THREAT_THRESHOLDS = {
        "ENERGY_PARASITE":  0.02,   # energy per task < 2%
        "FITNESS_FRAUD":    0.95,   # fitness > 95% in gen < 3
        "STAGNATION":       50,     # age > 50 with no children and 0 fitness gain
        "MUTATION_STORM":   0.25,   # mutation_rate > 25%
        "MONOCULTURE":      0.80,   # one species > 80% of population
        "DEAD_WEIGHT":      20,     # age > 20 with 0 tasks completed
    }

    def __init__(self):
        self.threats: list[ThreatSignal] = []
        self.quarantined: list[str] = []        # agent IDs under quarantine
        self.scans: int = 0
        self.neutralized: int = 0

    def scan(self, population: "Population") -> list[ThreatSignal]:
        """Full immune scan of the population. Returns new threats found."""
        self.scans += 1
        new_threats = []
        living = [a for a in population.agents if a.alive]

        if not living:
            return []

        new_threats.extend(self._detect_energy_parasites(living))
        new_threats.extend(self._detect_fitness_fraud(living))
        new_threats.extend(self._detect_stagnation(living))
        new_threats.extend(self._detect_mutation_storm(living))
        new_threats.extend(self._detect_monoculture(living))
        new_threats.extend(self._detect_dead_weight(living))

        self.threats.extend(new_threats)
        if len(self.threats) > 300:
            self.threats = self.threats[-300:]

        return new_threats

    def respond(self, population: "Population", threats: list[ThreatSignal]) -> dict:
        """
        Apply immune responses to detected threats.
        Graduated responses: warn → quarantine → terminate → diversify.
        """
        actions = []

        for threat in threats:
            if threat.resolved:
                continue

            agent = next((a for a in population.agents if a.id == threat.target_id), None)

            if threat.threat_type == "ENERGY_PARASITE" and agent:
                # Drain energy as penalty
                agent.energy *= 0.5
                threat.resolved = True
                actions.append(f"drained energy of parasite {agent.id}")

            elif threat.threat_type == "FITNESS_FRAUD" and agent:
                # Reset fitness to honest level
                agent.genome.fitness_score *= 0.3
                threat.resolved = True
                actions.append(f"reset fraudulent fitness of {agent.id}")

            elif threat.threat_type == "STAGNATION" and agent:
                # Force mutation to break stagnation
                agent.genome.mutation_rate = min(0.2, agent.genome.mutation_rate * 3)
                threat.resolved = True
                actions.append(f"forced mutation in stagnant agent {agent.id}")

            elif threat.threat_type == "MUTATION_STORM" and agent:
                # Clamp runaway mutation
                agent.genome.mutation_rate = 0.08
                threat.resolved = True
                actions.append(f"clamped mutation storm in {agent.id}")

            elif threat.threat_type == "MONOCULTURE":
                # Inject diverse agents
                self._inject_diversity(population)
                threat.resolved = True
                self.neutralized += 1
                actions.append("injected diversity to counter monoculture")

            elif threat.threat_type == "DEAD_WEIGHT" and agent:
                # Kill the agent
                agent.alive = False
                threat.resolved = True
                actions.append(f"terminated dead-weight agent {agent.id}")

            if threat.resolved:
                self.neutralized += 1

        return {
            "threats_processed": len(threats),
            "actions": actions,
            "total_neutralized": self.neutralized,
        }

    # ─── Threat Detectors ────────────────────────────────────────────────────

    def _detect_energy_parasites(self, agents: list) -> list[ThreatSignal]:
        threats = []
        for a in agents:
            if a.age > 5 and a.tasks_done_count() == 0 and a.energy < 0.3:
                threats.append(ThreatSignal(
                    "ENERGY_PARASITE", 0.4, a.id,
                    f"Agent consuming energy without task output (age={a.age})"
                ))
        return threats

    def _detect_fitness_fraud(self, agents: list) -> list[ThreatSignal]:
        threats = []
        for a in agents:
            if a.genome.generation < 3 and a.genome.fitness_score > 0.95:
                threats.append(ThreatSignal(
                    "FITNESS_FRAUD", 0.7, a.id,
                    f"Suspiciously high fitness {a.genome.fitness_score:.2f} in gen {a.genome.generation}"
                ))
        return threats

    def _detect_stagnation(self, agents: list) -> list[ThreatSignal]:
        threats = []
        for a in agents:
            if (a.age > self.THREAT_THRESHOLDS["STAGNATION"] and
                    a.children_count == 0 and
                    a.genome.fitness_score < 0.01):
                threats.append(ThreatSignal(
                    "STAGNATION", 0.5, a.id,
                    f"Agent stagnant for {a.age} cycles with no reproduction"
                ))
        return threats

    def _detect_mutation_storm(self, agents: list) -> list[ThreatSignal]:
        return [
            ThreatSignal(
                "MUTATION_STORM", 0.8, a.id,
                f"Mutation rate {a.genome.mutation_rate:.3f} exceeds safe threshold"
            )
            for a in agents
            if a.genome.mutation_rate > self.THREAT_THRESHOLDS["MUTATION_STORM"]
        ]

    def _detect_monoculture(self, agents: list) -> list[ThreatSignal]:
        if not agents:
            return []
        from collections import Counter
        counts = Counter(a.genome.specialization for a in agents)
        dominant_spec, dominant_count = counts.most_common(1)[0]
        dominance = dominant_count / len(agents)
        if dominance > self.THREAT_THRESHOLDS["MONOCULTURE"]:
            return [ThreatSignal(
                "MONOCULTURE", 0.9, "ecosystem",
                f"Species '{dominant_spec}' dominates at {dominance:.1%} — diversity collapse risk"
            )]
        return []

    def _detect_dead_weight(self, agents: list) -> list[ThreatSignal]:
        return [
            ThreatSignal(
                "DEAD_WEIGHT", 0.3, a.id,
                f"Agent age {a.age} with zero task completions"
            )
            for a in agents
            if a.age > self.THREAT_THRESHOLDS["DEAD_WEIGHT"] and len(a.task_history) == 0
        ]

    def _inject_diversity(self, population: "Population"):
        """Counter monoculture by spawning underrepresented species."""
        from frontier_framework.evolution.genome import Genome, Specialization
        from frontier_framework.evolution.genome import AgentCell as AC
        import random as _r

        present = {a.genome.specialization for a in population.agents if a.alive}
        all_specs = {s.value for s in Specialization}
        missing = all_specs - present

        for spec in list(missing)[:3]:
            genome = Genome(
                reasoning_style  = _r.random(),
                curiosity_level  = _r.random(),
                memory_retention = _r.random(),
                risk_tolerance   = _r.random(),
                specialization   = spec,
                mutation_rate    = _r.uniform(0.03, 0.08),
                cooperation_bias = _r.random(),
            )
            if len(population.agents) < population.max_size:
                population.agents.append(AC(genome=genome, energy=0.6))

    def report(self) -> dict:
        unresolved = [t for t in self.threats if not t.resolved]
        recent = sorted(self.threats[-20:], key=lambda t: t.timestamp, reverse=True)
        return {
            "total_scans":       self.scans,
            "total_threats":     len(self.threats),
            "unresolved":        len(unresolved),
            "neutralized":       self.neutralized,
            "quarantined_count": len(self.quarantined),
            "recent_threats":    [t.to_dict() for t in recent],
            "threat_breakdown":  self._breakdown(),
        }

    def _breakdown(self) -> dict:
        from collections import Counter
        counts = Counter(t.threat_type for t in self.threats)
        return dict(counts)


# Patch AgentCell with a helper the immune system needs
def _tasks_done_count(self):
    return len(self.task_history)

from frontier_framework.evolution.genome import AgentCell
AgentCell.tasks_done_count = _tasks_done_count