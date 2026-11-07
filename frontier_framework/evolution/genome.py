"""
Frontier Framework — Genome & Agent Cell
Biological analog: DNA + cellular organism
Each agent carries heritable traits that guide cognition, evolve under selection pressure.
"""

from __future__ import annotations
import uuid
import random
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Specialization(str, Enum):
    PLANNER     = "planner"
    MEMORY      = "memory"
    CREATIVITY  = "creativity"
    VERIFICATION= "verification"
    DEFENSE     = "defense"
    RESEARCHER  = "researcher"
    ENGINEER    = "engineer"
    SYNTHESIZER = "synthesizer"


@dataclass
class Genome:
    """
    The heritable blueprint of an agent cell.
    Analogous to DNA — controls cognition style, energy budget, reproduction.
    """
    reasoning_style: float      = 0.5   # 0=deductive, 1=abductive
    curiosity_level: float      = 0.5   # drives exploration vs exploitation
    memory_retention: float     = 0.5   # how much context an agent preserves
    risk_tolerance: float       = 0.5   # willingness to commit to uncertain paths
    specialization: str         = Specialization.PLANNER
    mutation_rate: float        = 0.05  # probability of trait mutation on reproduction
    cooperation_bias: float     = 0.5   # tendency to merge vs compete

    # Fitness tracking (not heritable, computed)
    fitness_score: float        = 0.0
    generation: int             = 0

    def mutate(self) -> "Genome":
        """Produce a mutated child genome. Core of evolution."""
        def _mutate(val: float) -> float:
            if random.random() < self.mutation_rate:
                delta = random.gauss(0, 0.1)
                return max(0.0, min(1.0, val + delta))
            return val

        return Genome(
            reasoning_style  = _mutate(self.reasoning_style),
            curiosity_level  = _mutate(self.curiosity_level),
            memory_retention = _mutate(self.memory_retention),
            risk_tolerance   = _mutate(self.risk_tolerance),
            specialization   = self.specialization if random.random() > self.mutation_rate
                               else random.choice(list(Specialization)).value,
            mutation_rate    = max(0.001, _mutate(self.mutation_rate)),
            cooperation_bias = _mutate(self.cooperation_bias),
            generation       = self.generation + 1,
        )

    @classmethod
    def crossover(cls, parent_a: "Genome", parent_b: "Genome") -> "Genome":
        """Sexual recombination — blend two genomes."""
        def _blend(a: float, b: float) -> float:
            w = random.random()
            return w * a + (1 - w) * b

        dominant = parent_a if parent_a.fitness_score >= parent_b.fitness_score else parent_b
        return cls(
            reasoning_style  = _blend(parent_a.reasoning_style, parent_b.reasoning_style),
            curiosity_level  = _blend(parent_a.curiosity_level, parent_b.curiosity_level),
            memory_retention = _blend(parent_a.memory_retention, parent_b.memory_retention),
            risk_tolerance   = _blend(parent_a.risk_tolerance, parent_b.risk_tolerance),
            specialization   = dominant.specialization,
            mutation_rate    = _blend(parent_a.mutation_rate, parent_b.mutation_rate),
            cooperation_bias = _blend(parent_a.cooperation_bias, parent_b.cooperation_bias),
            generation       = max(parent_a.generation, parent_b.generation) + 1,
        )

    def to_dict(self) -> dict:
        return {
            "reasoning_style":  round(self.reasoning_style, 3),
            "curiosity_level":  round(self.curiosity_level, 3),
            "memory_retention": round(self.memory_retention, 3),
            "risk_tolerance":   round(self.risk_tolerance, 3),
            "specialization":   self.specialization,
            "mutation_rate":    round(self.mutation_rate, 4),
            "cooperation_bias": round(self.cooperation_bias, 3),
            "fitness_score":    round(self.fitness_score, 4),
            "generation":       self.generation,
        }


@dataclass
class AgentCell:
    """
    A living cognitive unit. Born from a genome, consumes compute (energy),
    reasons on tasks, can reproduce, can die.
    """
    id: str                         = field(default_factory=lambda: str(uuid.uuid4())[:8])
    genome: Genome                  = field(default_factory=Genome)
    energy: float                   = 1.0       # compute budget (FLOPs analog), 0 = dead
    lifespan: int                   = 100       # max reasoning cycles
    age: int                        = 0
    alive: bool                     = True
    memory: list                    = field(default_factory=list)
    task_history: list              = field(default_factory=list)
    children_count: int             = 0
    created_at: float               = field(default_factory=time.time)

    # Thresholds
    REPRODUCTION_ENERGY: float      = 0.75
    DEATH_ENERGY: float             = 0.05
    ENERGY_PER_THOUGHT: float       = 0.08

    def think(self, task: str, context: str = "") -> dict:
        """
        Execute a reasoning step. Costs energy. Returns structured output.
        Energy model: bad agents starve, good agents earn resources.
        """
        if not self.alive:
            return {"error": "agent is dead", "id": self.id}

        self.energy -= self.ENERGY_PER_THOUGHT
        self.age += 1

        if self.energy <= self.DEATH_ENERGY or self.age >= self.lifespan:
            self.alive = False
            return {"status": "died", "id": self.id, "cause": "energy_exhaustion" if self.energy <= self.DEATH_ENERGY else "old_age"}

        # Reasoning quality shaped by genome traits
        confidence = (self.genome.reasoning_style * 0.4 +
                      self.genome.curiosity_level * 0.3 +
                      (1 - self.genome.risk_tolerance) * 0.3)

        result = {
            "id":           self.id,
            "specialization": self.genome.specialization,
            "task":         task,
            "confidence":   round(confidence, 3),
            "energy_left":  round(self.energy, 3),
            "generation":   self.genome.generation,
            "age":          self.age,
            "output":       self._generate_reasoning(task, confidence),
        }

        self.task_history.append(result)
        if len(self.memory) < int(self.genome.memory_retention * 20):
            self.memory.append({"task": task, "ts": time.time()})

        # Fitness update: reward successful reasoning
        self.genome.fitness_score = min(1.0, self.genome.fitness_score + confidence * 0.01)
        self.energy = min(1.0, self.energy + confidence * 0.03)  # good thinking earns back energy

        return result

    def tasks_done_count(self) -> int:
        """Number of completed reasoning steps for this agent."""
        return len(self.task_history)

    def can_reproduce(self) -> bool:
        return self.alive and self.energy >= self.REPRODUCTION_ENERGY

    def reproduce(self, partner: Optional["AgentCell"] = None) -> "AgentCell":
        """Cognitive mitosis — spawn a descendant."""
        if partner:
            child_genome = Genome.crossover(self.genome, partner.genome).mutate()
        else:
            child_genome = self.genome.mutate()

        self.energy *= 0.6  # reproduction costs energy
        self.children_count += 1

        return AgentCell(
            genome=child_genome,
            energy=0.5,
        )

    def _generate_reasoning(self, task: str, confidence: float) -> str:
        """Produce a reasoning descriptor shaped by genome."""
        style = "deductive" if self.genome.reasoning_style < 0.4 else \
                "abductive" if self.genome.reasoning_style > 0.7 else "inductive"
        curiosity = "deep-exploration" if self.genome.curiosity_level > 0.7 else \
                    "focused" if self.genome.curiosity_level < 0.3 else "balanced"
        return f"[{self.genome.specialization}:{style}:{curiosity}] confidence={confidence:.2f}"

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "alive":          self.alive,
            "energy":         round(self.energy, 3),
            "age":            self.age,
            "lifespan":       self.lifespan,
            "children_count": self.children_count,
            "genome":         self.genome.to_dict(),
            "memory_size":    len(self.memory),
            "tasks_done":     len(self.task_history),
            "created_at":     self.created_at,
        }