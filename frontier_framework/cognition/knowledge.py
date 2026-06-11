"""
Frontier Framework — Knowledge Ecosystem (Stage Ω-2)
Knowledge itself becomes a living species.
Ideas are born, mutate, compete, and go extinct.
The strongest ideas reproduce. Weak ideas die.
Science and mathematics become ecosystems.
"""

from __future__ import annotations
import uuid
import time
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Idea:
    """
    A knowledge unit that lives, reproduces, and can go extinct.
    Analogous to a meme or scientific hypothesis as a living entity.
    """
    id: str                 = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str            = ""
    domain: str             = "general"         # science, engineering, art, governance...
    fitness: float          = 0.5               # reproductive success
    novelty: float          = 0.5               # how different from parent
    utility: float          = 0.5               # usefulness to the ecosystem
    generation: int         = 0
    parent_ids: list        = field(default_factory=list)
    children_ids: list      = field(default_factory=list)
    alive: bool             = True
    born_at: float          = field(default_factory=time.time)
    died_at: Optional[float]= None
    citations: int          = 0                 # how often this idea is referenced

    def cite(self):
        """Another agent referenced this idea — increases fitness."""
        self.citations += 1
        self.fitness = min(1.0, self.fitness + 0.02)

    def age_decay(self, rate: float = 0.005):
        """Ideas decay over time without citations."""
        self.fitness = max(0.0, self.fitness - rate)
        if self.fitness <= 0.05:
            self.die()

    def die(self):
        self.alive = False
        self.died_at = time.time()

    def mutate(self, mutation_strength: float = 0.15) -> "Idea":
        """Produce a variant idea — mutation drives novelty."""
        child = Idea(
            content     = f"{self.content} [variant-{random.randint(100,999)}]",
            domain      = self.domain,
            fitness     = max(0.1, self.fitness + random.gauss(0, mutation_strength)),
            novelty     = min(1.0, self.novelty + random.uniform(0, 0.2)),
            utility     = max(0.0, self.utility + random.gauss(0, 0.1)),
            generation  = self.generation + 1,
            parent_ids  = [self.id],
        )
        self.children_ids.append(child.id)
        return child

    @classmethod
    def crossover(cls, idea_a: "Idea", idea_b: "Idea") -> "Idea":
        """Two ideas combine to form a synthesis — interdisciplinary breakthrough."""
        return cls(
            content     = f"[{idea_a.domain}×{idea_b.domain}] {idea_a.content[:30]}+{idea_b.content[:30]}",
            domain      = "synthesis",
            fitness     = (idea_a.fitness + idea_b.fitness) / 2 * 1.1,  # hybrid vigor
            novelty     = min(1.0, (idea_a.novelty + idea_b.novelty) / 2 + 0.15),
            utility     = (idea_a.utility + idea_b.utility) / 2,
            generation  = max(idea_a.generation, idea_b.generation) + 1,
            parent_ids  = [idea_a.id, idea_b.id],
        )

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "content":     self.content[:100],
            "domain":      self.domain,
            "fitness":     round(self.fitness, 3),
            "novelty":     round(self.novelty, 3),
            "utility":     round(self.utility, 3),
            "generation":  self.generation,
            "citations":   self.citations,
            "alive":       self.alive,
            "parent_ids":  self.parent_ids,
            "children_count": len(self.children_ids),
        }


class KnowledgeEcosystem:
    """
    The living library — a population of ideas that evolves continuously.

    Ideas compete for cognitive resources (agent attention).
    Highly cited ideas reproduce. Ignored ideas go extinct.
    Cross-domain synthesis produces breakthrough offspring with hybrid vigor.
    """

    DOMAINS = ["science", "engineering", "art", "governance", "mathematics",
               "philosophy", "strategy", "language", "systems", "synthesis"]

    def __init__(self, seed_ideas: int = 30):
        self.ideas: list[Idea] = []
        self.extinct: list[Idea] = []
        self.cycle: int = 0
        self.created_at = time.time()
        self._seed(seed_ideas)

    def _seed(self, n: int):
        domains = self.DOMAINS
        for i in range(n):
            self.ideas.append(Idea(
                content  = f"Seed concept #{i+1}",
                domain   = domains[i % len(domains)],
                fitness  = random.uniform(0.3, 0.8),
                novelty  = random.uniform(0.2, 0.9),
                utility  = random.uniform(0.2, 0.8),
            ))

    def tick(self) -> dict:
        """
        One knowledge evolution cycle:
        1. Decay all ideas (ideas not cited fade)
        2. Kill unfit ideas (extinction)
        3. Reproduce fit ideas (mutation + crossover)
        4. Return stats
        """
        self.cycle += 1
        alive_ideas = [i for i in self.ideas if i.alive]

        # 1. Age decay
        for idea in alive_ideas:
            idea.age_decay(rate=random.uniform(0.002, 0.01))

        # 2. Extinction
        just_died = [i for i in alive_ideas if not i.alive]
        self.extinct.extend(just_died)
        self.ideas = [i for i in self.ideas if i.alive]
        alive_ideas = self.ideas

        # 3. Reproduction — fittest ideas mutate
        new_ideas = []
        reproducers = sorted(alive_ideas, key=lambda i: i.fitness, reverse=True)

        for idea in reproducers[:max(1, len(reproducers) // 4)]:
            if len(self.ideas) + len(new_ideas) < 200:
                new_ideas.append(idea.mutate())

        # Cross-domain synthesis (occasional)
        if len(alive_ideas) >= 2 and random.random() < 0.3:
            a, b = random.sample(alive_ideas, 2)
            if a.domain != b.domain:
                synth = Idea.crossover(a, b)
                new_ideas.append(synth)

        self.ideas.extend(new_ideas)

        return {
            "cycle":         self.cycle,
            "alive":         len(self.ideas),
            "extinct":       len(self.extinct),
            "new_ideas":     len(new_ideas),
            "avg_fitness":   round(sum(i.fitness for i in self.ideas) / max(1, len(self.ideas)), 3),
            "avg_novelty":   round(sum(i.novelty for i in self.ideas) / max(1, len(self.ideas)), 3),
            "domains":       list({i.domain for i in self.ideas}),
        }

    def inject(self, content: str, domain: str = "general") -> Idea:
        """Add an externally generated idea to the ecosystem."""
        idea = Idea(
            content=content,
            domain=domain,
            fitness=0.6,
            novelty=0.7,
            utility=0.6,
        )
        self.ideas.append(idea)
        return idea

    def query(self, domain: str = None, top_k: int = 10) -> list[Idea]:
        """Retrieve highest-fitness ideas, optionally filtered by domain."""
        pool = [i for i in self.ideas if i.alive]
        if domain:
            pool = [i for i in pool if i.domain == domain]
        return sorted(pool, key=lambda i: i.fitness, reverse=True)[:top_k]

    def cite_idea(self, idea_id: str) -> bool:
        """Record a citation — strengthens the idea's fitness."""
        for idea in self.ideas:
            if idea.id == idea_id and idea.alive:
                idea.cite()
                return True
        return False

    def status(self) -> dict:
        alive = [i for i in self.ideas if i.alive]
        domain_counts = {}
        for i in alive:
            domain_counts[i.domain] = domain_counts.get(i.domain, 0) + 1

        return {
            "total_ideas":   len(alive),
            "extinct_count": len(self.extinct),
            "cycles":        self.cycle,
            "domain_map":    domain_counts,
            "avg_fitness":   round(sum(i.fitness for i in alive) / max(1, len(alive)), 3),
            "avg_novelty":   round(sum(i.novelty for i in alive) / max(1, len(alive)), 3),
            "top_ideas":     [i.to_dict() for i in self.query(top_k=5)],
            "most_cited":    [i.to_dict() for i in sorted(alive, key=lambda x: x.citations, reverse=True)[:5]],
        }