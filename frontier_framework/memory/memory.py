"""
Frontier Framework — Distributed Memory (Stage 6: Collective Consciousness)
Shared cognitive memory layer across all agents in the ecosystem.
Agents read, write, and compete for memory slots.
Memory decays unless reinforced. High-relevance memories persist.
Analogous to the common communication layer in a planetary AI network.
"""

from __future__ import annotations
import time
import uuid
import math
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class MemoryTrace:
    """
    A single memory unit — content + metadata + decay dynamics.
    Like a synaptic connection: strengthened by use, weakened by time.
    """
    id: str                 = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str            = ""
    tags: list              = field(default_factory=list)
    strength: float         = 1.0           # 0=forgotten, 1=vivid
    access_count: int       = 0
    created_at: float       = field(default_factory=time.time)
    last_accessed: float    = field(default_factory=time.time)
    author_id: str          = "system"
    memory_type: str        = "episodic"    # episodic | semantic | procedural | working
    importance: float       = 0.5
    alive: bool             = True

    # Decay constants
    BASE_DECAY: float       = 0.01          # per cycle
    ACCESS_BOOST: float     = 0.15          # each access strengthens

    def access(self) -> str:
        """Retrieve this memory — strengthens it."""
        self.access_count += 1
        self.last_accessed = time.time()
        self.strength = min(1.0, self.strength + self.ACCESS_BOOST)
        return self.content

    def decay(self, cycles: int = 1):
        """Time weakens unused memories."""
        time_factor = math.exp(-self.BASE_DECAY * cycles * (1 - self.importance))
        self.strength *= time_factor
        if self.strength < 0.05:
            self.alive = False

    def reinforce(self, amount: float = 0.2):
        """External reinforcement — another agent cites this memory."""
        self.strength = min(1.0, self.strength + amount)
        self.importance = min(1.0, self.importance + 0.05)

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "content":      self.content[:120],
            "tags":         self.tags,
            "strength":     round(self.strength, 3),
            "access_count": self.access_count,
            "memory_type":  self.memory_type,
            "importance":   round(self.importance, 3),
            "author_id":    self.author_id,
            "alive":        self.alive,
            "age_s":        round(time.time() - self.created_at, 1),
        }


class DistributedMemory:
    """
    Collective memory shared across all agents.
    Stage 6 analog: the common communication layer between AI organisms.

    Architecture:
    - Working memory:    small, fast, high-decay (active context)
    - Episodic memory:   experiences, task outcomes
    - Semantic memory:   facts, knowledge, domain expertise
    - Procedural memory: learned patterns, strategies

    Selection pressure applies here too:
    - Frequently accessed memories get stronger (long-term potentiation)
    - Unused memories decay and are pruned (forgetting as optimization)
    - Important memories are preserved regardless of access frequency
    """

    MEMORY_TYPES = ["working", "episodic", "semantic", "procedural"]

    def __init__(self, max_entries: int = 1000, decay_interval: int = 5):
        self.traces: dict[str, MemoryTrace] = {}
        self.max_entries = max_entries
        self.decay_interval = decay_interval
        self.cycle = 0
        self.forgotten: list[dict] = []
        self._type_index: dict[str, list] = defaultdict(list)
        self._tag_index: dict[str, list] = defaultdict(list)
        self.stats_history: list[dict] = []

    # ── Write ─────────────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        memory_type: str = "episodic",
        tags: list = None,
        author_id: str = "system",
        importance: float = 0.5,
    ) -> MemoryTrace:
        """Store a new memory. If at capacity, evict weakest trace."""
        if len(self.traces) >= self.max_entries:
            self._evict_weakest()

        trace = MemoryTrace(
            content=content,
            memory_type=memory_type,
            tags=tags or [],
            author_id=author_id,
            importance=importance,
        )
        self.traces[trace.id] = trace
        self._type_index[memory_type].append(trace.id)
        for tag in (tags or []):
            self._tag_index[tag].append(trace.id)

        return trace

    def remember_batch(self, items: list[dict]) -> list[MemoryTrace]:
        """Store multiple memories efficiently."""
        return [self.remember(**item) for item in items]

    # ── Read ──────────────────────────────────────────────────────────────────

    def recall(self, trace_id: str) -> Optional[str]:
        """Retrieve a specific memory by ID."""
        trace = self.traces.get(trace_id)
        if trace and trace.alive:
            return trace.access()
        return None

    def search(
        self,
        query: str = "",
        memory_type: str = None,
        tags: list = None,
        top_k: int = 10,
        min_strength: float = 0.1,
    ) -> list[MemoryTrace]:
        """
        Search memories by content substring, type, or tags.
        Returns strongest matching traces.
        """
        pool = [t for t in self.traces.values() if t.alive and t.strength >= min_strength]

        if memory_type:
            pool = [t for t in pool if t.memory_type == memory_type]

        if tags:
            pool = [t for t in pool if any(tag in t.tags for tag in tags)]

        if query:
            q = query.lower()
            pool = [t for t in pool if q in t.content.lower()]

        return sorted(pool, key=lambda t: t.strength * t.importance, reverse=True)[:top_k]

    def recall_recent(self, n: int = 10, memory_type: str = None) -> list[MemoryTrace]:
        """Most recently created/accessed memories."""
        pool = [t for t in self.traces.values() if t.alive]
        if memory_type:
            pool = [t for t in pool if t.memory_type == memory_type]
        return sorted(pool, key=lambda t: t.last_accessed, reverse=True)[:n]

    def recall_strongest(self, n: int = 10) -> list[MemoryTrace]:
        """Top memories by strength × importance."""
        pool = [t for t in self.traces.values() if t.alive]
        return sorted(pool, key=lambda t: t.strength * t.importance, reverse=True)[:n]

    # ── Reinforce / Forget ────────────────────────────────────────────────────

    def reinforce(self, trace_id: str, amount: float = 0.2) -> bool:
        """Strengthen a memory — called when agents reference it."""
        trace = self.traces.get(trace_id)
        if trace and trace.alive:
            trace.reinforce(amount)
            return True
        return False

    def forget(self, trace_id: str) -> bool:
        """Intentionally delete a memory."""
        trace = self.traces.pop(trace_id, None)
        if trace:
            self.forgotten.append(trace.to_dict())
            return True
        return False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def tick(self) -> dict:
        """
        One memory cycle: decay all traces, prune dead ones.
        Memory evolution — unused memories die, important ones persist.
        """
        self.cycle += 1
        alive_before = len([t for t in self.traces.values() if t.alive])

        for trace in list(self.traces.values()):
            if trace.alive:
                trace.decay(cycles=1)

        # Prune dead traces
        dead = [tid for tid, t in self.traces.items() if not t.alive]
        for tid in dead:
            trace = self.traces.pop(tid)
            self.forgotten.append(trace.to_dict())
            for tag in trace.tags:
                self._tag_index[tag] = [x for x in self._tag_index[tag] if x != tid]

        alive_after = len(self.traces)
        stats = {
            "cycle":        self.cycle,
            "alive":        alive_after,
            "forgotten":    len(dead),
            "total_forgotten": len(self.forgotten),
            "avg_strength": round(
                sum(t.strength for t in self.traces.values()) / max(1, alive_after), 3
            ),
        }
        self.stats_history.append(stats)
        return stats

    def _evict_weakest(self):
        """Remove weakest memory to make room."""
        if not self.traces:
            return
        weakest_id = min(
            self.traces,
            key=lambda tid: self.traces[tid].strength * self.traces[tid].importance
        )
        self.forget(weakest_id)

    # ── Agent Integration ─────────────────────────────────────────────────────

    def agent_write(self, agent_id: str, content: str, tags: list = None, importance: float = 0.5) -> MemoryTrace:
        """Agent deposits a memory into the collective."""
        return self.remember(
            content=content,
            memory_type="episodic",
            tags=(tags or []) + [f"agent:{agent_id}"],
            author_id=agent_id,
            importance=importance,
        )

    def agent_read(self, agent_id: str, query: str, top_k: int = 5) -> list[dict]:
        """Agent retrieves relevant memories from the collective."""
        results = self.search(query=query, top_k=top_k)
        for t in results:
            t.access()
        return [t.to_dict() for t in results]

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        alive = [t for t in self.traces.values() if t.alive]
        type_counts = defaultdict(int)
        for t in alive:
            type_counts[t.memory_type] += 1

        return {
            "total_alive":     len(alive),
            "total_forgotten": len(self.forgotten),
            "cycle":           self.cycle,
            "capacity":        f"{len(alive)}/{self.max_entries}",
            "type_breakdown":  dict(type_counts),
            "avg_strength":    round(sum(t.strength for t in alive) / max(1, len(alive)), 3),
            "avg_importance":  round(sum(t.importance for t in alive) / max(1, len(alive)), 3),
            "top_memories":    [t.to_dict() for t in self.recall_strongest(5)],
            "recent_memories": [t.to_dict() for t in self.recall_recent(5)],
        }