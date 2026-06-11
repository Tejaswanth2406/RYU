"""
Frontier Framework — FastAPI Application
REST API surface for your cognitive AI system to plug into.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from frontier_framework.evolution.population import Population
from frontier_framework.evolution.genome import AgentCell, Genome, Specialization

app = FastAPI(
    title="Frontier Framework",
    description="Evolutionary cognitive ecosystem — digital organism layer for your AI system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Ecosystem ────────────────────────────────────────────────────────
ecosystem = Population(initial_size=20, max_size=100)


# ─── Request Models ──────────────────────────────────────────────────────────

class TickRequest(BaseModel):
    tasks: Optional[list[str]] = None
    cycles: int = 1

class MitosisRequest(BaseModel):
    problem: str
    branches: int = 6

class SpawnRequest(BaseModel):
    specialization: Optional[str] = None
    count: int = 1

class EvolutionConfig(BaseModel):
    selection_pressure: Optional[float] = None
    max_size: Optional[int] = None
    mitosis_threshold: Optional[float] = None


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Frontier Framework",
        "status": "alive",
        "philosophy": "The frontier model is not a machine — it is the seed of an infinite evolutionary cascade.",
        "endpoints": [
            "GET  /status          — ecosystem snapshot",
            "POST /tick            — run N evolutionary cycles",
            "POST /mitosis         — cognitive mitosis on a problem",
            "POST /spawn           — inject new agents",
            "GET  /agents          — list all living agents",
            "GET  /agents/{id}     — inspect one agent",
            "GET  /species         — species map",
            "GET  /graveyard       — deceased agents",
            "POST /config          — tune evolution parameters",
            "DELETE /reset         — restart ecosystem",
        ]
    }


@app.get("/status")
def status():
    """Full ecosystem snapshot — population health, species, history."""
    return ecosystem.status()


@app.post("/tick")
def tick(req: TickRequest):
    """
    Run N evolutionary cycles.
    Each cycle: agents think → fitness updates → selection → reproduction.
    """
    if req.cycles < 1 or req.cycles > 50:
        raise HTTPException(400, "cycles must be 1–50")

    results = []
    for _ in range(req.cycles):
        stats = ecosystem.tick(tasks=req.tasks)
        results.append(stats)

    return {
        "cycles_run": req.cycles,
        "final_state": results[-1],
        "history": results,
    }


@app.post("/mitosis")
def mitosis(req: MitosisRequest):
    """
    Cognitive Mitosis (Stage 7).
    Hard problem → N explorer agents → survive/die → harvest best insights.
    Thinking becomes evolution.
    """
    if req.branches < 2 or req.branches > 50:
        raise HTTPException(400, "branches must be 2–50")

    return ecosystem.cognitive_mitosis(req.problem, req.branches)


@app.post("/spawn")
def spawn(req: SpawnRequest):
    """Inject new agents into the ecosystem (seed diversity)."""
    if req.count < 1 or req.count > 20:
        raise HTTPException(400, "count must be 1–20")

    specs = [s.value for s in Specialization]
    if req.specialization and req.specialization not in specs:
        raise HTTPException(400, f"invalid specialization. choose from {specs}")

    new_agents = []
    for _ in range(req.count):
        genome = Genome(
            reasoning_style  = __import__('random').random(),
            curiosity_level  = __import__('random').random(),
            memory_retention = __import__('random').random(),
            risk_tolerance   = __import__('random').random(),
            specialization   = req.specialization or __import__('random').choice(specs),
            mutation_rate    = __import__('random').uniform(0.02, 0.12),
            cooperation_bias = __import__('random').random(),
        )
        agent = AgentCell(genome=genome, energy=0.6)
        ecosystem.agents.append(agent)
        new_agents.append(agent.to_dict())

    return {"spawned": req.count, "agents": new_agents}


@app.get("/agents")
def list_agents(limit: int = 50, sort_by: str = "fitness"):
    """List living agents sorted by fitness, energy, or age."""
    agents = [a for a in ecosystem.agents if a.alive]

    sort_key = {
        "fitness":    lambda a: a.genome.fitness_score,
        "energy":     lambda a: a.energy,
        "age":        lambda a: a.age,
        "generation": lambda a: a.genome.generation,
    }.get(sort_by, lambda a: a.genome.fitness_score)

    agents.sort(key=sort_key, reverse=True)
    return {
        "total": len(agents),
        "agents": [a.to_dict() for a in agents[:limit]],
    }


@app.get("/agents/{agent_id}")
def get_agent(agent_id: str):
    """Inspect a specific agent by ID."""
    agent = next((a for a in ecosystem.agents if a.id == agent_id), None)
    if not agent:
        raise HTTPException(404, f"agent '{agent_id}' not found")
    return agent.to_dict()


@app.get("/species")
def species_map():
    """Species distribution across cognitive niches."""
    return {
        "species": ecosystem.get_species_map(),
        "total_living": len([a for a in ecosystem.agents if a.alive]),
    }


@app.get("/graveyard")
def graveyard(limit: int = 20):
    """Deceased agents — memory of what died and why."""
    recent = ecosystem.graveyard[-limit:]
    recent.reverse()
    return {
        "total_deaths": len(ecosystem.graveyard),
        "recent": recent,
    }


@app.get("/events")
def events(limit: int = 30):
    """Recent ecosystem events."""
    recent = ecosystem.event_log[-limit:]
    recent.reverse()
    return {"events": recent}


@app.post("/config")
def configure(cfg: EvolutionConfig):
    """Tune evolution parameters at runtime."""
    changed = {}
    if cfg.selection_pressure is not None:
        if not 0.0 <= cfg.selection_pressure <= 1.0:
            raise HTTPException(400, "selection_pressure must be 0.0–1.0")
        ecosystem.selection_pressure = cfg.selection_pressure
        changed["selection_pressure"] = cfg.selection_pressure
    if cfg.max_size is not None:
        if not 10 <= cfg.max_size <= 500:
            raise HTTPException(400, "max_size must be 10–500")
        ecosystem.max_size = cfg.max_size
        changed["max_size"] = cfg.max_size
    if cfg.mitosis_threshold is not None:
        ecosystem.mitosis_threshold = cfg.mitosis_threshold
        changed["mitosis_threshold"] = cfg.mitosis_threshold

    return {"updated": changed, "current": {
        "selection_pressure": ecosystem.selection_pressure,
        "max_size": ecosystem.max_size,
        "mitosis_threshold": ecosystem.mitosis_threshold,
    }}


@app.delete("/reset")
def reset(initial_size: int = 20):
    """Restart the ecosystem from scratch — intentional death for adaptability."""
    global ecosystem
    ecosystem = Population(initial_size=initial_size, max_size=100)
    return {"status": "reset", "initial_size": initial_size, "message": "Death prevents stagnation."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)