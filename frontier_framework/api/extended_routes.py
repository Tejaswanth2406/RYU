"""
Frontier Framework — Extended API Routes
Immune system + Knowledge ecosystem endpoints.
Mount this into main.py with: app.include_router(extended_router)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontier_framework.cognition.immune import ImmuneSystem
from frontier_framework.cognition.knowledge import KnowledgeEcosystem
from frontier_framework.orchestration.orchestrator import CognitiveOrchestrator, CognitiveTask

extended_router = APIRouter()

# ─── Shared State (injected from main) ───────────────────────────────────────
immune_system = ImmuneSystem()
knowledge_eco = KnowledgeEcosystem(seed_ideas=30)
orchestrator: Optional[CognitiveOrchestrator] = None   # set by main after ecosystem init


# ─── Immune System Routes ─────────────────────────────────────────────────────

@extended_router.get("/immune/report")
def immune_report():
    """Immune system health report — threats, neutralizations, quarantine."""
    return immune_system.report()


@extended_router.post("/immune/scan")
def immune_scan():
    """
    Trigger an immune scan of the active population.
    Detects: energy parasites, fitness fraud, stagnation, mutation storms, monoculture.
    """
    from frontier_framework.api.main import ecosystem
    threats = immune_system.scan(ecosystem)
    response = immune_system.respond(ecosystem, threats)
    return {
        "threats_found": len(threats),
        "new_threats":   [t.to_dict() for t in threats],
        "response":      response,
    }


# ─── Knowledge Ecosystem Routes ──────────────────────────────────────────────

@extended_router.get("/knowledge/status")
def knowledge_status():
    """Knowledge ecosystem snapshot — living ideas, extinct, domain map."""
    return knowledge_eco.status()


@extended_router.post("/knowledge/tick")
def knowledge_tick(cycles: int = 1):
    """Evolve the knowledge ecosystem N cycles."""
    if cycles < 1 or cycles > 20:
        raise HTTPException(400, "cycles must be 1–20")
    results = [knowledge_eco.tick() for _ in range(cycles)]
    return {"cycles": cycles, "final": results[-1], "history": results}


class IdeaRequest(BaseModel):
    content: str
    domain: str = "general"

@extended_router.post("/knowledge/inject")
def inject_idea(req: IdeaRequest):
    """Inject an externally generated idea into the knowledge ecosystem."""
    if req.domain not in knowledge_eco.DOMAINS + ["general"]:
        raise HTTPException(400, f"unknown domain. valid: {knowledge_eco.DOMAINS}")
    idea = knowledge_eco.inject(req.content, req.domain)
    return {"injected": idea.to_dict()}


@extended_router.get("/knowledge/query")
def query_ideas(domain: Optional[str] = None, top_k: int = 10):
    """Query top-fitness ideas, optionally filtered by domain."""
    ideas = knowledge_eco.query(domain=domain, top_k=top_k)
    return {"domain": domain, "ideas": [i.to_dict() for i in ideas]}


@extended_router.post("/knowledge/cite/{idea_id}")
def cite_idea(idea_id: str):
    """Cite an idea — increases its fitness and reproductive success."""
    success = knowledge_eco.cite_idea(idea_id)
    if not success:
        raise HTTPException(404, f"idea '{idea_id}' not found or extinct")
    return {"cited": idea_id, "status": "fitness_increased"}


# ─── Orchestrator Routes ──────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    content: str
    specialization: Optional[str] = None
    priority: float = 0.5
    use_mitosis: bool = False
    mitosis_branches: int = 5

@extended_router.post("/orchestrate/task")
def orchestrate_task(req: TaskRequest):
    """Route a cognitive task through the best-fit agent."""
    from frontier_framework.api.main import ecosystem
    orch = CognitiveOrchestrator(ecosystem)
    task = CognitiveTask(
        content=req.content,
        required_specialization=req.specialization,
        priority=req.priority,
        use_mitosis=req.use_mitosis,
        mitosis_branches=req.mitosis_branches,
    )
    result = orch.process(task)
    return result.to_dict()


@extended_router.post("/orchestrate/pipeline")
def orchestrate_pipeline(req: TaskRequest):
    """
    Run a full cognitive pipeline:
    planner → researcher → creativity → synthesizer → verification
    """
    from frontier_framework.api.main import ecosystem
    orch = CognitiveOrchestrator(ecosystem)
    results = orch.run_pipeline(req.content)
    return {
        "pipeline_stages": len(results),
        "results": [r.to_dict() for r in results],
        "final_output": results[-1].output if results else None,
    }


@extended_router.get("/orchestrate/metrics")
def orchestrate_metrics():
    """Orchestrator performance metrics."""
    from frontier_framework.api.main import ecosystem
    orch = CognitiveOrchestrator(ecosystem)
    return orch.metrics()