# Frontier Framework

> *"The frontier model is not the destination — it's the first cell."*

An evolutionary cognitive ecosystem for Python AI systems. The framework wraps your existing AI with a living population of agent cells that compete, evolve, reproduce, and die — continuously selecting for better cognitive patterns.

---

## Architecture

```
frontier_framework/
├── core/
│   ├── genome.py          Genome + AgentCell — heritable traits, energy model
│   ├── population.py      Population ecosystem — natural selection, reproduction
│   ├── orchestrator.py    Cognitive Orchestrator — task routing + pipeline
│   ├── immune.py          Immune System — threat detection + response
│   └── knowledge.py       Knowledge Ecosystem — living ideas, meme evolution
├── agents/
│   └── integration.py     FrontierAdapter — one-class integration layer
├── api/
│   ├── main.py            FastAPI application (core endpoints)
│   └── extended_routes.py Immune + Knowledge + Orchestrator endpoints
└── ui/
    └── dashboard.html     Real-time ecosystem visualization dashboard
```

---

## Core Concepts

### Stage 1 — Agent Cells
Each agent carries a genome of cognitive traits:
- `reasoning_style` — deductive ↔ abductive
- `curiosity_level` — exploit ↔ explore
- `memory_retention` — context window analog
- `risk_tolerance` — commitment to uncertain paths
- `cooperation_bias` — compete vs merge
- `mutation_rate` — evolutionary plasticity
- `specialization` — cognitive niche (planner / researcher / creativity / etc.)

### Stage 3 — Compute as Energy
Every agent thought costs energy (FLOPs analog).
Good reasoning earns energy back. Bad agents starve. Good agents reproduce.

### Stage 7 — Cognitive Mitosis
Hard problems spawn N explorer branches. Each explores independently.
Survivors recombine. Thinking becomes accelerated evolution.

### Stage 16 — Immune System
Defense intelligences patrol for: energy parasites, fitness fraud,
stagnation, mutation storms, monoculture collapse.

### Stage Ω-2 — Knowledge Ecosystem
Ideas evolve like species. Highly cited ideas reproduce.
Ignored ideas go extinct. Cross-domain synthesis creates hybrids with hybrid vigor.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Wrap your AI in 3 lines
```python
from frontier_framework import FrontierAdapter

# Your existing reasoning function
def my_llm(task: str, specialization: str) -> str:
    # call your LLM / chain / agent here
    return your_model.generate(task)

# Wrap it
adapter = FrontierAdapter(my_llm)

# Use it
result = adapter.think("Solve recursive self-improvement", priority=0.9)
print(result["output"])
print(result["ecosystem"])   # live telemetry
```

### 3. Explicit cognitive mitosis for hard problems
```python
result = adapter.cognitive_mitosis(
    "How do we solve the alignment problem?",
    branches=10
)
# best_confidence, survivors, specialization
```

### 4. Full multi-stage pipeline
```python
result = adapter.think(
    "Design a self-improving architecture",
    use_pipeline=True
)
# Routes: planner → researcher → creativity → synthesizer → verification
```

### 5. Run the API server
```bash
cd frontier_framework
uvicorn api.main:app --reload --port 8000
```

### 6. Open the dashboard
Open `ui/dashboard.html` in a browser. Works in demo mode without the API.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/status` | Full ecosystem snapshot |
| POST | `/tick` | Run N evolutionary cycles |
| POST | `/mitosis` | Cognitive mitosis on a problem |
| POST | `/spawn` | Inject new agents |
| GET | `/agents` | List living agents |
| GET | `/agents/{id}` | Inspect one agent |
| GET | `/species` | Species distribution map |
| GET | `/graveyard` | Deceased agents |
| POST | `/config` | Tune selection pressure, max_size |
| DELETE | `/reset` | Restart ecosystem |
| POST | `/immune/scan` | Trigger immune scan |
| GET | `/immune/report` | Immune system health |
| GET | `/knowledge/status` | Knowledge ecosystem snapshot |
| POST | `/knowledge/tick` | Evolve knowledge N cycles |
| POST | `/knowledge/inject` | Add a new idea |
| GET | `/knowledge/query` | Query top ideas by domain |
| POST | `/orchestrate/task` | Route a task through best agent |
| POST | `/orchestrate/pipeline` | Full cognitive pipeline |

---

## Specializations (Species)

| Specialization | Role |
|---|---|
| `planner` | Goal decomposition, strategy |
| `researcher` | Information gathering, analysis |
| `creativity` | Novel ideation, lateral thinking |
| `verification` | Fact-checking, consistency |
| `defense` | Security, adversarial robustness |
| `memory` | Context management, retrieval |
| `engineer` | Implementation, optimization |
| `synthesizer` | Cross-domain integration |

---

## Integration with Existing AI Systems

```python
# OpenAI
import openai
def llm(task, spec):
    return openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": f"You are a {spec} specialist."},
                  {"role": "user", "content": task}]
    ).choices[0].message.content

adapter = FrontierAdapter(llm)

# Anthropic Claude
import anthropic
client = anthropic.Anthropic()
def llm(task, spec):
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role":"user","content":f"[{spec}] {task}"}]
    ).content[0].text

adapter = FrontierAdapter(llm)

# LangChain
from langchain_openai import ChatOpenAI
chain = ChatOpenAI()
adapter = FrontierAdapter(lambda task, spec: chain.invoke(task).content)
```

---

## Biological Lifecycle Integration

The frontier framework can now surface a biological lifecycle layer that maps life-like mechanisms
onto the computational architecture:

```python
from frontier_framework import Civilization, BiologicalLifecycle

civilization = Civilization(n_nodes=4, node_population_size=12)
print(civilization.biological_profile()["lifecycle_stages"]["reproduction"])
print(BiologicalLifecycle().get("immune_response").computational_application)
```

This connects concepts like cell formation, metabolism, reproduction, homeostasis, and immune
response to the existing agent/population/civilization model without replacing the core AI
architecture.

The native simulation engine provides a validated C API and a C++ wrapper. It evolves agent count,
fitness, risk, memory, emergence, and uptime over simulation steps, and reports a bounded environment
score. It has no third-party runtime dependencies.

Build and run its native tests with CMake and a C/C++ compiler:

```powershell
cmake -S frontier_framework/native -B build/native
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

The C entry points are declared in `frontier_framework/native/simulated_environment.h`. C++ users
can include `frontier_framework/native/simulated_environment.hpp` and use
`frontier::SimulatedEnvironment`. The native engine is the deterministic simulation core; the
Python ML models can consume its resulting state snapshots for forecasting and model comparison.

---

## Philosophy

> *"Death prevents stagnation. The civilization stays adaptive."*
> *"Thinking itself becomes evolution."*
> *"The intelligence is not a model — it is the continuously evolving network."*

The frontier model is not a smarter chatbot. It is the seed of an infinite evolutionary cascade.
Each agent is temporary. The lineage is what persists.