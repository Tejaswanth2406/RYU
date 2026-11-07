<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/11b062cc-9d77-4a7a-8eb9-7000105acfda" />

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

### First measurable forecast experiment

The initial forecast target is the civilization's average fitness **one tick ahead**. Each successful
`Civilization.tick()` records the pre-tick state and the actual post-tick average fitness in a
`FitnessExperiment`. This observation collector is separate from the ML simulation ensemble, so
synthetic seed labels are not treated as observed outcomes.

To collect multiple independent runs, share one collector across civilizations, then reserve entire
run IDs for evaluation:

```python
from frontier_framework import Civilization, FitnessExperiment

experiment = FitnessExperiment()
for _ in range(3):
    civilization = Civilization(fitness_experiment=experiment)
    for _ in range(20):
        civilization.tick()

held_out_run = experiment.observations[-1].run_id
report = experiment.evaluate_persistence_baseline([held_out_run])
print(report["mae"])
```

The baseline predicts that next-tick average fitness stays equal to current average fitness. The
report computes MAE only for explicitly held-out complete runs, and requires at least one other
run to remain separate. This first step records and evaluates the baseline; it does not train a
forecast model yet.

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

### Neural Cell Brain (C++20)

The native project also contains a modular 19-file neural-cell brain architecture under
`frontier_framework/native/neural_brain/`. It connects configurable neural cells, weighted graph
message passing, feature encoding, three-tier memory, predictive models, and an adaptive learning
controller.

The CPU core implements decision trees, bagged Random Forests, residual Gradient Boosting, and a
small MLP. XGBoost is an optional callback adapter for an externally supplied backend; it is not
reimplemented or required to build the core. Graph aggregation supports weighted-mean and
similarity-attention modes. The learning controller trains on supplied examples and adapts expert weights from observed
prediction error, inference latency, and ensemble disagreement; it does not fabricate a training
dataset.

The 19 files are:

- Headers: `types.hpp`, `config.hpp`, `neural_cell.hpp`, `brain_graph.hpp`, `gnn_layer.hpp`,
  `memory_layer.hpp`, `feature_encoder.hpp`, `tree_models.hpp`, `ensemble.hpp`, and
  `learning_controller.hpp` in `frontier_framework/native/neural_brain/include/neural_brain/`.
- Implementations: the corresponding nine `.cpp` files (all headers except `config.hpp`) in
  `frontier_framework/native/neural_brain/src/`.

The native targets and integration test use the existing CMake project:

```powershell
cmake -S frontier_framework/native -B build/native
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

For C++ consumers, add `frontier_framework/native/neural_brain/include` to include paths and
link the `frontier_neural_brain` CMake target. The central entry point is
`neural_brain/learning_controller.hpp`; a minimal training/inference flow is:

```cpp
neural_brain::LearningController brain;
brain.add_cell(1, "planner");
brain.add_cell(2, "researcher");
brain.connect(1, 2, 0.7);
for (int i = 0; i < 4; ++i) {
    brain.learn({0.2 * i, 0.8 - 0.2 * i}, 0.1 * i);
}
auto result = brain.infer({0.3, 0.7});
```

An external XGBoost binding can be registered through `brain.ensemble().add_model(...)` using an
`XGBoostAdapter` with fit/predict callbacks.

The GNN currently performs deterministic CPU message passing rather than gradient-trained
convolution; the attention mode is a similarity-weighted aggregation. Classification is exposed as
a binary decision over the regression ensemble output. Multiclass classification, trainable GNN
weights, persistent on-disk memory, GPU acceleration, and automatic XGBoost discovery are not
implemented yet.

---

## Philosophy

> *"Death prevents stagnation. The civilization stays adaptive."*
> *"Thinking itself becomes evolution."*
> *"The intelligence is not a model — it is the continuously evolving network."*

The frontier model is not a smarter chatbot. It is the seed of an infinite evolutionary cascade.
Each agent is temporary. The lineage is what persists.
