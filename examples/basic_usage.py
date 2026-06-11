"""
Frontier Framework — Usage Examples
Run this to see the framework in action without a real LLM.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontier_framework import (
    FrontierAdapter, Population, CognitiveOrchestrator,
    CognitiveTask, ImmuneSystem, KnowledgeEcosystem
)


def separator(title=""):
    print(f"\n{'─'*60}")
    if title: print(f"  {title}")
    print('─'*60)


# ─────────────────────────────────────────────────────────────
# Example 1: Basic adapter usage (no LLM — genome-based reasoning)
# ─────────────────────────────────────────────────────────────
separator("Example 1: FrontierAdapter — Basic Thinking")

adapter = FrontierAdapter(population_size=20, auto_evolve=True)
print(repr(adapter))

result = adapter.think("What is the optimal path to AGI?", priority=0.7)
print(f"\nTask:        {result['agent']['task']}")
print(f"Agent:       {result['agent']['agent_id']} ({result['agent']['specialization']})")
print(f"Confidence:  {result['agent']['confidence']}")
print(f"Output:      {result['agent']['output']}")
print(f"Ecosystem:   {result['ecosystem']}")


# ─────────────────────────────────────────────────────────────
# Example 2: Cognitive Mitosis — hard problem exploration
# ─────────────────────────────────────────────────────────────
separator("Example 2: Cognitive Mitosis")

mitosis = adapter.cognitive_mitosis(
    hard_problem="How do we build a recursive self-improving architecture?",
    branches=8
)
print(f"Branches spawned:  {mitosis['branches_spawned']}")
print(f"Survivors:         {mitosis['survivors']}")
print(f"Best confidence:   {mitosis['best_confidence']:.3f}")
print(f"Best specialization: {mitosis['best_specialization']}")


# ─────────────────────────────────────────────────────────────
# Example 3: Full evolution cycles
# ─────────────────────────────────────────────────────────────
separator("Example 3: Evolutionary Cycles")

print("Running 10 evolution cycles...")
evo = adapter.evolve(cycles=10, tasks=["analyze", "plan", "verify", "synthesize"])
final = evo['final_population']
print(f"Alive agents:  {final.get('alive_count', '?')}")
print(f"Avg fitness:   {final.get('avg_fitness', '?')}")
print(f"Max gen:       {final.get('max_generation', '?')}")
print(f"Species count: {final.get('species_count', '?')}")


# ─────────────────────────────────────────────────────────────
# Example 4: Immune System scan
# ─────────────────────────────────────────────────────────────
separator("Example 4: Immune System")

scan = adapter.immune_scan()
print(f"Threats found:     {len(scan['threats'])}")
print(f"Actions taken:     {scan['response']['actions']}")
print(f"Total neutralized: {scan['response']['total_neutralized']}")


# ─────────────────────────────────────────────────────────────
# Example 5: Knowledge Ecosystem
# ─────────────────────────────────────────────────────────────
separator("Example 5: Knowledge Ecosystem")

eco = KnowledgeEcosystem(seed_ideas=20)

# Inject ideas
eco.inject("Transformer attention is a form of evolutionary selection", "science")
eco.inject("Compute efficiency mirrors biological energy optimization", "engineering")
eco.inject("Recursive self-reference is the basis of consciousness", "philosophy")

# Evolve 5 cycles
for _ in range(5):
    eco.tick()

status = eco.status()
print(f"Living ideas:  {status['total_ideas']}")
print(f"Extinct:       {status['extinct_count']}")
print(f"Avg fitness:   {status['avg_fitness']}")
print(f"Domains:       {list(status['domain_map'].keys())}")
print(f"\nTop 3 ideas:")
for idea in status['top_ideas'][:3]:
    print(f"  [{idea['domain']}] fitness={idea['fitness']} gen={idea['generation']} — {idea['content'][:60]}")


# ─────────────────────────────────────────────────────────────
# Example 6: Full pipeline (multi-stage cognitive routing)
# ─────────────────────────────────────────────────────────────
separator("Example 6: Cognitive Pipeline")

pop = Population(initial_size=30)
orch = CognitiveOrchestrator(pop)
stages = orch.run_pipeline("Design an AI that can improve its own architecture")

print(f"Pipeline stages completed: {len(stages)}")
for r in stages:
    print(f"  [{r.specialization:12}] conf={r.confidence:.3f}  {r.output[:60]}")


# ─────────────────────────────────────────────────────────────
# Example 7: With a real LLM (template — uncomment to use)
# ─────────────────────────────────────────────────────────────
separator("Example 7: Real LLM Integration (template)")

print("""
# Uncomment and fill in your LLM to activate full capability:

# --- Anthropic Claude ---
# import anthropic
# client = anthropic.Anthropic()
# def llm_fn(task: str, specialization: str) -> str:
#     return client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=1024,
#         system=f"You are a {specialization} specialist in an evolutionary AI ecosystem.",
#         messages=[{"role": "user", "content": task}]
#     ).content[0].text
#
# adapter = FrontierAdapter(llm_fn, population_size=30)
# result = adapter.think("Solve the alignment problem", priority=1.0, use_pipeline=True)
""")

separator("Complete")
print(f"\nFinal adapter state: {repr(adapter)}")
status = adapter.status()
print(f"Session tasks processed: {status['session']['tasks_processed']}")
print(f"Uptime: {status['session']['uptime_s']}s")
print(f"\nThe frontier model is not the destination — it is the first cell.\n")