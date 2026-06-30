import json
import os
import glob
import numpy as np

# --- Load real nuPlan intersection data ---
with open("results/nuplan_extraction/traversing_intersection.json") as f:
    real_scenarios = json.load(f)

print(f"Loaded {len(real_scenarios)} real intersection scenarios\n")

real_speeds = []
real_agent_counts = []

for scenario in real_scenarios:
    traj = scenario["trajectory"]
    speeds = [np.sqrt(p["velocity_x"]**2 + p["velocity_y"]**2) for p in traj]
    agent_counts = [p["num_nearby_agents"] for p in traj]
    real_speeds.extend(speeds)
    real_agent_counts.extend(agent_counts)

print("=== REAL nuPlan intersection driving ===")
print(f"Speed (m/s): mean={np.mean(real_speeds):.2f}, std={np.std(real_speeds):.2f}, max={np.max(real_speeds):.2f}")
print(f"Nearby agents: mean={np.mean(real_agent_counts):.2f}, max={np.max(real_agent_counts)}")

# --- Load synthetic highway-env junction crossing evidence ---
evidence_files = glob.glob("results/evaluation/evidence/junction_crossing*.json")
print(f"\n\nFound {len(evidence_files)} synthetic junction_crossing evidence files (failure events)")

sim_ttcs = []
sim_hazard_distances = []
sim_action_deviation_count = 0
sim_categories = {}

for ef in evidence_files:
    with open(ef) as f:
        record = json.load(f)

    cat = record.get("category", "Unknown")
    sim_categories[cat] = sim_categories.get(cat, 0) + 1

    for step in record.get("evidence_trace", []):
        if step.get("ttc") is not None:
            sim_ttcs.append(step["ttc"])
        if step.get("hazard_distance") is not None:
            sim_hazard_distances.append(step["hazard_distance"])
        if step.get("action_deviation"):
            sim_action_deviation_count += 1

print("\n=== SYNTHETIC highway-env junction crossing (failure events only) ===")
if sim_ttcs:
    print(f"TTC at logged steps (s): mean={np.mean(sim_ttcs):.2f}, min={np.min(sim_ttcs):.2f}, max={np.max(sim_ttcs):.2f}")
if sim_hazard_distances:
    print(f"Hazard distance (m): mean={np.mean(sim_hazard_distances):.2f}, min={np.min(sim_hazard_distances):.2f}")
print(f"Steps with unsafe action deviation: {sim_action_deviation_count}")
print(f"\nFailure category breakdown: {sim_categories}")

print("\n\n=== COMPARISON NOTE ===")
print("Real nuPlan data reflects ALL driving (not just failures).")
print("Synthetic evidence files reflect ONLY failure events.")
print("Direct numeric comparison is limited - this is a structural/density")
print("comparison, not an apples-to-apples speed comparison.")
