import json
import os
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
import numpy as np

OUTPUT_DIR = os.path.expanduser("~/rl-driving-agent-diagnosis/results/nuplan_extraction")
os.makedirs(OUTPUT_DIR, exist_ok=True)

builder = NuPlanScenarioBuilder(
    data_root=os.environ['NUPLAN_DATA_ROOT'] + '/nuplan-v1.1/splits/mini/data/cache/mini',
    map_root=os.environ['NUPLAN_MAPS_ROOT'],
    sensor_root=None,
    db_files=None,
    map_version='nuplan-maps-v1.0',
)

TARGETS = {
    "near_pedestrian_on_crosswalk": 1,
    "near_high_speed_vehicle": 6,
    "following_lane_with_slow_lead": 2,
    "near_construction_zone_sign": 1,
    "traversing_crosswalk": 2,
}

for scenario_type, expected_count in TARGETS.items():
    print(f"\nExtracting '{scenario_type}' (expected ~{expected_count})...")

    scenario_filter = ScenarioFilter(
        scenario_types=[scenario_type],
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=50,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=True,
        shuffle=False,
    )

    worker = Sequential()
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"  Found {len(scenarios)} scenarios")

    if not scenarios:
        print(f"  Skipping - none found")
        continue

    all_records = []
    for s in scenarios:
        n_iter = s.get_number_of_iterations()
        record = {
            "token": s.token,
            "log_name": s.log_name,
            "scenario_type": s.scenario_type,
            "duration_s": float(s.duration_s.time_s),
            "num_iterations": n_iter,
            "trajectory": []
        }
        for i in range(n_iter):
            ego = s.get_ego_state_at_iteration(i)
            tracked = s.get_tracked_objects_at_iteration(i)
            n_agents = len(tracked.tracked_objects) if tracked else 0
            speed = np.sqrt(
                ego.dynamic_car_state.rear_axle_velocity_2d.x**2 +
                ego.dynamic_car_state.rear_axle_velocity_2d.y**2
            )
            record["trajectory"].append({
                "iteration": i,
                "x": ego.rear_axle.x,
                "y": ego.rear_axle.y,
                "heading": ego.rear_axle.heading,
                "speed_ms": float(speed),
                "num_nearby_agents": n_agents,
            })
        all_records.append(record)

    output_path = os.path.join(OUTPUT_DIR, f"{scenario_type}.json")
    with open(output_path, "w") as f:
        json.dump(all_records, f, indent=2)

    speeds = [p["speed_ms"] for r in all_records for p in r["trajectory"]]
    agents = [p["num_nearby_agents"] for r in all_records for p in r["trajectory"]]
    print(f"  Saved {len(all_records)} scenarios to {output_path}")
    print(f"  Speed: mean={np.mean(speeds):.2f}, max={np.max(speeds):.2f} m/s")
    print(f"  Nearby agents: mean={np.mean(agents):.2f}, max={np.max(agents)}")

print("\nDone extracting all scenario types.")
