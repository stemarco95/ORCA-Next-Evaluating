import json
import os

from oc_scores import evaluation, collect_parameter_sweep_scores, plot_adaptivity_heatmap, plot_distinguishability_comparison
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    eval_dir = "evaluation/scenario-3/"

    times = [500, 1000, 2000, 5000, 7000, 10000] 
    umin_values = [0.0, 0.2, 0.4, 0.6]

    i = 0
    for t in times:
        for u_min in umin_values:
            filename = f"evaluations/{i}"

            evaluation(eval_dir, u_acc=0.8, u_target=0.9, t_recovery=t, u_min=u_min, c_norm=1, output_file=filename, add_robust_system="Scenario3-TheoreticalRobust")
            i += 1


    collect_parameter_sweep_scores(os.path.join(eval_dir, "evaluations"))

    systems = ["Scenario3-AdaptiveLearner", "Scenario3-OptimisingLearner", "Scenario3-NonAdaptive"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    
    # Flatten the 2x2 axes array so we can easily iterate over it alongside the systems
    for ax, system in zip(axes.flat, systems):
        plot_adaptivity_heatmap(os.path.join(eval_dir, "evaluations/collected_scores.json"), system, ax=ax)

    plt.subplots_adjust(wspace=0.35)   
    plt.tight_layout()  

    # Save the combined figure
    output_dir = os.path.join(eval_dir, "heatmaps")
    os.makedirs(output_dir, exist_ok=True)
    
    # Saving as PDF for high-quality, space-efficient document inclusion
    output_path = os.path.join(output_dir, "combined_adaptivity_heatmaps.pdf")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Combined heatmaps saved to: {output_path}")

    systems = ["Scenario3-TheoreticalRobust", "Scenario3-AdaptiveLearner", "Scenario3-OptimisingLearner", "Scenario3-NonAdaptive"]
    plot_distinguishability_comparison(os.path.join(eval_dir, "evaluations/collected_scores.json"), metric='accumulated_adaptivity', systems=systems, name="all_systems")

    file = os.path.join(eval_dir, "evaluations/9.json")
    with open(file, "r") as f:
        data = json.load(f)

        print("Survival-Focus System:")
        print("Average Survival Phase Duration:")
        d = np.array(data["adaptivity"]["log_Scenario3-AdaptiveLearner_phases.json"]["phase_durations"])
        print(d.mean())

        print("Optimization-Focus System:")
        print("Average Survival Phase Duration:")
        d = np.array(data["adaptivity"]["log_Scenario3-OptimisingLearner_phases.json"]["phase_durations"])
        print(d.mean())


