import json
import os

from oc_scores import evaluation, collect_parameter_sweep_scores, plot_adaptivity_heatmap, plot_distinguishability_comparison

import os
import json
import glob
import pandas as pd
import numpy as np
import re

import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    eval_dir = "evaluation/scenario-1/"

    times = [500, 750, 1000, 2000, 4000] 
    umin_values = [0.0, 0.2, 0.4, 0.6]

    i = 0
    for t in times:
        for u_min in umin_values:
            filename = f"evaluations/{i}"

            evaluation(eval_dir, u_acc=0.8, u_target=0.9, t_recovery=t, u_min=u_min, c_norm=1, output_file=filename)
            i += 1


    collect_parameter_sweep_scores(os.path.join(eval_dir, "evaluations"))

    systems = ["Scenario1-ModelCycler", "Scenario1-PerfectSelector", "Scenario1-Robust", "Scenario1-NonAdaptive"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    
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

    plot_distinguishability_comparison(os.path.join(eval_dir, "evaluations/collected_scores.json"), metric='accumulated_adaptivity', systems=systems, name="all_systems")

    systems_subset = ["Scenario1-ModelCycler", "Scenario1-Robust", "Scenario1-NonAdaptive"]
    plot_distinguishability_comparison(os.path.join(eval_dir, "evaluations/collected_scores.json"), metric='accumulated_adaptivity', systems=systems_subset, name="without_perfect_selector")
