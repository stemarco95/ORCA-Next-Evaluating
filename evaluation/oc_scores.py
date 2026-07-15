"""
Evaluation and Visualization Module for Modular Reinforcement Learning Systems.
This script provides utilities to calculate Organic Computing (OC) metrics 
(Adaptivity, Self-Optimization), process experimental phase logs, and generate 
heatmaps and stacked utility traces for the thesis scenarios.
"""

import json
import os
from typing import List
import glob
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from orca_next.monitor.oc_phases import Phase, SurvivalPhase, OptimizationPhase, TargetPhase, from_dict, OcPhases


def adaptivity(survival_phases: list[SurvivalPhase], u_acc: float, u_delta: float, u_min: float, t_max: float, c_norm: float, min_duration: float = 0):
    """
    Calculates the accumulated adaptivity score, effort, and efficiency for a list of survival phases.
    """
    scores = []
    efforts = []
    weights = []

    for phase in survival_phases:
        # Ignore phases shorter than the minimum duration threshold
        if phase.duration < min_duration:
            continue
            
        score = phase._calculate_adaptivity_score(t_max, u_delta, u_min)
        scores.append(score)

        effort = phase._calculate_self_configuration_score(c_norm, t_max)
        efforts.append(effort)

        weights.append(phase._calculate_adaptivity_weight(t_max))

    # Calculate weighted averages
    accumulated_score = np.average(scores, weights=weights) if scores else 1.0
    accumulated_effort = np.average(efforts, weights=weights) if efforts else 0.0
    
    # Efficiency is the ratio of adaptivity to configuration effort (smoothed to prevent div by zero)
    efficiency = accumulated_score / (accumulated_effort + 1)

    # Calculate statistical properties
    score_mean = np.mean(scores) if scores else 1.0
    score_std = np.std(scores) if scores else 0.0
    
    valid_durations = [phase.duration for phase in survival_phases if phase.duration >= min_duration]
    duration_mean = np.mean(valid_durations) if scores else 0.0
    duration_std = np.std(valid_durations) if scores else 0.0

    efficiency_mean = np.mean([score / (effort + 1) for score, effort in zip(scores, efforts)]) if scores else 0.0
    efficiency_std = np.std([score / (effort + 1) for score, effort in zip(scores, efforts)]) if scores else 0.0

    return {
        "accumulated_adaptivity": accumulated_score,
        "accumulated_self_configuration": accumulated_effort,
        "adaptivity_efficiency": efficiency,
        "valid_phases": len(scores),
        "t_max": t_max,
        "u_acc": u_acc,
        "u_delta": u_delta,
        "u_min": u_min,
        "adaptivity_score_mean": score_mean,
        "adaptivity_score_std": score_std,
        "adaptivity_duration_mean": duration_mean,
        "adaptivity_duration_std": duration_std,
        "adaptivity_efficiency_mean": efficiency_mean,
        "adaptivity_efficiency_std": efficiency_std,
        "phase_adaptivity": scores,
        "phase_self_configuration": efforts,
        "phase_durations": valid_durations
    }

def self_optimization(phases: list, u_acc: float, u_target: float, c_norm: float):
    """
    Calculates the accumulated self-optimization score, effort, and efficiency 
    for a given list of Optimization and Target phases.
    """
    if not phases:
        return 

    scores = []
    durations = []
    efforts = []
    types = []
    
    for phase in phases:
        score = phase._calculate_self_optimization_score(u_acc, u_target)
        scores.append(score)

        durations.append(phase.duration)

        effort = phase._calculate_self_configuration_score(c_norm)
        efforts.append(effort)

        types.append(type(phase).__name__)

    # Calculate weighted averages (weighted by phase duration)
    accumulated_score = np.average(scores, weights=durations) if scores else 0.0
    accumulated_effort = np.average(efforts, weights=durations) if efforts else 0.0
    efficiency = accumulated_score / (accumulated_effort + 1)

    # Calculate statistical properties
    score_mean = np.mean(scores) if scores else 0.0
    score_std = np.std(scores) if scores else 0.0

    duration_mean = np.mean(durations) if durations else 0.0
    duration_std = np.std(durations) if durations else 0.0

    efficiency_mean = np.mean([score / (effort + 1) for score, effort in zip(scores, efforts)]) if scores else 0.0
    efficiency_std = np.std([score / (effort + 1) for score, effort in zip(scores, efforts)]) if scores else 0.0

    return {
        "accumulated_self_optimization": accumulated_score,
        "accumulated_self_configuration": accumulated_effort,
        "self_optimization_efficiency": efficiency,
        "valid_phases": len(phases),
        "u_acc": u_acc, 
        "u_target": u_target,
        "self_optimization_score_mean": score_mean,
        "self_optimization_score_std": score_std,
        "self_optimization_duration_mean": duration_mean,
        "self_optimization_duration_std": duration_std,
        "self_optimization_efficiency_mean": efficiency_mean,
        "self_optimization_efficiency_std": efficiency_std,
        "phase_self_optimization": scores,
        "phase_self_configuration": efforts,
        "phase_durations": durations,
        "phase_types": types
    }

def evaluation(base_dir: str, u_acc: float, u_target: float, t_recovery: int, u_min: float, c_norm: float, output_file: str = None, add_robust_system: str = None):
    """
    Parses directory logs to extract phase data, computes OC metrics across all systems,
    and aggregates them into a final JSON report.
    """
    # 1. Scan directory for phase files and load them
    phase_files = [f for f in os.listdir(base_dir) if f.startswith("") and f.endswith("_phases.json")]

    # 2. Create dicts to hold phases by type; keys are the origin file names (systems)
    survival_phases = {}
    optimization_phases = {}
    target_phases = {}

    for file in phase_files:
        with open(os.path.join(base_dir, file), 'r') as f:
            data = json.load(f)
            survival_phases[file] = [from_dict(phase_dict) for phase_dict in data.get("survival_phases", [])]
            optimization_phases[file] = [from_dict(phase_dict) for phase_dict in data.get("optimization_phases", [])]
            target_phases[file] = [from_dict(phase_dict) for phase_dict in data.get("target_phases", [])]

    # 3. Filter Initial Survival Phases
    # Drop the first survival phase if it starts at step 0 to ensure we measure adaptivity 
    # to actual disruptions, rather than rewarding a high initial baseline performance.
    for system, phases in survival_phases.items():
        if phases and phases[0]._start_step == 0:
            survival_phases[system] = phases[1:]

    # 4. Compute adaptivity score for the Survival Phases of all systems
    adaptivity_scores = {}
    u_delta =  u_acc - u_min
    for system, phases in survival_phases.items():
        adaptivity_scores[system] = adaptivity(phases, u_acc, u_delta, u_min, t_recovery, c_norm)

    # 5. Compute self-optimization score for Optimization and Target Phases
    self_optimization_scores = {}
    phases_to_evaluate = {}
    
    for system in optimization_phases.keys():
        phases_to_evaluate[system] = optimization_phases[system] + target_phases.get(system, [])
   
    for system, phases in phases_to_evaluate.items():
        self_optimization_scores[system] = self_optimization(phases, u_acc, u_target, c_norm)

    # 6. Compute the overarching OC Performance Index for each system
    ocp_efficiency = {}
    for system in adaptivity_scores.keys():
        all_phases = phases_to_evaluate.get(system, []) + survival_phases.get(system, [])
        # Pass actual duration to survival phase to get the average over the whole lifetime
        effort = np.mean([phase._calculate_self_configuration_score(c_norm, phase.duration) for phase in all_phases]) 

        adaptivity_efficiency = adaptivity_scores[system]["adaptivity_efficiency"]
        self_optimization_efficiency = self_optimization_scores[system]["self_optimization_efficiency"]
        ocp_efficiency[system] = adaptivity_efficiency * (1 + self_optimization_efficiency) / 2

    # 7. Inject baseline perfectly robust system (if specified)
    if add_robust_system:
        adaptivity_scores[add_robust_system] = {
            "accumulated_adaptivity": 1.0,
            "accumulated_self_configuration": 0.0,
            "adaptivity_efficiency": 1.0,
            "valid_phases": 0,
            "t_max": t_recovery,
            "u_acc": u_acc,
            "u_delta": u_delta,
            "u_min": u_min,
            "adaptivity_score_mean": 1.0,
            "adaptivity_score_std": 0.0,
            "adaptivity_duration_mean": 0.0,
            "adaptivity_duration_std": 0.0,
            "adaptivity_efficiency_mean": 1.0,
            "adaptivity_efficiency_std": 0.0,
            "phase_adaptivity": [],
            "phase_self_configuration": [],
            "phase_durations": []
        }

        self_optimization_scores[add_robust_system] = {
            "accumulated_self_optimization": 1.0,
            "accumulated_self_configuration": 0.0,
            "self_optimization_efficiency": 1.0,
            "valid_phases": 0,
            "u_acc": u_acc,
            "u_target": u_target,
            "self_optimization_score_mean": 1.0,
            "self_optimization_score_std": 0.0,
            "self_optimization_duration_mean": 0.0,
            "self_optimization_duration_std": 0.0,
            "self_optimization_efficiency_mean": 1.0,
            "self_optimization_efficiency_std": 0.0,
            "phase_self_optimization": [],
            "phase_self_configuration": [],
            "phase_durations": [],
            "phase_types": []
        }

        ocp_efficiency[add_robust_system] = 1.0
    
    # 8. Compile and Save Results
    results = {
        "adaptivity": adaptivity_scores,
        "self_optimization": self_optimization_scores,
        "oc_performance_index": ocp_efficiency,
        "evaluation_parameters": {
            "u_acc": u_acc,
            "u_target": u_target,
            "t_max": t_recovery,
            "u_min": u_min
        }
    }

    file_name = output_file + ".json" if output_file else "oc_scores.json"
    dir_name = os.path.dirname(os.path.join(base_dir, file_name))
    os.makedirs(dir_name, exist_ok=True)

    with open(os.path.join(base_dir, file_name), 'w') as f:
        json.dump(results, f, indent=4)


def calculate_min_gap(series):
    """Calculates the minimum distance between any two adjacent scores in a series."""
    sorted_scores = series.sort_values().values
    if len(sorted_scores) < 2:
        return 0.0
    # np.diff calculates the difference between consecutive elements
    return np.min(np.diff(sorted_scores))


def collect_parameter_sweep_scores(folder_path: str) -> pd.DataFrame:
    """
    Parses a folder of JSON files containing system scores and evaluation parameters.
    Flattens the nested data into a tabular format for heatmap plotting and analysis.
    """
    records = {}
    
    # Locate all JSON files in the specified directory
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error parsing JSON in file: {file_path}")
                continue
                
        # Extract the varied parameters for this specific file
        eval_params = data.get("evaluation_parameters", {})
        t_max = eval_params.get("t_max")
        u_min = eval_params.get("u_min")
        U_acc = eval_params.get("u_acc")
        U_target = eval_params.get("u_target")
        
        # Use the keys inside 'adaptivity' as the baseline for all evaluated systems
        if "adaptivity" not in data:
            continue
            
        for system_key in data["adaptivity"].keys():
            # Clean up the system name via regex for readability in tables and plots
            # Example: extracts "Scenario1-ModelCycler" from "log_Scenario1-ModelCycler_2026..."
            match = re.search(r'log_(.*?)_\d{4}-\d{2}-\d{2}', system_key)
            if not match:
                match = re.search(r'(Scenario\d+-\w+)_phases', system_key)
            
            system_name = match.group(1) if match else system_key
            
            # Extract Adaptivity Metrics
            adaptivity_data = data.get("adaptivity", {}).get(system_key, {})
            acc_adaptivity = adaptivity_data.get("accumulated_adaptivity")
            adaptivity_eff = adaptivity_data.get("adaptivity_efficiency")
            survival_effort = adaptivity_data.get("accumulated_self_configuration")
            
            # Extract Self-Optimization Metrics
            self_opt_data = data.get("self_optimization", {}).get(system_key, {})
            acc_self_opt = self_opt_data.get("accumulated_self_optimization")
            self_opt_eff = self_opt_data.get("self_optimization_efficiency")
            optimization_effort = self_opt_data.get("accumulated_self_configuration")
            
            # Extract Top-Level OC Indices
            oc_perf = data.get("oc_performance_index", {}).get(system_key)
            oc_perf_eff = data.get("oc_performance_index_efficiency", {}).get(system_key)
            
            # Compile the record
            records[system_name] = records.get(system_name, []) + [{
                "source_file": os.path.basename(file_path),
                "t_max": t_max,
                "u_min": u_min,
                "u_acc": U_acc,
                "u_target": U_target,
                "accumulated_adaptivity": acc_adaptivity,
                "accumulated_self_configuration_adaptivity": survival_effort,
                "adaptivity_efficiency": adaptivity_eff,
                "accumulated_self_optimization": acc_self_opt,
                "accumulated_self_configuration_self_optimization": optimization_effort,
                "self_optimization_efficiency": self_opt_eff,
                "oc_performance_index": oc_perf,
                "oc_performance_index_efficiency": oc_perf_eff
            }]

    # Write aggregated records to JSON for easier downstream access
    with open(os.path.join(folder_path, "collected_scores.json"), 'w') as f:
        json.dump(records, f, indent=4)


def plot_adaptivity_heatmap(json_filepath: str, system_name: str, ax=None):
    """
    Loads flattened JSON data, parses a specified system's parameter sweep, 
    and plots a 2D heatmap of accumulated adaptivity over t_max and u_min.
    """
    # 1. Load and parse the JSON file
    with open(json_filepath, 'r') as file:
        data = json.load(file)
        
    # 2. Validate that the system exists in the dataset
    if system_name not in data:
        raise KeyError(f"System '{system_name}' not found. Available systems are: {list(data.keys())}")
        
    # 3. Structure data into Pandas DataFrame
    system_records = data[system_name]
    df = pd.DataFrame(system_records)
    
    # 4. Ensure the required columns exist
    required_columns = ['t_max', 'u_min', 'accumulated_adaptivity']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' for '{system_name}'.")
            
    # 5. Pivot and sort the DataFrame for seaborn processing
    pivot_df = df.pivot(index='u_min', columns='t_max', values='accumulated_adaptivity')
    pivot_df = pivot_df.sort_index(ascending=False)
    
    # 6. Plot the heatmap
    if ax is None:
        ax = plt.gca() # Fallback if no ax is provided
        
    heatmap = sns.heatmap(pivot_df, 
                annot=True, 
                annot_kws={"size": 14}, 
                fmt=".3f",           
                cmap="viridis",      
                cbar_kws={'label': '$A_\\text{cum}$'},
                ax=ax)
    
    # Adjust colorbar aesthetics
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14, pad=8)              
    cbar.set_label('$A_\\text{cum}$', size=16, labelpad=20) 
    
    # 7. Format titles and axes
    # Clean up system names for display readability
    title = system_name.split("-")[-1] if "-" in system_name else system_name
    if title == "NonAdaptive":
        title = "Non-Adaptive"
    elif title == "AdaptiveLearner":
        title = "Survival-Focus"
    elif title == "OptimisingLearner":
        title = "Optimization-Focus"

    ax.set_title(f"{title}", pad=15, fontsize=18)             
    ax.set_xlabel("$t_{max}$", fontsize=16, labelpad=12)      
    ax.set_ylabel("$U_{min}$", fontsize=16, labelpad=12)      
    ax.tick_params(axis='x', rotation=45, labelsize=14, pad=6) 
    ax.tick_params(axis='y', rotation=0, labelsize=14, pad=6)  


def plot_distinguishability_comparison(json_filepath: str, metric: str = 'accumulated_adaptivity', systems: list = None, name: str = "Comparison"):
    """
    Parses systems from a parameter sweep dataset, calculates Statistical Variance 
    and Minimum Pairwise Gap for each parameter pairing, and plots them side-by-side.
    """
    # 1. Load the JSON data
    with open(json_filepath, 'r') as file:
        data = json.load(file)
        
    # 2. Flatten the data: combine target systems into a single master record list
    all_records = []
    for system_name, records in data.items():
        if systems is not None and system_name not in systems:
            continue
        for record in records:
            record_copy = record.copy()
            record_copy['system'] = system_name
            all_records.append(record_copy)
            
    # 3. Create the Pandas DataFrame
    df = pd.DataFrame(all_records)
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in the dataset.")
        
    # 4. Calculate Spread Metrics
    # A) Variance
    var_df = df.groupby(['u_min', 't_max'])[metric].var().reset_index()
    pivot_var = var_df.pivot(index='u_min', columns='t_max', values=metric).sort_index(ascending=False)
    
    # B) Minimum Pairwise Gap
    gap_df = df.groupby(['u_min', 't_max'])[metric].apply(calculate_min_gap).reset_index()
    pivot_gap = gap_df.pivot(index='u_min', columns='t_max', values=metric).sort_index(ascending=False)
    
    # 5. Build Side-by-Side Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # --- LEFT PLOT: Variance ---
    heatmap1 = sns.heatmap(pivot_var, annot=True, annot_kws={"size": 14}, fmt=".3f",           
                           cmap="viridis", cbar_kws={'label': 'Variance'}, ax=axes[0])
    cbar1 = heatmap1.collections[0].colorbar
    cbar1.ax.tick_params(labelsize=14, pad=8)              
    cbar1.set_label('Variance', size=16, labelpad=20)      

    axes[0].set_title("Statistical Variance", pad=15, fontsize=18)
    axes[0].set_xlabel("$t_{max}$", fontsize=16, labelpad=12)  
    axes[0].set_ylabel("$U_{min}$", fontsize=16, labelpad=12)  
    axes[0].tick_params(axis='x', rotation=45, labelsize=14, pad=6) 
    axes[0].tick_params(axis='y', labelsize=14, pad=6)              

    # --- RIGHT PLOT: Minimum Pairwise Gap ---
    heatmap2 = sns.heatmap(pivot_gap, annot=True, annot_kws={"size": 14}, fmt=".3f",           
                           cmap="viridis", cbar_kws={'label': 'Minimum Gap'}, ax=axes[1])
    cbar2 = heatmap2.collections[0].colorbar
    cbar2.ax.tick_params(labelsize=14, pad=8)              
    cbar2.set_label('Minimum Gap', size=16, labelpad=20)   

    axes[1].set_title("Minimum Pairwise Gap (Maximin)", pad=15, fontsize=18)
    axes[1].set_xlabel("$t_{max}$", fontsize=16, labelpad=12)  
    axes[1].set_ylabel("") # Suppress Y-label for clean layout

    axes[1].tick_params(axis='x', rotation=45, labelsize=14, pad=6) 
    axes[1].tick_params(axis='y', labelsize=14, pad=6)              

    # 6. Overall Layout adjustments
    plt.subplots_adjust(wspace=0.35)
    plt.tight_layout()
    
    # 7. Export figure
    output_dir = os.path.join(os.path.dirname(json_filepath), name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"distinguishability_comparison_{metric}.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Comparison heatmap saved to: {output_path}")


def plot_combined_systems_to_pdf(csv_files: list, output_pdf_path: str, params: dict, titles: List[str], height_per_system: float = 3.5):
    """
    Plots the runtime utility traces for multiple systems vertically, sharing the 
    x-axis, with titles positioned on the right-hand side, and saves to PDF.
    """
    num_files = len(csv_files)
    
    # 1. Stack vertically and share the X axis
    fig, axes = plt.subplots(nrows=num_files, ncols=1, sharex=True, figsize=(12, height_per_system * num_files))
    if num_files == 1:
        axes = [axes]
    
    # 2. Iterate through files and plot using OcPhases tool
    for i, (ax, csv_path) in enumerate(zip(axes, csv_files)):
        oc_phases = OcPhases(data_path=csv_path, ax=ax, **params)
        oc_phases.update() 
        oc_phases.stop() 
        
        # Axis Cleanup and Title Placement
        ax.set_title(titles[i], fontsize=12, pad=10)
        
        # Remove internal x-axis labels ('Step') to prevent clutter on shared axes
        if i < num_files - 1:
            ax.set_xlabel("")
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

    # 3. Create a Single Master Legend
    handles, labels = axes[0].get_legend_handles_labels()
    
    for ax_i in axes:
        leg = ax_i.get_legend()
        if leg:
            leg.remove()
            
    # Format master legend on bottom center
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    # 4. Layout Adjustments to accommodate master legend
    legend_height_inches = 1.2
    total_figure_height = height_per_system * num_files
    dynamic_bottom_margin = legend_height_inches / total_figure_height

    plt.subplots_adjust(hspace=0.2, bottom=dynamic_bottom_margin)
    
    # 5. Save Output
    fig.savefig(output_pdf_path, bbox_inches='tight')
    print(f"Combined vertical PDF saved to: {output_pdf_path}")
    plt.close(fig)
   

# =========================================================
# Scenario Plotting Wrappers
# Defines configuration and triggers plot generation for the 
# specific experimental scenarios outlined in the thesis.
# =========================================================

def scenario1():
    """Generates the combined trace plot for Scenario 1: Multi-Model Architectures."""
    params = {
        "u_acc": 0.8,
        "u_target": 0.9,
        "optim_phase_threshold": 500,
        "survival_phase_threshold": 500,
        "survival_phase_utility_threshold": 0.9
    }
    
    files_to_plot = [
        "evaluation/scenario-1/log_Scenario1-ModelCycler_2026-06-02_16-36-28.csv",
        "evaluation/scenario-1/log_Scenario1-NonAdaptive_2026-06-02_16-45-25.csv",
        "evaluation/scenario-1/log_Scenario1-PerfectSelector_2026-06-02_16-52-19.csv",
        "evaluation/scenario-1/log_Scenario1-Robust_2026-06-02_16-40-58.csv"
    ]

    titles = ["ModelCycler", "Non-Adaptive", "PerfectSelector", "Robust"]
    
    plot_combined_systems_to_pdf(files_to_plot, "figures/scenario-1-traces.pdf", params, titles)


def scenario2():
    """Generates the combined trace plot for Scenario 2: Failing Adaptive Agent."""
    params = {
        "u_acc": 0.8,
        "u_target": 0.9,
        "optim_phase_threshold": 1000,
        "survival_phase_threshold": 500,
        "survival_phase_utility_threshold": 0.9
    }

    files_to_plot = [
        "evaluation/scenario-2/log_Scenario2-NonAdaptive.csv",
        "evaluation/scenario-2/log_Scenario2-Failing.csv",
        "evaluation/scenario-2/log_Scenario2-AdaptiveLearner.csv",
        "evaluation/scenario-2/log_Scenario2-OptimisingLearner.csv",
    ]

    titles = ["Non-Adaptive", "Failing", "Survival-Focus", "Optimization-Focus"]

    plot_combined_systems_to_pdf(files_to_plot, "figures/scenario-2-traces.pdf", params, titles)


def scenario3():
    """Generates the combined trace plot for Scenario 3: Continual Learning Comparison."""
    params = {
        "u_acc": 0.8,
        "u_target": 0.9,
        "optim_phase_threshold": 1000,
        "survival_phase_threshold": 500,
        "survival_phase_utility_threshold": 0.9
    }

    files_to_plot = [
        "evaluation/scenario-3/log_Scenario3-NonAdaptive.csv",
        "evaluation/scenario-3/log_Scenario3-AdaptiveLearner.csv",
        "evaluation/scenario-3/log_Scenario3-OptimisingLearner.csv"
    ]

    titles = ["Non-Adaptive", "Survival-Focus", "Optimization-Focus"]

    plot_combined_systems_to_pdf(files_to_plot, "figures/scenario-3-traces.pdf", params, titles)