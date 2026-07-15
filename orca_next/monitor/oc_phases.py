from abc import ABC, abstractmethod
import json
import os
import re
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Optional

import pandas as pd

class Phase(ABC):
    """
    Abstract base class representing an operational phase of a system.
    Instances of this class are strictly immutable after creation.
    """
    def __init__(
            self, 
            u_acc: float,
            u_target: float,
            timespan: tuple[int, int], 
            utility_trace: np.ndarray, 
            self_configuration_trace: np.ndarray
        ):
        # Basic validation
        if timespan[0] > timespan[1]:
            raise ValueError("Start step must be less than or equal to end step.")

        # Assign private variables
        self._start_step: int = timespan[0]
        self._end_step: int = timespan[1]
        
        # Copy arrays to prevent external mutation by reference
        self._utility_trace: np.ndarray = np.array(utility_trace, copy=True)
        self._self_configuration_trace: np.ndarray = np.array(self_configuration_trace, copy=True)

    def to_dict(self) -> dict:
        """Helper to convert the phase data into a dictionary format for JSON serialization."""
        return {
            "phase_type": self.__class__.__name__,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "duration": self.duration,
            "utility_trace": self.utility_trace.tolist(),
            "self_configuration_trace": self.self_configuration_trace.tolist()
        }

    # ==========================================
    # Public Read-Only Properties
    # ==========================================

    @property
    def start_step(self) -> int:
        return self._start_step

    @property
    def end_step(self) -> int:
        return self._end_step
    
    @property
    def duration(self) -> int:
        """Helper property to get the length of the phase."""
        return self._end_step - self._start_step

    @property
    def utility_trace(self) -> np.ndarray:
        # Return a copy so the caller cannot modify the internal array
        return self._utility_trace.copy()

    @property
    def self_configuration_trace(self) -> np.ndarray:
        # Return a copy so the caller cannot modify the internal array
        return self._self_configuration_trace.copy()

    # ==========================================
    # Abstract Internal Methods
    # ==========================================

    @abstractmethod
    def _calculate_adaptivity_score(self, t_max: float, u_delta: float, u_min: float) -> float:
        """Calculate the adaptivity score. Must be implemented by subclass."""
        pass

    @abstractmethod
    def _calculate_self_optimization_score(self, u_acc: float, u_target: float) -> float:
        """Calculate the self-optimization score. Must be implemented by subclass."""
        pass

    @abstractmethod
    def _calculate_self_configuration_score(self, c_norm: float, t_max: float = 1.0) -> float:
        """Calculate the self-configuration effort score. Must be implemented by subclass."""
        pass

    def _calculate_adaptivity_weight(self, t_max: float) -> float:
        """Helper to calculate a time-based weight for adaptivity scoring."""
        if t_max <= 0:
            return 1.0
        
        return np.ceil(self.duration / t_max)

class SurvivalPhase(Phase):
    def _calculate_adaptivity_score(self, t_max: float, u_delta: float, u_min: float) -> float:    
        
        weight = self._calculate_adaptivity_weight(t_max)
        t_eval = weight * t_max
        
        u_acc = u_min + u_delta
        u_clip = np.minimum(self.utility_trace, u_acc)
        u_clip = np.maximum(u_clip, u_min)  # Ensure we don't count anything below u_min

        return (np.sum(u_clip - u_min) + u_delta * (t_eval - self.duration)) / (t_eval * u_delta) / weight if t_eval > 0 and u_delta > 0 else 0.0

    def _calculate_self_optimization_score(self, u_acc: float, u_target: float) -> float:
        # The self-optimization score of a survival phase is per definition 0
        return 0.0

    def _calculate_self_configuration_score(self, c_norm: float, t_max: float) -> float:
        """Calculate the self-configuration score."""
        # Clip the score at 1.0, which can occur if the in failure to adapt if duration > t_max

        weight = self._calculate_adaptivity_weight(t_max)
        t_eval = weight * t_max

        return np.minimum(np.sum(self.self_configuration_trace) / (c_norm * t_eval), 1.0) / weight if c_norm > 0 and t_max > 0 else 0.0

class OptimizationPhase(Phase):
    def _calculate_adaptivity_score(self, t_max: float, u_delta: float, u_min: float) -> float:
        # The adaptivity score of an optimization phase is per definition 1
        return 1.0

    def _calculate_self_optimization_score(self, u_acc: float, u_target: float) -> float:
        capped_utility = np.minimum(self.utility_trace, u_target)
        capped_utility = np.maximum(capped_utility, u_acc)  # Ensure we don't count anything below u_acc
        surplus_utility = capped_utility - u_acc
        summed_surplus = np.sum(surplus_utility)
        normalization_factor = (u_target - u_acc) * self.duration
        if normalization_factor == 0:
            return 0.0
        self_optimization_score = summed_surplus / normalization_factor
        return self_optimization_score
        
        # return (np.sum(np.minimum(self.utility_trace, u_target)) - u_acc) / ((u_target - u_acc) * self.duration)
    
    def _calculate_self_configuration_score(self, c_norm: float, t_max: float = 1.0) -> float:
        """Calculate the self-configuration score."""
        summed_config = np.sum(self.self_configuration_trace)
        norm1 = summed_config / (c_norm) if c_norm > 0 else 0.0
        norm2 = norm1 / (self.duration) if self.duration > 0 else 0.0
        return norm2
    
class TargetPhase(Phase):
    def _calculate_adaptivity_score(self, t_max: float, u_delta: float, u_min: float) -> float:
        # The adaptivity score of a target phase is per definition 1
        return 1.0

    def _calculate_self_optimization_score(self, u_acc: float, u_target: float) -> float:
        # The self-optimization score of a target phase is per definition 1
        return 1.0

    def _calculate_self_configuration_score(self, c_norm: float, t_max: float = 1.0) -> float:
        """Calculate the self-configuration score."""
        return np.sum(self.self_configuration_trace) / (c_norm * self.duration) if c_norm > 0 and self.duration > 0 else 0.0

# ==========================================
# Helper Functions
# ==========================================

# Helper function to reconstruct a Phase object from a dictionary (e.g., loaded from JSON)
def from_dict(phase_dict: dict) -> Phase:
    phase_type = phase_dict.get("phase_type")
    if phase_type == "SurvivalPhase":
        return SurvivalPhase(
            u_acc=phase_dict.get("u_acc", 0.4),
            u_target=phase_dict.get("u_target", 0.46),
            timespan=(phase_dict["start_step"], phase_dict["end_step"]),
            utility_trace=np.array(phase_dict["utility_trace"]),
            self_configuration_trace=np.array(phase_dict["self_configuration_trace"])
        )
    elif phase_type == "OptimizationPhase":
        return OptimizationPhase(
            u_acc=phase_dict.get("u_acc", 0.4),
            u_target=phase_dict.get("u_target", 0.46),
            timespan=(phase_dict["start_step"], phase_dict["end_step"]),
            utility_trace=np.array(phase_dict["utility_trace"]),
            self_configuration_trace=np.array(phase_dict["self_configuration_trace"])
        )
    elif phase_type == "TargetPhase":
        return TargetPhase(
            u_acc=phase_dict.get("u_acc", 0.4),
            u_target=phase_dict.get("u_target", 0.46),
            timespan=(phase_dict["start_step"], phase_dict["end_step"]),
            utility_trace=np.array(phase_dict["utility_trace"]),
            self_configuration_trace=np.array(phase_dict["self_configuration_trace"])
        )
    else:
        raise ValueError(f"Unknown phase type in JSON: {phase_type}")

class OcPhases:
    def __init__(
            self, 
            data_path: str, 
            u_acc: float, 
            u_target: float, 
            optim_phase_threshold: int,
            survival_phase_threshold: int,
            survival_phase_utility_threshold: float,
            stop_event=None,
            ax=None
        ):


        self.stop_event = stop_event
        self.data_path = data_path

        self.u_acc = u_acc
        self.u_target = u_target

        self.optim_phase_threshold = optim_phase_threshold
        self.survival_phase_threshold = survival_phase_threshold
        self.survival_phase_utility_threshold = survival_phase_utility_threshold

        self.phases: List[Phase] = []
        
        # State tracking for the active (ongoing) phase
        self.current_step = 0
        self.current_phase_type: Optional[str] = None
        self.current_phase_start_step = 0
        
        # Buffers for the ongoing phase
        self._utility_buffer: List[float] = []
        self._config_buffer: List[float] = []

        # Buffers for aggregate scores across all phases
        self._cum_adaptivity: List[float] = []
        self._cum_adaptivity_finalization_steps: List[int] = []

        self._cum_self_optimisation: List[float] = []
        self._cum_self_optimisation_finalization_steps: List[int] = []


        if ax is None:
            # Original behavior: Create a new figure
            self.fig, self.ax = plt.subplots(figsize=(15, 5))
            plt.tight_layout(pad=3.0)
            self._owns_figure = True
        else:
            # Combined behavior: Use the provided axes
            self.ax = ax
            self.fig = ax.figure
            self._owns_figure = False

    def _determine_phase_type(self, utility: float) -> str:
        """Helper to map a utility value to its phase type."""
        if utility < self.u_acc:
            return "survival"
        elif utility < self.u_target:
            return "optimization"
        else:
            return "target"

    def _finalize_active_phase(self, new_phase_type: str = None):
        """Instantiates the correct Phase object and appends it to history."""
        if self.current_phase_type is None or len(self._utility_buffer) == 0:
            return

        timespan = (self.current_phase_start_step, self.current_step)
        
        # Select the correct class based on the type
        if self.current_phase_type == "survival":
            phase_cls = SurvivalPhase
        elif self.current_phase_type == "optimization":
            phase_cls = OptimizationPhase
        elif self.current_phase_type == "target":
            phase_cls = TargetPhase
        else:
            raise ValueError(f"Unknown phase type: {self.current_phase_type}")

        # Instantiate the immutable phase object
        phase_obj = phase_cls(
            u_acc=self.u_acc,
            u_target=self.u_target,
            timespan=timespan,
            utility_trace=np.array(self._utility_buffer),
            self_configuration_trace=np.array(self._config_buffer)
        )

        finalized_is_optim = isinstance(phase_obj, OptimizationPhase)
        finalized_is_short_optim = phase_obj.duration < self.optim_phase_threshold    
        last_is_survival = len(self.phases) > 0 and isinstance(self.phases[-1], SurvivalPhase)
        next_is_survival = new_phase_type == "survival"

        finalized_is_survival = isinstance(phase_obj, SurvivalPhase)
        finalized_is_short_survival = phase_obj.duration < self.survival_phase_threshold
        utility_not_below_threshold = np.all(phase_obj.utility_trace >= self.survival_phase_utility_threshold * self.u_acc)
        last_is_optim = len(self.phases) > 0 and isinstance(self.phases[-1], OptimizationPhase)
        next_is_optim = new_phase_type == "optimization"

        if (finalized_is_optim and finalized_is_short_optim) and last_is_survival and next_is_survival:
            last_phase = self.phases[-1]
            merged_phase = SurvivalPhase(
                u_acc=self.u_acc,
                u_target=self.u_target,
                timespan=(last_phase.start_step, phase_obj.end_step),
                utility_trace=np.concatenate([last_phase.utility_trace, phase_obj.utility_trace]),
                self_configuration_trace=np.concatenate([last_phase.self_configuration_trace, phase_obj.self_configuration_trace])
            )
            self.phases[-1] = merged_phase  # Replace the last phase with the merged one

        elif finalized_is_survival and finalized_is_short_survival and last_is_optim and next_is_optim and utility_not_below_threshold:
            last_phase = self.phases[-1]
            merged_phase = OptimizationPhase(
                u_acc=self.u_acc,
                u_target=self.u_target,
                timespan=(last_phase.start_step, phase_obj.end_step),
                utility_trace=np.concatenate([last_phase.utility_trace, phase_obj.utility_trace]),
                self_configuration_trace=np.concatenate([last_phase.self_configuration_trace, phase_obj.self_configuration_trace])
            )
            self.phases[-1] = merged_phase  # Replace the last phase with the merged one

        elif finalized_is_survival and last_is_survival:
            last_phase = self.phases[-1]
            merged_phase = SurvivalPhase(
                u_acc=self.u_acc,
                u_target=self.u_target,
                timespan=(last_phase.start_step, phase_obj.end_step),   
                utility_trace=np.concatenate([last_phase.utility_trace, phase_obj.utility_trace]),
                self_configuration_trace=np.concatenate([last_phase.self_configuration_trace, phase_obj.self_configuration_trace])
            )
            self.phases[-1] = merged_phase  # Replace the last phase with the merged

        elif finalized_is_optim and last_is_optim:
            last_phase = self.phases[-1]
            merged_phase = OptimizationPhase(
                u_acc=self.u_acc,
                u_target=self.u_target,
                timespan=(last_phase.start_step, phase_obj.end_step),   
                utility_trace=np.concatenate([last_phase.utility_trace, phase_obj.utility_trace]),
                self_configuration_trace=np.concatenate([last_phase.self_configuration_trace, phase_obj.self_configuration_trace])
            )
            self.phases[-1] = merged_phase  # Replace the last phase with the merged one

        else:
            self.phases.append(phase_obj)

    def _update(self, utility: float, self_configuration: float = 0.0):
        """Called every step to provide new metrics."""
        new_phase_type = self._determine_phase_type(utility)

        # Initialize the very first phase
        if self.current_phase_type is None:
            self.current_phase_type = new_phase_type
            self.current_phase_start_step = self.current_step

        # Detect Phase Transition
        elif new_phase_type != self.current_phase_type:
            # 1. Finalize the old phase
            self._finalize_active_phase(new_phase_type)
            
            # 2. Reset buffers and state for the new phase
            self._utility_buffer.clear()
            self._config_buffer.clear()
            self.current_phase_type = new_phase_type
            self.current_phase_start_step = self.current_step

        # Append data to the active buffers
        self._utility_buffer.append(utility)
        self._config_buffer.append(self_configuration)

        self.current_step += 1

    # ==========================================
    # Helpers:
    # ==========================================
    def _get_phase_color(self, phase_identifier) -> str:
        """Returns the appropriate color based on phase type or string name."""
        if isinstance(phase_identifier, str):
            p_type = phase_identifier
        else:
            if isinstance(phase_identifier, SurvivalPhase): p_type = "survival"
            elif isinstance(phase_identifier, OptimizationPhase): p_type = "optimization"
            elif isinstance(phase_identifier, TargetPhase): p_type = "target"
            else: p_type = "unknown"

        if p_type == "survival": return "orange"
        if p_type == "optimization": return "green"
        if p_type == "target": return "blue"
        return "gray"

    def _get_phase_priority(self, phase_identifier) -> int:
        """Returns the stitching priority: Survival (3) > Optimization (2) > Target (1)."""
        if isinstance(phase_identifier, str):
            p_type = phase_identifier
        else:
            if isinstance(phase_identifier, SurvivalPhase): p_type = "survival"
            elif isinstance(phase_identifier, OptimizationPhase): p_type = "optimization"
            elif isinstance(phase_identifier, TargetPhase): p_type = "target"
            else: p_type = "unknown"

        if p_type == "survival": return 3
        if p_type == "optimization": return 2
        return 1 # Target phases (and unknowns) get the lowest priority

    # ==========================================
    # Data Processing Pipeline
    # ==========================================
    def _process_new_csv_data(self):
        """Reads new rows from the CSV and updates internal buffers."""
        if not os.path.exists(self.data_path):
            return
        
        try:
            data = pd.read_csv(self.data_path)
            if data.empty or 'utility' not in data.columns or 'self-configuration' not in data.columns:
                return
            
            # Iterate only over new steps we haven't processed yet
            for step in range(self.current_step, len(data)):
                utility = data.at[step, 'utility']
                config = data.at[step, 'self-configuration']

                if not pd.isna(utility) and not pd.isna(config):
                    self._update(utility, config)
                    
        except Exception as e:
            # Catch half-written rows or pandas parsing errors silently during live updates
            print(f"Warning during live data read: {e}")

    # ==========================================
    # Rendering Pipeline
    # ==========================================
    def _setup_plot_axes(self):
        """Configures the aesthetic background, grid, thresholds, and labels."""
        self.ax.clear()
        self.ax.grid(True, linestyle=':', alpha=0.6)

        # Draw threshold lines
        self.ax.axhline(self.u_acc, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        self.ax.axhline(self.u_target, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)

        # Subtly shade the threshold zones
        self.ax.axhspan(0, self.u_acc, color='orange', alpha=0.03)
        self.ax.axhspan(self.u_acc, self.u_target, color='green', alpha=0.03)
        self.ax.axhspan(self.u_target, 1.05, color='blue', alpha=0.03)

        self.ax.set_ylim(0.0, 1.05)
        self.ax.set_title("Utility Phases")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Utility / Configuration Effort")

    def _plot_historical_phases(self):
        """Plots completed phases with priority-based stitching to prevent visual gaps."""
        for i, phase in enumerate(self.phases):
            x_vals = np.arange(phase.start_step, phase.end_step)
            y_vals = phase.utility_trace
            c_vals = phase.self_configuration_trace
            
            color = self._get_phase_color(phase)
            curr_prio = self._get_phase_priority(phase)

            # === STITCHING LOGIC ===
            
            # 1. Stitch Left (if we dominate the previous phase)
            if i > 0:
                prev_phase = self.phases[i-1]
                if curr_prio > self._get_phase_priority(prev_phase):
                    x_vals = np.insert(x_vals, 0, prev_phase.end_step - 1)
                    y_vals = np.insert(y_vals, 0, prev_phase.utility_trace[-1])
                    c_vals = np.insert(c_vals, 0, prev_phase.self_configuration_trace[-1])

            # 2. Stitch Right (if we dominate/tie the next phase AND we are not a Target phase)
            if curr_prio > 1:  # Enforces: "target phases never stitch"
                if i < len(self.phases) - 1:
                    next_phase = self.phases[i+1]
                    if curr_prio >= self._get_phase_priority(next_phase):
                        x_vals = np.append(x_vals, next_phase.start_step)
                        y_vals = np.append(y_vals, next_phase.utility_trace[0])
                        c_vals = np.append(c_vals, next_phase.self_configuration_trace[0])
                elif len(self._utility_buffer) > 0:
                    # Check gap against the active ongoing phase
                    if curr_prio >= self._get_phase_priority(self.current_phase_type):
                        x_vals = np.append(x_vals, self.current_phase_start_step)
                        y_vals = np.append(y_vals, self._utility_buffer[0])
                        c_vals = np.append(c_vals, self._config_buffer[0])

            self.ax.plot(x_vals, y_vals, color=color, linewidth=1)
            self.ax.plot(x_vals, c_vals, color="gray", alpha=0.3, linewidth=1)


    def _plot_active_phase(self):
        """Plots the currently ongoing phase buffer."""
        if not self._utility_buffer:
            return
            
        active_x = np.arange(self.current_phase_start_step, self.current_step)
        active_y = np.array(self._utility_buffer)
        active_c = np.array(self._config_buffer)
        
        color = self._get_phase_color(self.current_phase_type)
        curr_prio = self._get_phase_priority(self.current_phase_type)

        # Active phase stitches left if it dominates the last historical phase
        if len(self.phases) > 0 and curr_prio > 1:
            last_phase = self.phases[-1]
            if curr_prio > self._get_phase_priority(last_phase):
                active_x = np.insert(active_x, 0, last_phase.end_step - 1)
                active_y = np.insert(active_y, 0, last_phase.utility_trace[-1])
                active_c = np.insert(active_c, 0, last_phase.self_configuration_trace[-1])

        self.ax.plot(active_x, active_y, color=color, linewidth=1.)
        self.ax.plot(active_x, active_c, color="gray", linewidth=1., alpha=0.3)
    def _add_plot_legend(self):
        """Injects dummy lines to construct a clean, consistent legend."""
        self.ax.plot([], [], color='orange', linewidth=2.0, label='Survival Phase')
        self.ax.plot([], [], color='green', linewidth=2.0, label='Acceptable Phase')
        self.ax.plot([], [], color='blue', linewidth=2.0, label='Target Phase')
        self.ax.plot([], [], color='gray', alpha=0.5, linewidth=1.5, label='Self-Configuration Effort')
        self.ax.plot([], [], color='red', linestyle='--', label='$U_{acc}$')
        self.ax.plot([], [], color='blue', linestyle='--', label='$U_{target}$')
        
        if len(self.ax.lines) > 0:
            self.ax.legend(loc='lower right', framealpha=0.9)

    # ==========================================
    # Main Update Loop (Called by FuncAnimation)
    # ==========================================
    def update(self, frame=None):
        """Main step function mapping data updates to visual rendering."""
        # 1. Ingest new data
        self._process_new_csv_data()
        
        # 2. Render visuals
        self._setup_plot_axes()
        self._plot_historical_phases()
        self._plot_active_phase()
        self._add_plot_legend()

        # 3. Handle Graceful Shutdown
        if self.stop_event and self.stop_event.is_set():
            self.stop()
            return

    def flush(self):
        """Call this at the end of the simulation to save the final incomplete phase."""
        self._finalize_active_phase()
        self._utility_buffer.clear()
        self._config_buffer.clear()
        self.current_phase_type = None

    def start(self):
        # Update every 1000ms
        self.ani = FuncAnimation(self.fig, self.update, interval=500, cache_frame_data=False)
        plt.show()

    def stop(self):
        print("Saving final phase plot and structured data...")
        if hasattr(self, 'ani') and self.ani and self.ani.event_source:
            self.ani.event_source.stop()

        # 1. Ensure the final data points are flushed into a phase before saving
        self.flush()
        
        base_path = self.data_path.replace('.csv', '')

        # ==========================================
        # Save Structured JSON Data
        # ==========================================
        survival_phases = []
        optimization_phases = []
        target_phases = []

        for i, phase in enumerate(self.phases):
            if isinstance(phase, SurvivalPhase):
                survival_phases.append(phase.to_dict())
            elif isinstance(phase, OptimizationPhase):
                optimization_phases.append(phase.to_dict())
            elif isinstance(phase, TargetPhase):
                target_phases.append(phase.to_dict())
        phases_data = {
            "survival_phases": survival_phases,
            "optimization_phases": optimization_phases,
            "target_phases": target_phases
        }
        json_path = f"{base_path}_phases.json"
        try:
            with open(json_path, 'w') as f:
                json.dump(phases_data, f, indent=4)
            print(f"Structured data saved to {json_path}")
        except Exception as e:
            print(f"Error saving structured data: {e}")

        # ==========================================
        # Save Figure
        # ==========================================
        if self._owns_figure:
            png_path = f"{base_path}_phases.png"
            try:
                self.fig.savefig(png_path, bbox_inches='tight')
                print(f"Plot saved successfully to {png_path}")
            except Exception as e:
                print(f"Error saving plot: {e}")

            plt.close(self.fig)

    # ==========================================
    # Global Trace Retrieval
    # ==========================================

    def get_concatenated_utility_trace(self) -> np.ndarray:
        """Returns the full utility trace from start to finish."""
        # 1. Collect traces from all completed phases
        traces = [p.utility_trace for p in self.phases]
        
        # 2. Append the active buffer if it has data (so real-time plots don't lag)
        if self._utility_buffer:
            traces.append(np.array(self._utility_buffer))
            
        if not traces:
            return np.array([])
            
        return np.concatenate(traces)

