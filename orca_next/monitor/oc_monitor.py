

import csv
from datetime import datetime
import os
from multiprocessing import Event, Process

from core.audit_logger import AuditLogger
from core.mediator import Mediator
from core.messages import Message
from monitor.oc_monitor_plotter import MonitorPlotter
from monitor.oc_phases import OcPhases
from monitor.oc_state import OcState
from typing import Callable, Dict

from utils.context import Context

from monitor.base_metric import auto_load_metrics

def _run_plotter(log_path, selected_metrics, plotting_limits, stop_event):
    plotter = MonitorPlotter(log_path, selected_metrics, plotting_limits, stop_event)
    plotter.start()

def _run_phases(log_path, u_acc, u_target, optim_phase_threshold, survival_phase_threshold, survival_phase_utility_threshold, stop_event):
    phases = OcPhases(log_path, u_acc, u_target, optim_phase_threshold, survival_phase_threshold, survival_phase_utility_threshold, stop_event)
    phases.start()

class OCMonitor:
    """
    Core-level monitor.
    """

    def __init__(self, mediator: Mediator, monitor_config: dict, title: str = "Runtime Monitor"):
        """
        Initialize the OCMonitor.

        Args:
            mediator (Mediator): Mediator instance used to publish metric messages and
                obtain latest messages/state snapshots.
            monitor_config (dict): Configuration dictionary for the monitor. Expected
                keys include "metrics" (list of metric configurations) and optional
                "scores" for phase detection parameters.
            title (str): Human-readable title used for the log filename.

        Behavior:
            - Creates a monitoring directory and a timestamped CSV log file path.
            - Loads available metric classes via auto_load_metrics and instantiates
              configured metrics.
            - Starts background processes for plotting and phase detection if
              requested in the configuration.
        """

        self.mediator = mediator
        self.oc_state = OcState(monitor_config)
        self.title = title

        os.makedirs("monitoring", exist_ok=True)

        self.plot = False

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = f"monitoring/log_{self.title}_{timestamp_str}.csv"
       
        self.step_count = 0

        metric_registry = auto_load_metrics()
        aggregate_registry = {}  # For future use if we implement aggregators
        
        metrics_from_states = []
        metrics_from_metrics = []

        metrics_to_plot = []
        plotting_limits = []

        for metric_conf in monitor_config.get("metrics", []):
            metric_name = metric_conf.get("name")
            metric_type = metric_conf.get("operation")
            plot_metric = metric_conf.get("plot", False)
            plotting_limit = metric_conf.get("plotting_limits", None)
            
            if metric_name is None or metric_type is None:
                raise ValueError(f"Invalid metric configuration: {metric_conf}")
            if metric_type not in metric_registry:
                raise ValueError(f"Unknown metric type '{metric_type}' in configuration: {metric_conf}")
            
            metric_cls = metric_registry[metric_type]
            metric_obj = metric_cls(metric_name, metric_conf)

            if metric_obj.uses_only_states():
                metrics_from_states.append(metric_obj)
            else:
                metrics_from_metrics.append(metric_obj)

            if plot_metric:
                self.plot = True
                metrics_to_plot.append(metric_name) 
                plotting_limits.append(plotting_limit)

        metrics_from_metrics = self._sort_metrics_by_dependencies(metrics_from_metrics)
        self.metrics = metrics_from_states + metrics_from_metrics

        self.plot_stop_event = Event()
        if self.plot:
            # Create a cross-process event flag
            self.plot_proc = Process(
                target=_run_plotter, 
                args=(self.log_path, metrics_to_plot, plotting_limits, self.plot_stop_event), 
                daemon=True
            )
            self.plot_proc.start()

        self.phases_stop_event = Event()
        if "scores" in monitor_config:
            self.u_acc = monitor_config["scores"].get("u_acc", 0.4)
            self.u_target = monitor_config["scores"].get("u_target", 0.46)
            self.optim_phase_threshold = monitor_config["scores"].get("optim_phase_threshold", 200)
            self.survival_phase_threshold = monitor_config["scores"].get("survival_phase_threshold", 100)
            self.survival_phase_utility_threshold = monitor_config["scores"].get("survival_phase_utility_threshold", 0.95)

            self.phases_proc = Process(
                target=_run_phases,
                args=(self.log_path, self.u_acc, self.u_target, self.optim_phase_threshold, self.survival_phase_threshold, self.survival_phase_utility_threshold, self.phases_stop_event),
                daemon=True
            )
            self.phases_proc.start()


    # Call this method when your main program is terminating!
    def shutdown_plotter(self):
        """Signal background plotter and phase processes to stop and wait for them.

        This will set the cross-process events used by the plotter and phases
        processes. If the processes were started, the method will attempt to
        join them with a short timeout to allow graceful shutdown.
        """
        # Signal stop events if they exist
        if hasattr(self, 'plot_stop_event') and self.plot_stop_event is not None:
            self.plot_stop_event.set()

        if hasattr(self, 'phases_stop_event') and self.phases_stop_event is not None:
            self.phases_stop_event.set()

        # Wait for the processes to finish (if they were started)
        if hasattr(self, 'plot_proc') and getattr(self, 'plot_proc') is not None and self.plot_proc.is_alive():
            self.plot_proc.join(timeout=3)

        if hasattr(self, 'phases_proc') and getattr(self, 'phases_proc') is not None and self.phases_proc.is_alive():
            self.phases_proc.join(timeout=3)

    def step(self):
        """Perform one monitoring step.

        Sequence:
        1. Extract and update internal state from the mediator.
        2. Compute all configured metrics in dependency order.
           - If a metric's inputs are not available, its value will be None.
           - Metric results are stored in step_results and optionally saved or published
             according to metric flags.
        3. Append values that must be persisted to the CSV log.

        This method increments the internal step counter after processing.
        """

        # 1. Extract state from mediator
        current_snapshot, state_to_save = self.oc_state.update(self.mediator.latest_messages, self.step_count)

        # 2. Compute all metrics
        step_results = {"step": self.step_count}
        results_to_log = state_to_save.copy()  # Start with state values that need to be saved, then add metric results

        for metric in self.metrics:
            # check if inputs for this metric are available in step_results or current_snapshot
            if not metric.inputs_available(current_snapshot, step_results):
                val = None  # or some default value indicating missing inputs
            else:
                val = metric(current_snapshot, step_results)

            step_results[metric.name] = val
            if metric.has_to_be_saved():
                results_to_log[metric.name] = val

            if metric.has_to_be_published():
                publish_msg = Message(
                    topic=metric.name,
                    sender="OCMonitor",
                    payload=Context.with_info({metric.name: val})
                )
                self.mediator.publish(publish_msg, cycle=1, persist_reset=True)

        # 3. Save to CSV
        self._save_to_csv(results_to_log)

        self.step_count += 1

    def _sort_metrics_by_dependencies(self, metrics):
        """
        Topologically sort metrics according to their metric-to-metric dependencies.

        Each metric is expected to expose an attribute `input_metrics` which is an
        iterable of metric names it depends on. Only dependencies that refer to
        other metrics in the provided `metrics` list are considered; missing
        dependencies are ignored (they may be provided by state values instead).

        This implementation performs a DFS-based topological sort and will
        raise a ValueError if a cyclic dependency is detected among the
        supplied metrics.

        Args:
            metrics (Iterable): list of metric objects with `name` and
                `input_metrics` attributes.

        Returns:
            List: metrics sorted so that dependencies come before dependents.
        """

        sorted_metrics = []
        visited = set()

        def visit(metric):
            if metric.name in visited:
                return
            visited.add(metric.name)
            for dep in metric.input_metrics:
                dep_metric = next((m for m in metrics if m.name == dep), None)
                if dep_metric:
                    visit(dep_metric)
            sorted_metrics.append(metric)

        for metric in metrics:
            visit(metric)

        return sorted_metrics

    def _save_to_csv(self, results):
        """Append a dictionary of results to the CSV log file.

        Creates the file and writes a header row if it does not already exist.

        Args:
            results (dict): Mapping of column names to values to be written as one CSV row.
        """

        file_exists = os.path.exists(self.log_path)
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)
            f.flush()

    def publish_initial(self):
        """Publish initial values for metrics that require an initial broadcast.

        For each metric whose has_to_be_published() returns True, publish a
        Message with a payload containing the metric name mapped to None. The
        mediator is invoked with cycle=1 and persist_reset=True to ensure the
        initial state is available to subscribers.
        """

        for metric in self.metrics:
            if metric.has_to_be_published():
                publish_msg = Message(
                    topic=metric.name,
                    sender="OCMonitor",
                    payload=Context.with_info({metric.name: None})  # Initial value can be None or some default
                )
                self.mediator.publish(publish_msg, cycle=1, persist_reset=True)

    def shutdown(self):
        """Perform shutdown procedures for the monitor.

        Currently this triggers plotter shutdown. Additional cleanup can be
        added here as needed.
        """

        self.shutdown_plotter()
            