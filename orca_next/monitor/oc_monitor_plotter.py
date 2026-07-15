import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MonitorPlotter:
    def __init__(self, log_path: str, selected_metrics: list, plotting_limits: list = None, stop_event=None):
        self.log_path = log_path
        self.selected_metrics = selected_metrics
        self.plotting_limits = plotting_limits or [None] * len(selected_metrics)
        
        # Store the event
        self.stop_event = stop_event
        
        # 1. Group metrics by their limits
        # We use a string representation of the list as a key for the dictionary
        self.groups = {}
        for metric, limit in zip(self.selected_metrics, self.plotting_limits):
            limit_key = tuple(limit) if limit else None
            if limit_key not in self.groups:
                self.groups[limit_key] = []
            self.groups[limit_key].append(metric)

        # 2. Setup Plotting grid (one subplot per group)
        num_groups = len(self.groups)
        self.fig, self.axs = plt.subplots(num_groups, 1, sharex=True, figsize=(10, 4 * num_groups))
        
        # Ensure axs is always iterable even if there's only one plot
        if num_groups == 1:
            self.axs = [self.axs]
            
        plt.tight_layout(pad=3.0)

    def update(self, frame):
        # Check if the main process wants to stop
        if self.stop_event and self.stop_event.is_set():
            self.stop()
            return

        if not os.path.exists(self.log_path):
            return
        
        try:
            data = pd.read_csv(self.log_path)
            if data.empty:
                return
            
            # Use data index as the step count
            steps = data.index

            # 3. Update each group/subplot
            for ax, (limit, metrics) in zip(self.axs, self.groups.items()):
                ax.clear()
                for metric in metrics:
                    if metric in data.columns:
                        d = pd.to_numeric(data[metric], errors='coerce')  # Convert to numeric, coerce errors to NaN
                        
                        # Create a boolean mask of where the data is NOT NaN
                        valid_mask = d.notna()

                        # Filter BOTH the x (steps) and y (d) arrays using the mask
                        # Note: np.array(steps) ensures this works even if steps is a standard Python list
                        steps_clean = np.array(steps)[valid_mask]
                        d_clean = d[valid_mask]

                        # Plot the filtered data
                        ax.plot(steps_clean, d_clean, label=metric)
                
                if limit:
                    ax.set_ylim(limit)
                
                ax.set_title(f"Metrics Group: {', '.join(metrics)}")
                ax.legend(loc='upper left')
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)
            
            self.axs[-1].set_xlabel("Step")
            
        except Exception as e:
            print(f"Plotting error: {e}")

    def start(self):
        # Update every 1000ms
        ani = FuncAnimation(self.fig, self.update, interval=1000, cache_frame_data=False)
        plt.show()

    def stop(self):
        print("Saving final plot...")
        # Save the final plot as a PNG before closing
        try:
            self.fig.savefig(self.log_path.replace('.csv', '.png'), bbox_inches='tight')
            print(f"Plot saved successfully to {self.log_path.replace('.csv', '.png')}")
        except Exception as e:
            print(f"Error saving plot: {e}")

        # Closing the figure breaks the plt.show() blocking loop, allowing the process to exit naturally
        plt.close(self.fig)