import matplotlib.pyplot as plt
import numpy as np

# Define the utility data points
u_adapt = np.array([0.65, 0.85, 0.25, 0.65, 0.75])
x_adapt = np.array([0, 1, 2, 3, 4])

# Map the adaptation steps to the SuOC timescale (1 adapt step = 3 SuOC steps)
u_suoc = np.repeat(u_adapt, 3)
x_suoc = np.arange(15)

# Set up the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#ffffff") # white background for graph paper

# Define a function to style the axes to look like the sketch
def format_axis(ax, x_max):
    # Hide standard spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Draw custom axes with arrows
    ax.annotate('', xy=(1.02, 0), xytext=(0, 0), xycoords='axes fraction', 
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate('', xy=(0, 1.05), xytext=(0, 0), xycoords='axes fraction', 
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    
    # Grid styling (Graph paper look)
    # ax.grid(True, which='both', color='#e0e0d0', linestyle='-', linewidth=1)
    
    # Axis limits
    ax.set_ylim(0, 1)
    ax.set_xlim(0, x_max)
    # ax.set_yticks([]) # Hide Y-axis ticks

# --- Left Plot (Adaptation Timescale) ---
ax1.plot(x_adapt, u_adapt, color='#1f77b4', linestyle=':', marker='x', markersize=6, lw=2)
format_axis(ax1, 4.5)
ax1.set_xticks(x_adapt)

# Labels for Left Plot
ax1.text(-0.1, 1.05, 'u', fontsize=16, fontstyle='italic', transform=ax1.transAxes)
ax1.text(0.95, -0.1, r'$j_{adapt}$', fontsize=16, transform=ax1.transAxes)

# --- Right Plot (SuOC Timescale) ---
ax2.plot(x_suoc, u_suoc, color='#0f8554', linestyle=':', marker='x', markersize=6, lw=2)
format_axis(ax2, 14.5)

# Primary X-ticks (0, 3, 6, 9, 12)
ax2.set_xticks(np.arange(15))
xtick_labels = ['0' if i == 0 else str(i) if i % 3 == 0 else '' for i in range(15)]
ax2.set_xticklabels(xtick_labels, fontsize=12)

# Labels for Right Plot
ax2.text(-0.1, 1.05, 'u', fontsize=16, fontstyle='italic', transform=ax2.transAxes)
ax2.text(0.95, -0.1, r'$j_{SuOC}$', fontsize=16, transform=ax2.transAxes)

# Adjust layout to prevent clipping
plt.tight_layout()
plt.savefig('utility_timescales.pdf', bbox_inches='tight')