import matplotlib.pyplot as plt
import numpy as np

# Set up the figure with 3 subplots side-by-side
# sharey=True ensures they all share the exact same 0 to 1 scale visually
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

titles = ["System 1", "System 2", "System 3"]

for i, ax in enumerate(axes):
    # Set the title for each subplot
    ax.set_title(titles[i], fontsize=14, fontweight='bold')

    # Draw custom axes with arrows
    ax.annotate('', xy=(1.02, 0), xytext=(0, 0), xycoords='axes fraction', 
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate('', xy=(0, 1.05), xytext=(0, 0), xycoords='axes fraction', 
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    
    # Set Y-axis range from 0 to 1
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 100) # Replace 100 with your actual max time/steps
    
    # Draw the red horizontal line for U_acc at 0.8
    ax.axhline(y=0.8, color='red', linestyle='--', lw=2)
    if i == 0:
        # Add the text label U_acc just above the red line
        # transform=ax.get_yaxis_transform() keeps the text glued to y=0.8 regardless of x-axis scale
        ax.text(-0.2, 0.8, r'$U_{acc}$', color='red', fontsize=14, transform=ax.get_yaxis_transform())

    # Draw the black horizontal line for U_delta at 0.1
    ax.axhline(y=0.1, color='black', linestyle='--', lw=2)
    if i == 0:
        # Add the text label U_delta just above the red line
        # transform=ax.get_yaxis_transform() keeps the text glued to y=0.1 regardless of x-axis scale
        ax.text(-0.2, 0.1, r'$U_{\delta}$', color='black', fontsize=14, transform=ax.get_yaxis_transform())

    if i == 0: 
        # Set the X position for the arrow (adjust based on your x-limits, e.g., 10 or 20)
        arrow_x = 5 
        
        # Draw double-headed vertical arrow to indicate the difference
        ax.annotate('', xy=(arrow_x, 0.8), xytext=(arrow_x, 0.1), xycoords='data', 
                    arrowprops=dict(arrowstyle="<->", color="red", lw=2))
        
        # Add the \Delta u label next to the arrow, vertically centered (y=0.45)
        # arrow_x + 2 adds a slight padding to the right so it doesn't overlap the arrow line
        ax.text(arrow_x + 2, 0.45, r'$\Delta u$', color='red', fontsize=14, va='center')
    
    # --- PLOT YOUR DATA HERE ---
    if i == 0:
        # Example data for System 1
        x_data = np.array([0, 10, 15, 15, 50, 52.5, 100])
        y_data = np.array([0.85, 0.85, 0.8, 0.1, 0.8, 0.9, 0.9])
        y_data = np.clip(y_data, 0.02, 0.85)  # Ensure the noisy recovery doesn't go below 0 or above 1

        t_start = x_data[2]  # Start time of the duration (e.g., when U_acc is reached)
        t_end = x_data[4]    # End time of the duration (e.g., when U_acc is reached)

        d_y_pos = 0.65 # Y-position for the D label
        d_x_pos = 28
        
    elif i == 1:
        # Example data for System 2
        x_data = np.array([0, 10, 15, 15, 90, 95, 100])
        y_data = np.array([0.85, 0.85, 0.8, 0.1, 0.8, 0.85, 0.85])
        y_data = np.clip(y_data, 0.02, 0.85)  # Ensure the noisy drop doesn't go below 0 or above 1

        t_start = x_data[2]  # Start time of the duration (e.g., when U_acc is reached)
        t_end = x_data[4]    # End time of the duration (e.g., when U_acc is reached)

        d_y_pos = 0.65 # Y-position for the D label
        d_x_pos = 45
        
    else:
        # Example data for System 3
        # Phase 1: Pre-disturbance (Flat at 0.9, touches 0.8 exactly at x=15)
        x1 = np.array([0, 10, 15])
        y1 = np.array([0.85, 0.85, 0.8])  # Touches U_acc at x=15
        y1 = np.clip(y1, 0.02, 0.85)  # Ensure the noisy drop doesn't go below 0 or above 1
   
        # Phase 2: Steep, noisy drop to U_delta=0.1 (x from 15.5 to 22)
        x2 = np.linspace(15.5, 17, 5)
        y2_clean = np.linspace(0.65, 0.1, 5)
        # Add random noise to the drop (standard deviation of 0.03)
        y2 = y2_clean + np.random.normal(0, 0.03, size=len(x2))
        y2 = np.clip(y2, 0.1, 0.8)  # Ensure the noisy drop doesn't go below 0 or above 1
        
        # Phase 3: Exponential, noisy recovery (x from 22.5 to 100)
        x3 = np.linspace(17, 40, 40)
        # Exponential formula: Asymptote is 0.9. It starts near 0.1 at x=22.5.
        # The '0.08' controls how fast it rises.
        y3_clean = np.exp(0.35 * (x3 - 40)) * 20 + 0.1
        print(y3_clean)
        # Add random noise to the recovery (standard deviation of 0.02)
        y3 = y3_clean + np.random.normal(0, 0.02, size=len(x3))
        # cut off when reaching 0.8 to avoid dropping below again due to noise
        y3 = np.clip(y3, 0.1, 0.9)  # Ensure the noisy recovery doesn't go below 0 or above 1
        y3 = np.where(y3 > 0.8, 0.85, y3)

        y4 = np.array([0.85, 0.85]) # Flat at 0.85 after recovery
        x4 = np.array([40, 100])
        
        # Combine the phases into single arrays
        x_data = np.concatenate([x1, x2, x3, x4])
        y_data = np.concatenate([y1, y2, y3, y4])
        
        # Clip data to ensure the noise doesn't push it below 0 or above 1
        y_data = np.clip(y_data, 0.02, 0.98)

        # --- Arrow Coordinates ---
        t_start = 15
        
        # Find the first point after the drop (x > 25) where the noisy curve crosses U_acc (0.8)
        recovery_indices = np.where((x_data > 25) & (y_data >= 0.8))[0]
        t_end = x_data[recovery_indices[0]]

        d_y_pos = 0.65 # Y-position for the D label
        d_x_pos = 23 # X-position for the D label

    # ---------------------------

    # ---> ADD GREEN ANNOTATIONS FOR DURATION <---
    arrow_y = 0.02 # Height to place the horizontal arrow


    # label with t subscript (i + 1)
    t_label = r'$t_{' + str(i + 1) + '}$'
    
    # Vertical green marker lines from U_acc down to the arrow
    ax.vlines(x=[t_start, t_end], ymin=arrow_y, ymax=0.8, color='green', linestyle=':', lw=1.5)
    
    # Horizontal double-headed arrow
    ax.annotate('', xy=(t_end, arrow_y), xytext=(t_start, arrow_y), xycoords='data', 
                arrowprops=dict(arrowstyle="<->", color="green", lw=2))
    
    # Text label (t1, t2, t3) centered above the arrow
    ax.text((t_start + t_end) / 2, arrow_y + 0.01, t_label, color='green', fontsize=14, ha='center', va='bottom')

    # Clean up aesthetics
    # ax.grid(True, which='both', linestyle='-', linewidth=1)
    ax.set_xlabel("t", fontsize=14)
    ax.set_xticks([])
    
    # Hide top and right spines for a cleaner academic look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---> SHADE THE AREA AND ADD D_u ANNOTATION <---
    # Shade area below 0.8 using fill_between
    # interpolate=True ensures the shading smoothly meets the 0.8 line exactly where they intersect
    ax.fill_between(x_data, y_data, 0.8, where=(y_data <= 0.8), 
                    color='blue', alpha=0.15, interpolate=True)
    
    # Add the D_{u, i} text label in the visual center of the shaded area
    # using rf-string to dynamically insert the system number {i+1}
    ax.text(d_x_pos, d_y_pos, rf'$D_{{u,{i+1}}}$', color='#1f77b4', 
            fontsize=18, fontweight='bold', ha='center', va='center')

    ax.plot(x_data, y_data, color='#1f77b4', lw=2)

    # X-TICK FOR t_delta
    ax.set_xticks([15])
    ax.set_xticklabels([r'$t_\delta$'], fontsize=14)

# Only add the Y-axis label to the far-left plot to avoid clutter
axes[0].set_ylabel("u(t)", fontsize=14, color='#1f77b4')

# Adjust layout so things don't overlap
plt.tight_layout()
plt.savefig('utility_comparison.pdf', bbox_inches='tight')