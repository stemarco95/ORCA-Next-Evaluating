import matplotlib.pyplot as plt
import numpy as np

# --- Setup Figure & Axes ---
fig, ax = plt.subplots(figsize=(12, 6))

# Hide standard spines and ticks
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Graph paper grid
ax.grid(True, which='both', linestyle='-', linewidth=1)

# X and Y limits (Inverting X-axis so 0 is on the right)
ax.set_xlim(48, -10)
ax.set_ylim(-1.5, 1.5)

# --- Draw Custom Axes ---
# X-axis line (double-ended arrow)
ax.annotate('', xy=(-8, 0), xytext=(48, 0), xycoords='data',
            arrowprops=dict(arrowstyle="<-", color="black", lw=1.5))

# Y-axis line (at x = 45)
ax.annotate('', xy=(45, 1.4), xytext=(45, -1.4), xycoords='data',
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

# --- Ticks and Labels ---
# X-axis
ax.vlines(x=[30, 10], ymin=-0.05, ymax=0.05, color='black', lw=1.5)
ax.text(30, -0.15, '30', fontsize=14, ha='center')
ax.text(10, -0.15, '10', fontsize=14, ha='center')
ax.text(0, -0.15, '0', fontsize=14, ha='center')
ax.text(-6, -0.15, r'$d$ in [m]', fontsize=14, ha='center')

# X-axis Top Labels (stacked)
ax.text(30, 0.1, '$d_{target}$\n$+ d_{norm}$', fontsize=14, ha='center', va='bottom')
ax.text(10, 0.1, '$d_{target}$', fontsize=14, ha='center', va='bottom')

# Y-axis
ax.hlines(y=1.0, xmin=45.5, xmax=44.5, color='black', lw=1.5)
ax.hlines(y=-1.0, xmin=45.5, xmax=44.5, color='black', lw=1.5)

ax.text(46, 1.0, '0.5', fontsize=14, ha='right', va='center')
ax.text(46, -1.0, '-0.5', fontsize=14, ha='right', va='center')
ax.text(46, 0.1, '0.0', fontsize=14, ha='right', va='center')


ax.text(46, 1.35, r'$r$', fontsize=16, ha='right', va='center')

# --- Define the Mathematical Curve ---
# 1. From d=0 to d=10 (Concave down, peaking at 1)
d1 = np.linspace(0, 10, 50)
y1 = (1 - (10 - d1) / 10)**2

# 2. From d=10 to d=30 (Concave up, decaying to 0, slope matches the linear part)
d2 = np.linspace(10, 30, 100)
y2 = (1 - (d2 - 10) / 20)**2

# 3. From d=30 onwards (Linear decay)
d3 = np.linspace(30, 48, 50)
y3 = 1 - (d3 - 10) / 20 

# Combine arrays
d_curve = np.concatenate([d1, d2[1:], d3[1:]])
y_curve = np.concatenate([y1, y2[1:], y3[1:]])

# Plot main curve
ax.plot(d_curve, y_curve, color='#0f8554', lw=2)

# --- Crash Penalty Drop ---
# Vertical line dropping at 0
ax.annotate('', xy=(0, -1.3), xytext=(0, 0), xycoords='data',
            arrowprops=dict(arrowstyle="->", color="#0f8554", lw=2))

# Text and thin curved arrow pointing to the crash penalty tip
ax.annotate('crash penalty\n$-3,0$',
            xy=(-0.4, -1.25), xycoords='data', 
            xytext=(-4.5, -0.9), textcoords='data',
            arrowprops=dict(arrowstyle="->", color="#0f8554", lw=1.2, connectionstyle="arc3,rad=0.2"),
            color='#0f8554', fontsize=14, ha='center', va='center')

plt.tight_layout()
plt.savefig('reward_function.pdf', bbox_inches='tight')