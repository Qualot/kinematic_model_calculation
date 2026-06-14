import argparse
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # Required for setting tick intervals
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# --- Font and output configurations for Type 3 font prevention ---
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Times New Roman"

# --- Font Size Customization ---
title_fontsize = 14
label_fontsize = 12       # For axis labels (phi, theta) and colorbar label
tick_fontsize = 24        # For axis ticks and colorbar ticks
contour_fontsize = 24      # For numbers on the contour lines

# --- Command line argument parsing ---
parser = argparse.ArgumentParser(
    description="Plot a contour-overlayed heatmap from a CSV file."
)
parser.add_argument(
    "-c",
    "--csv",
    type=str,
    required=True,
    help="Path to the input CSV file containing data",
)
args = parser.parse_args()

# --- Validate file existence ---
if not os.path.exists(args.csv):
    print(f"Error: The file '{args.csv}' does not exist.")
    exit(1)

# --- Load CSV data ---
df = pd.read_csv(args.csv)

# --- Extract and grid data ---
# Requested: x-axis -> phi, y-axis -> theta
x = df["phi"].values
y = df["theta"].values
z = df["curved"].values

# Create a 2D grid for contour mapping (100x100 resolution)
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
X, Y = np.meshgrid(xi, yi)

# Perform linear interpolation to grid the scatter data
Z = griddata((x, y), z, (X, Y), method="linear")

# --- Define custom contour levels ---
# Start at 86.0, increment by 10, up to the maximum value of the data
contour_start = 66.0
contour_step = 5.0
contour_levels = np.arange(contour_start, z.max() + contour_step, contour_step)

# --- Define Colorbar Range and Ticks ---
# Requested: Range from 65 to 100, Step by 5
cbar_min = 65.0
cbar_max = 100.0
cbar_ticks = np.arange(cbar_min, cbar_max + 1.0, 5.0)

# --- Plotting ---
plt.figure(figsize=(8, 6.5))

# 1. Draw the heatmap (fixed range from 65 to 100)
heatmap = plt.pcolormesh(X, Y, Z, cmap="viridis", shading="auto", vmin=cbar_min, vmax=cbar_max)

# 2. Overlay contour lines with explicit custom levels
contours = plt.contour(
    X, Y, Z, levels=contour_levels, colors="white", linewidths=0.8
)

# 3. Add labels to the contour lines (Uses contour_fontsize)
plt.clabel(contours, inline=True, fmt="%.1f", fontsize=contour_fontsize, colors="white")

# 4. Add a colorbar with specific ticks and range
cbar = plt.colorbar(heatmap, ticks=cbar_ticks)
#cbar.set_label("curved value", rotation=270, labelpad=15, fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)  # Changes colorbar tick font size

# 5. Set axis labels and title with specific font sizes
#plt.xlabel(r"$\phi$ (phi)", fontsize=label_fontsize)
#plt.ylabel(r"$\theta$ (theta)", fontsize=label_fontsize)
#plt.title("Heatmap with Contours (curved)", fontsize=title_fontsize)

# 6. Customize ticks (Intervals, Limits, and Font Sizes)
ax = plt.gca()  # Get current axes to modify tick parameters
ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

# Set x-axis range explicitly from -0.8 to 0.8
ax.set_xlim(-0.8, 0.8)

# Set x-axis tick interval to 0.2
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))

# 7. Add dotted reference lines (phi = pi/8, theta = pi/2)
target_phi = - np.pi / 8
target_theta = np.pi / 2

# Vertical dotted line at phi = pi/8
ax.axvline(x=target_phi, color="black", linestyle="--", linewidth=2.0, alpha=0.7)

# Horizontal dotted line at theta = pi/2
ax.axhline(y=target_theta, color="black", linestyle="--", linewidth=2.0, alpha=0.7)

# Adjust layout automatically to prevent clipping
plt.tight_layout()

# Optional: Save the figure as a PDF to preserve vector format and fonts
# plt.savefig('heatmap_curved.pdf', bbox_inches='tight')

# Display the plot
plt.show()