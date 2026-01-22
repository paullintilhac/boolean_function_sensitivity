#!/usr/bin/env python3
"""
Analyze perturbation hessian results from perturbation_experiment.py

This script analyzes perturbation_hessian_results.csv to visualize how the 
90th-percentile perturbation ratio depends on degree, width, and T.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
print("Loading perturbation_hessian_results.csv...")
df = pd.read_csv("perturbation_hessian_results.csv")

print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"Unique sigma values: {sorted(df['sigma'].unique())}")

# Filter out sigma == 0.01 (exclude largest perturbations)
df = df[df['sigma'] != 0.01].copy()
print(f"After filtering sigma=0.01: {len(df)} rows")

# Calculate perturbation ratio: |trace_delta| / |trace_before|
# Use absolute values to handle negative traces
df['perturbation_ratio'] = np.abs(df['trace_delta']) / np.abs(df['trace_before'])

# Handle any infinite or NaN values that might result from division
df = df[~np.isinf(df['perturbation_ratio']) & ~np.isnan(df['perturbation_ratio'])].copy()

print(f"After removing inf/NaN: {len(df)} rows")
print(f"Perturbation ratio stats:")
print(df['perturbation_ratio'].describe())

# Calculate 90th percentile perturbation ratio for each (deg, width, T, sigma) combination
print("\nCalculating 90th percentile perturbation ratios...")
percentile_data = df.groupby(['deg', 'width', 'T', 'sigma'])['perturbation_ratio'].quantile(0.90).reset_index()
percentile_data = percentile_data.rename(columns={'perturbation_ratio': 'p90_perturbation_ratio'})

print(f"Calculated percentiles for {len(percentile_data)} unique combinations")
print(f"Unique values:")
print(f"  Degrees: {sorted(percentile_data['deg'].unique())}")
print(f"  Widths: {sorted(percentile_data['width'].unique())}")
print(f"  T values: {sorted(percentile_data['T'].unique())}")
print(f"  Sigma values: {sorted(percentile_data['sigma'].unique())}")

# Define color scheme for widths (matching existing plots)
width_colors = {
    1: 'red',
    7: 'brown',
    14: 'green',
    20: 'purple'
}

# Define line styles and markers for sigma values
sigma_styles = {
    1e-10: {'dash': 'solid', 'marker': 'circle'},
    1e-08: {'dash': 'dash', 'marker': 'square'},
    1e-06: {'dash': 'dashdot', 'marker': 'diamond'},
    0.0001: {'dash': 'dot', 'marker': 'triangle-up'}
}

# Get unique values for plotting
unique_widths = sorted(percentile_data['width'].unique())
unique_sigmas = sorted(percentile_data['sigma'].unique())

# ============================================================================
# Plot 1A: Perturbation vs Degree (Subplots by Sigma)
# ============================================================================
print("\nCreating Plot 1A: Perturbation vs Degree (Subplots by Sigma)...")

num_sigmas = len(unique_sigmas)
num_cols = 2
num_rows = (num_sigmas + num_cols - 1) // num_cols

fig1a = make_subplots(
    rows=num_rows, cols=num_cols,
    subplot_titles=[f"σ = {sigma}" for sigma in unique_sigmas],
    horizontal_spacing=0.15,
    vertical_spacing=0.15
)

for idx, sigma in enumerate(unique_sigmas):
    row = idx // num_cols + 1
    col = idx % num_cols + 1
    
    sigma_data = percentile_data[percentile_data['sigma'] == sigma]
    
    for width in unique_widths:
        subset = sigma_data[sigma_data['width'] == width].sort_values('deg')
        if len(subset) > 0:
            fig1a.add_trace(
                go.Scatter(
                    x=subset['deg'],
                    y=subset['p90_perturbation_ratio'],
                    mode='lines+markers',
                    name=f"Width={width}",
                    line=dict(color=width_colors.get(width, 'blue'), width=2),
                    marker=dict(size=6),
                    showlegend=(idx == 0)  # Only show legend for first subplot
                ),
                row=row, col=col
            )

fig1a.update_layout(
    title="90th Percentile Perturbation Ratio vs Degree (Faceted by Sigma)",
    height=300 * num_rows,
    width=1200,
    margin=dict(t=80, b=50, l=50, r=50)
)

# Update axes
for row_idx in range(1, num_rows + 1):
    for col_idx in range(1, num_cols + 1):
        fig1a.update_xaxes(title_text="Degree", row=row_idx, col=col_idx)
        fig1a.update_yaxes(title_text="90th Percentile Perturbation Ratio", row=row_idx, col=col_idx)

fig1a.show()
print("Plot 1A displayed")

# ============================================================================
# Plot 1B: Perturbation vs Degree (Line Styles by Sigma)
# ============================================================================
print("\nCreating Plot 1B: Perturbation vs Degree (Line Styles by Sigma)...")

fig1b = go.Figure()

for width in unique_widths:
    width_data = percentile_data[percentile_data['width'] == width]
    
    for sigma in unique_sigmas:
        subset = width_data[width_data['sigma'] == sigma].sort_values('deg')
        if len(subset) > 0:
            style = sigma_styles.get(sigma, {'dash': 'solid', 'marker': 'circle'})
            fig1b.add_trace(
                go.Scatter(
                    x=subset['deg'],
                    y=subset['p90_perturbation_ratio'],
                    mode='lines+markers',
                    name=f"Width={width}, σ={sigma}",
                    line=dict(
                        color=width_colors.get(width, 'blue'),
                        dash=style['dash'],
                        width=2
                    ),
                    marker=dict(symbol=style['marker'], size=6)
                )
            )

fig1b.update_layout(
    title="90th Percentile Perturbation Ratio vs Degree (Line Styles by Sigma)",
    xaxis_title="Degree",
    yaxis_title="90th Percentile Perturbation Ratio",
    height=600,
    width=1000,
    margin=dict(t=80, b=50, l=50, r=50),
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
)

fig1b.show()
print("Plot 1B displayed")

# ============================================================================
# Plot 2A: Perturbation vs T (Subplots by Sigma)
# ============================================================================
print("\nCreating Plot 2A: Perturbation vs T (Subplots by Sigma)...")

fig2a = make_subplots(
    rows=num_rows, cols=num_cols,
    subplot_titles=[f"σ = {sigma}" for sigma in unique_sigmas],
    horizontal_spacing=0.15,
    vertical_spacing=0.15
)

for idx, sigma in enumerate(unique_sigmas):
    row = idx // num_cols + 1
    col = idx % num_cols + 1
    
    sigma_data = percentile_data[percentile_data['sigma'] == sigma]
    
    for width in unique_widths:
        subset = sigma_data[sigma_data['width'] == width].sort_values('T')
        if len(subset) > 0:
            fig2a.add_trace(
                go.Scatter(
                    x=subset['T'],
                    y=subset['p90_perturbation_ratio'],
                    mode='lines+markers',
                    name=f"Width={width}",
                    line=dict(color=width_colors.get(width, 'blue'), width=2),
                    marker=dict(size=6),
                    showlegend=(idx == 0)  # Only show legend for first subplot
                ),
                row=row, col=col
            )

fig2a.update_layout(
    title="90th Percentile Perturbation Ratio vs T (Faceted by Sigma)",
    height=300 * num_rows,
    width=1200,
    margin=dict(t=80, b=50, l=50, r=50)
)

# Update axes
for row_idx in range(1, num_rows + 1):
    for col_idx in range(1, num_cols + 1):
        fig2a.update_xaxes(title_text="T (Sequence Length)", row=row_idx, col=col_idx)
        fig2a.update_yaxes(title_text="90th Percentile Perturbation Ratio", row=row_idx, col=col_idx)

fig2a.show()
print("Plot 2A displayed")

# ============================================================================
# Plot 2B: Perturbation vs T (Line Styles by Sigma)
# ============================================================================
print("\nCreating Plot 2B: Perturbation vs T (Line Styles by Sigma)...")

fig2b = go.Figure()

for width in unique_widths:
    width_data = percentile_data[percentile_data['width'] == width]
    
    for sigma in unique_sigmas:
        subset = width_data[width_data['sigma'] == sigma].sort_values('T')
        if len(subset) > 0:
            style = sigma_styles.get(sigma, {'dash': 'solid', 'marker': 'circle'})
            fig2b.add_trace(
                go.Scatter(
                    x=subset['T'],
                    y=subset['p90_perturbation_ratio'],
                    mode='lines+markers',
                    name=f"Width={width}, σ={sigma}",
                    line=dict(
                        color=width_colors.get(width, 'blue'),
                        dash=style['dash'],
                        width=2
                    ),
                    marker=dict(symbol=style['marker'], size=6)
                )
            )

fig2b.update_layout(
    title="90th Percentile Perturbation Ratio vs T (Line Styles by Sigma)",
    xaxis_title="T (Sequence Length)",
    yaxis_title="90th Percentile Perturbation Ratio",
    height=600,
    width=1000,
    margin=dict(t=80, b=50, l=50, r=50),
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
)

fig2b.show()
print("Plot 2B displayed")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nOverall statistics for 90th percentile perturbation ratios:")
print(percentile_data['p90_perturbation_ratio'].describe())

print("\nMaximum perturbation ratios by configuration:")
max_pert = percentile_data.loc[percentile_data['p90_perturbation_ratio'].idxmax()]
print(f"  Max ratio: {max_pert['p90_perturbation_ratio']:.6f}")
print(f"  Configuration: deg={max_pert['deg']}, width={max_pert['width']}, "
      f"T={max_pert['T']}, sigma={max_pert['sigma']}")

print("\nAverage perturbation ratios by width:")
width_avg = percentile_data.groupby('width')['p90_perturbation_ratio'].mean().sort_values(ascending=False)
for width, avg in width_avg.items():
    print(f"  Width {width}: {avg:.6f}")

print("\nAverage perturbation ratios by degree:")
deg_avg = percentile_data.groupby('deg')['p90_perturbation_ratio'].mean().sort_values(ascending=False)
for deg, avg in deg_avg.items():
    print(f"  Degree {deg}: {avg:.6f}")

print("\nAverage perturbation ratios by sigma:")
sigma_avg = percentile_data.groupby('sigma')['p90_perturbation_ratio'].mean().sort_values(ascending=False)
for sigma, avg in sigma_avg.items():
    print(f"  σ={sigma}: {avg:.6f}")

print("\nAverage perturbation ratios by T:")
T_avg = percentile_data.groupby('T')['p90_perturbation_ratio'].mean().sort_values(ascending=False)
for T, avg in T_avg.items():
    print(f"  T={T}: {avg:.6f}")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)

# Optional: Save plots
save_plots = False  # Set to True to save plots
if save_plots:
    print("\nSaving plots...")
    fig1a.write_image("plots/perturbation_vs_degree_subplots.png", width=1200, height=300*num_rows)
    fig1b.write_image("plots/perturbation_vs_degree_linestyles.png", width=1000, height=600)
    fig2a.write_image("plots/perturbation_vs_T_subplots.png", width=1200, height=300*num_rows)
    fig2b.write_image("plots/perturbation_vs_T_linestyles.png", width=1000, height=600)
    print("Plots saved to plots/ directory")
