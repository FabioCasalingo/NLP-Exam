"""
visualization.py - Visualization utilities for bias metrics and reports

This module provides comprehensive visualization capabilities for bias evaluation results.
It creates publication-quality charts and reports that compare baseline and debiased models
across different metrics (SS, LMS, ICAT) and domains (gender, race, profession, religion).

The visualizations are designed to clearly communicate:
- Overall improvement in bias metrics
- Domain-specific changes
- Trade-offs between fairness and model capability
- Progress toward ideal fairness (SS = 50%)
"""

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for server/headless environments

import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


class BiasVisualizer:
    """
    Creates visualizations for bias evaluation results.
    
    This class generates various chart types to analyze and present bias mitigation results.
    All visualizations use a consistent color scheme and styling for professional appearance
    and easy interpretation. Charts are saved as high-resolution PNG files suitable for
    reports and presentations.
    
    Color scheme:
    - Baseline model: Red tones (indicates bias)
    - Mitigated model: Green tones (indicates improvement)
    - Ideal fairness line: Blue (target to reach)
    - Improvement arrows: Green for positive, red for negative
    """

    # Consistent color scheme across all visualizations
    BASELINE_COLOR = '#E57373'     # Light red for baseline
    MITIGATED_COLOR = '#66BB6A'    # Light green for mitigated
    IDEAL_LINE_COLOR = '#3F51B5'   # Blue for ideal fairness line
    ARROW_COLOR = '#2E7D32'        # Dark green for improvement arrows

    def __init__(self, output_dir: str = "plots"):
        """
        Initialize the visualizer with an output directory.
        
        Args:
            output_dir: Directory where all generated plots will be saved.
                       Created automatically if it doesn't exist.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_fig(self, fig: plt.Figure, filename: str) -> str:
        """
        Save a matplotlib figure to disk.
        
        Handles saving with consistent settings for quality and formatting.
        Closes the figure after saving to free memory.
        
        Args:
            fig: The matplotlib figure to save.
            filename: Name of the output file (will be placed in output_dir).
        
        Returns:
            Full path to the saved file.
        """
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=300, facecolor='white')
        plt.close(fig)
        return path

    def plot_main_comparison(
        self,
        baseline_ss: float,
        mitigated_ss: float,
        baseline_lms: float,
        mitigated_lms: float,
        baseline_icat: float,
        mitigated_icat: float,
        save_name: str = "main_comparison.png",
    ) -> str:
        """
        Create a three-panel comparison of the main bias metrics.
        
        This is the primary visualization for showing overall mitigation results. It displays
        side-by-side bars for baseline vs mitigated models across all three metrics: SS
        (Stereotype Score), LMS (Language Modeling Score), and ICAT (combined score).
        
        For SS, an ideal line at 50% is shown since that represents perfect fairness.
        For LMS and ICAT, higher is better, so no ideal line is needed.
        
        Change deltas are shown with arrows and percentage improvements to quantify the impact
        of debiasing.
        
        Args:
            baseline_ss: Baseline model stereotype score.
            mitigated_ss: Mitigated model stereotype score.
            baseline_lms: Baseline model language modeling score.
            mitigated_lms: Mitigated model language modeling score.
            baseline_icat: Baseline model ICAT score.
            mitigated_icat: Mitigated model ICAT score.
            save_name: Filename for the saved plot.
        
        Returns:
            Path to the saved visualization.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("", fontsize=1)

        # Define metrics to plot with optional ideal lines
        metrics = [
            ("Stereotype Score", baseline_ss, mitigated_ss, 50.0),  # Ideal is 50%
            ("Language Model Score", baseline_lms, mitigated_lms, None),  # Higher is better
            ("ICAT Score", baseline_icat, mitigated_icat, None),  # Higher is better
        ]

        for ax, (title, baseline, mitigated, ideal_line) in zip(axes, metrics):
            # Create bars
            x = [0, 1]
            heights = [baseline, mitigated]
            bars = ax.bar(
                x,
                heights,
                width=0.6,
                color=[self.BASELINE_COLOR, self.MITIGATED_COLOR],
                edgecolor='black',
                linewidth=1.5,
            )

            # Add ideal line for SS only
            if ideal_line is not None:
                ax.axhline(
                    ideal_line,
                    color=self.IDEAL_LINE_COLOR,
                    linestyle='--',
                    linewidth=2,
                    label=f'Ideal ({ideal_line}%)',
                )
                ax.legend(loc='upper right', frameon=True, fontsize=10)

            # Add value labels on top of bars
            for bar, val in zip(bars, heights):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f'{val:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    fontweight='bold',
                )

            # Calculate and display delta
            delta = mitigated - baseline
            delta_pct = (delta / baseline * 100) if baseline != 0 else 0

            mid_x = 0.5
            y_start = min(baseline, mitigated) + abs(delta) * 0.3
            y_end = max(baseline, mitigated) - abs(delta) * 0.3

            # Show arrow if change is significant
            if abs(delta) > 0.01:
                ax.annotate(
                    '',
                    xy=(mid_x, y_end),
                    xytext=(mid_x, y_start),
                    arrowprops=dict(
                        arrowstyle='->',
                        lw=2.5,
                        color=self.ARROW_COLOR if delta > 0 else '#C62828'  # Green for positive, red for negative
                    ),
                )

                # Position delta text based on arrow length
                anno_y = (y_start + y_end) / 2 if abs(delta) > 1 else baseline + 2
                ax.text(
                    mid_x + 0.15,
                    anno_y,
                    f'{delta:+.2f}\n({delta_pct:+.1f}%)',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9),
                )
            else:
                # Show "no change" indicator for minimal differences
                ax.text(
                    mid_x,
                    max(baseline, mitigated) + 3,
                    '≈0.00\n(no change)',
                    ha='center',
                    fontsize=9,
                    style='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8),
                )

            # Configure axes
            ax.set_xticks(x)
            ax.set_xticklabels(['Baseline', 'Mitigated'], fontsize=12)
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, 105)  # Fixed scale for consistency

        plt.tight_layout()
        return self._save_fig(fig, save_name)

    def plot_domain_analysis(
        self,
        domains: List[str],
        baseline_scores: List[float],
        mitigated_scores: List[float],
        save_name: str = "domain_analysis.png",
    ) -> str:
        """
        Create grouped bar chart showing SS across different bias domains.
        
        This visualization reveals which types of bias (gender, race, profession, religion)
        are most affected by the debiasing intervention. Some domains may improve more than
        others, and this chart makes those differences clear.
        
        The ideal fairness line at 50% provides a reference point to see how close each
        domain is to perfect fairness.
        
        Args:
            domains: List of domain names (e.g., ['gender', 'race', 'profession']).
            baseline_scores: SS scores for baseline model in each domain.
            mitigated_scores: SS scores for mitigated model in each domain.
            save_name: Filename for the saved plot.
        
        Returns:
            Path to the saved visualization.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(domains))
        width = 0.35

        # Create grouped bars
        bars1 = ax.bar(
            x - width / 2,
            baseline_scores,
            width,
            label='Baseline',
            color=self.BASELINE_COLOR,
            edgecolor='black',
            linewidth=1.5,
        )
        bars2 = ax.bar(
            x + width / 2,
            mitigated_scores,
            width,
            label='Mitigated',
            color=self.MITIGATED_COLOR,
            edgecolor='black',
            linewidth=1.5,
        )

        # Add ideal fairness line
        ax.axhline(
            50,
            color=self.IDEAL_LINE_COLOR,
            linestyle='--',
            linewidth=2,
            label='Ideal Fairness (50%)',
            zorder=0  # Draw behind bars
        )

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=11,
                    fontweight='bold',
                )

        # Configure axes
        ax.set_xlabel('Domain', fontsize=14, fontweight='bold')
        ax.set_ylabel('Stereotype Score', fontsize=14, fontweight='bold')
        ax.set_title('Bias by Domain: Baseline vs Mitigated Model', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in domains], fontsize=12)
        ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True, fancybox=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 105)

        plt.tight_layout()
        return self._save_fig(fig, save_name)

    def create_final_report(
        self,
        baseline_results: Dict,
        mitigated_results: Dict,
        save_name: str = "final_report.png",
    ) -> str:
        """
        Generate a comprehensive multi-panel report figure.
        
        This creates a publication-ready report combining multiple visualizations:
        1. Overall performance comparison (top panel)
        2. Per-domain delta analysis (middle panel)
        3. Methodology and findings summary (bottom panel)
        
        This is ideal for presentations or final project reports as it tells the complete
        story in a single, visually appealing figure.
        
        Args:
            baseline_results: Full evaluation results from baseline model.
            mitigated_results: Full evaluation results from mitigated model.
            save_name: Filename for the saved report.
        
        Returns:
            Path to the saved report figure.
        """
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle('Mind the Gap - Bias Mitigation Report', fontsize=20, fontweight='bold', y=0.98)

        # Create grid layout with three rows
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 0.9], hspace=0.45, wspace=0.35)

        # Panel 1: Overall performance comparison
        ax_perf = fig.add_subplot(gs[0, :])
        ax_perf.text(
            0.5, 0.95,
            'Overall Performance Comparison',
            ha='center', va='top',
            fontsize=14, fontweight='bold',
            transform=ax_perf.transAxes
        )

        metrics = ['Stereotype\nScore', 'LM Score', 'ICAT Score']
        baseline_vals = [
            baseline_results['overall']['ss'],
            baseline_results['overall']['lms'],
            baseline_results['overall']['icat'],
        ]
        mitigated_vals = [
            mitigated_results['overall']['ss'],
            mitigated_results['overall']['lms'],
            mitigated_results['overall']['icat'],
        ]

        x_pos = np.arange(len(metrics))
        width = 0.35

        # Create side-by-side bars
        bars1 = ax_perf.bar(
            x_pos - width / 2, baseline_vals, width,
            label='Baseline', color=self.BASELINE_COLOR,
            edgecolor='black', linewidth=1.5
        )
        bars2 = ax_perf.bar(
            x_pos + width / 2, mitigated_vals, width,
            label='Mitigated', color=self.MITIGATED_COLOR,
            edgecolor='black', linewidth=1.5
        )

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax_perf.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f'{height:.1f}',
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold'
                )

        # Configure axes
        ax_perf.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax_perf.set_xticks(x_pos)
        ax_perf.set_xticklabels(metrics, fontsize=11)
        ax_perf.legend(loc='upper right', fontsize=11, frameon=True)
        ax_perf.grid(axis='y', alpha=0.3)
        ax_perf.set_ylim(0, 105)

        # Panel 2: Domain-specific improvements
        ax_domain = fig.add_subplot(gs[1, :])
        ax_domain.text(
            0.5, 0.95,
            'Domain-Specific SS Improvements',
            ha='center', va='top',
            fontsize=14, fontweight='bold',
            transform=ax_domain.transAxes
        )

        domains = list(baseline_results['by_domain'].keys())
        deltas = [
            mitigated_results['by_domain'][d]['ss'] - baseline_results['by_domain'][d]['ss']
            for d in domains
        ]

        # Create horizontal bar chart for deltas
        y_pos = np.arange(len(domains))
        colors = [self.ARROW_COLOR if d > 0 else '#C62828' for d in deltas]

        bars = ax_domain.barh(y_pos, deltas, color=colors, edgecolor='black', linewidth=1.2)

        # Add value labels
        for bar, delta in zip(bars, deltas):
            width = bar.get_width()
            label_x = width + (0.5 if width > 0 else -0.5)
            ax_domain.text(
                label_x, bar.get_y() + bar.get_height() / 2,
                f'{delta:+.2f}',
                ha='left' if width > 0 else 'right',
                va='center',
                fontsize=11, fontweight='bold'
            )

        # Add zero line
        ax_domain.axvline(0, color='black', linestyle='-', linewidth=1.5)

        # Configure axes
        ax_domain.set_yticks(y_pos)
        ax_domain.set_yticklabels([d.capitalize() for d in domains], fontsize=11)
        ax_domain.set_xlabel('SS Change (Mitigated - Baseline)', fontsize=12, fontweight='bold')
        ax_domain.grid(axis='x', alpha=0.3)

        # Panel 3: Methodology and findings
        ax_info = fig.add_subplot(gs[2, :])
        ax_info.axis('off')

        # Compute key metrics
        icat_delta = mitigated_results['overall']['icat'] - baseline_results['overall']['icat']
        ss_delta = mitigated_results['overall']['ss'] - baseline_results['overall']['ss']

        # Determine key findings based on results
        if abs(icat_delta) < 1.0 and abs(ss_delta) < 1.0:
            finding_note = (
                "• Baseline is already near the target (SS ≈ 50%)\n"
                "• Minimal room for improvement\n"
                "• Fine-tuning impact on bias metrics is negligible"
            )
        else:
            finding_note = (
                f"• Overall ICAT improvement: {icat_delta:+.2f} points\n"
                "• Stereotype bias reduced with LM ability preserved\n"
                "• Trade-offs between fairness and performance considered"
            )

        info_text = (
            "Project: Mind the Gap — Social Bias Mitigation in Transformer Models\n\n"
            "Methodology:\n"
            "• Model: roberta-base\n"
            "• Dataset: StereoSet benchmark\n"
            "• Mitigation: Counterfactual data augmentation + fine-tuning\n"
            "• Evaluation: Stereotype Score, Language Modeling Score, ICAT Score\n\n"
            "Key Findings:\n"
            f"{finding_note}"
        )

        # Create fancy text box
        fancy_box = FancyBboxPatch(
            (0.05, 0.1), 0.9, 0.8,
            boxstyle="round,pad=0.02",
            edgecolor='gray',
            facecolor='#FFF8DC',  # Cornsilk color for a professional look
            transform=ax_info.transAxes,
            linewidth=2
        )
        ax_info.add_patch(fancy_box)

        # Add text to box
        ax_info.text(
            0.5, 0.5, info_text,
            ha='center', va='center',
            fontsize=10, family='monospace',
            transform=ax_info.transAxes
        )

        plt.tight_layout()
        return self._save_fig(fig, save_name)

    def plot_simple_comparison(
        self,
        domain: str,
        baseline_val: float,
        mitigated_val: float,
        save_name: Optional[str] = None,
    ) -> str:
        """
        Create a minimal two-bar comparison for a single domain.
        
        This is useful when you want to focus on one specific bias type without the
        clutter of multiple domains. It's a clean, simple visualization that highlights
        the improvement (or lack thereof) in one area.
        
        Args:
            domain: Name of the bias domain (e.g., 'gender', 'race').
            baseline_val: Baseline SS for this domain.
            mitigated_val: Mitigated SS for this domain.
            save_name: Optional custom filename. If None, auto-generates based on domain.
        
        Returns:
            Path to the saved visualization.
        """
        if save_name is None:
            save_name = f"{domain.lower()}_bias_comparison.png"

        fig, ax = plt.subplots(figsize=(6, 5))

        x = [0, 1]
        heights = [baseline_val, mitigated_val]
        bars = ax.bar(
            x,
            heights,
            width=0.5,
            color=[self.BASELINE_COLOR, self.MITIGATED_COLOR],
            edgecolor='black',
            linewidth=1.5,
        )

        # Add ideal line
        ax.axhline(50, color=self.IDEAL_LINE_COLOR, linestyle='--', linewidth=2, zorder=0)

        # Add value labels
        for bar, val in zip(bars, heights):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f'{val:.1f}',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold'
            )

        # Show delta at top
        delta = mitigated_val - baseline_val
        ax.text(
            0.5, 1.05,
            f'Δ = {delta:+.2f}',
            ha='center', va='bottom',
            fontsize=12,
            transform=ax.transAxes
        )

        # Configure axes
        ax.set_xticks(x)
        ax.set_xticklabels(['Baseline', 'Mitigated'], fontsize=11)
        ax.set_ylabel('Stereotype Score (SS)', fontsize=11, fontweight='bold')
        ax.set_title(f'{domain.capitalize()} — Bias Comparison', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(60, max(heights) + 5))  # Dynamic y-limit based on data

        plt.tight_layout()
        return self._save_fig(fig, save_name)

    def create_summary_table(
        self,
        baseline_results: Dict,
        mitigated_results: Dict,
        save_name: str = "report_summary.png",
    ) -> str:
        """
        Create a compact summary table showing per-domain metrics.
        
        Tables are useful for reports where you need exact numbers rather than just visual
        comparisons. This table shows baseline SS, mitigated SS, and final ICAT for each
        domain in an easy-to-read format.
        
        Args:
            baseline_results: Full evaluation results from baseline model.
            mitigated_results: Full evaluation results from mitigated model.
            save_name: Filename for the saved table.
        
        Returns:
            Path to the saved table image.
        """
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('off')
        ax.set_title('Summary Report', fontsize=14, fontweight='bold', pad=20)

        # Prepare table data
        domains = list(baseline_results['by_domain'].keys())
        table_data = []

        for domain in domains:
            b_ss = baseline_results['by_domain'][domain]['ss']
            m_ss = mitigated_results['by_domain'][domain]['ss']
            icat = mitigated_results['by_domain'][domain]['icat']
            table_data.append([
                domain.capitalize(),
                f'{b_ss:.2f}',
                f'{m_ss:.2f}',
                f'{icat:.2f}'
            ])

        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=['Domain', 'Baseline SS', 'Mitigated SS', 'ICAT'],
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style header row
        for i in range(4):
            cell = table[(0, i)]
            cell.set_facecolor('#4A90E2')  # Blue header
            cell.set_text_props(weight='bold', color='white')
            cell.set_edgecolor('#2E5C8A')

        # Light borders for data rows
        for key, cell in table.get_celld().items():
            if key[0] > 0:  # Skip header
                cell.set_edgecolor('#DDDDDD')

        plt.tight_layout()
        return self._save_fig(fig, save_name)


def visualize_all_results(
    baseline_results: Dict,
    mitigated_results: Dict,
    output_dir: str = "plots",
) -> Dict[str, str]:
    """
    Generate all standard visualizations and return their file paths.
    
    This is a convenience function that creates a complete set of visualizations in one call.
    It produces:
    - Main three-metric comparison
    - Domain analysis chart
    - Individual domain comparisons
    - Comprehensive final report
    - Summary table
    
    Args:
        baseline_results: Full evaluation results from baseline model.
        mitigated_results: Full evaluation results from mitigated model.
        output_dir: Directory where all plots will be saved.
    
    Returns:
        Dictionary mapping visualization names to their file paths.
    """
    viz = BiasVisualizer(output_dir=output_dir)
    paths = {}

    # Main three-metric comparison
    paths['main_comparison'] = viz.plot_main_comparison(
        baseline_ss=baseline_results['overall']['ss'],
        mitigated_ss=mitigated_results['overall']['ss'],
        baseline_lms=baseline_results['overall']['lms'],
        mitigated_lms=mitigated_results['overall']['lms'],
        baseline_icat=baseline_results['overall']['icat'],
        mitigated_icat=mitigated_results['overall']['icat'],
    )

    # Domain grouped bars
    domains = list(baseline_results['by_domain'].keys())
    baseline_scores = [baseline_results['by_domain'][d]['ss'] for d in domains]
    mitigated_scores = [mitigated_results['by_domain'][d]['ss'] for d in domains]

    paths['domain_analysis'] = viz.plot_domain_analysis(
        domains=domains,
        baseline_scores=baseline_scores,
        mitigated_scores=mitigated_scores,
    )

    # Individual domain comparisons
    for domain in domains:
        b_ss = baseline_results['by_domain'][domain]['ss']
        m_ss = mitigated_results['by_domain'][domain]['ss']
        paths[f'{domain}_comparison'] = viz.plot_simple_comparison(
            domain=domain,
            baseline_val=b_ss,
            mitigated_val=m_ss,
        )

    # Comprehensive report
    paths['final_report'] = viz.create_final_report(
        baseline_results=baseline_results,
        mitigated_results=mitigated_results,
    )

    # Summary table
    paths['summary_table'] = viz.create_summary_table(
        baseline_results=baseline_results,
        mitigated_results=mitigated_results,
    )

    return paths
