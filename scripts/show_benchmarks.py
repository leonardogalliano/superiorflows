"""Benchmark visualization with grouped panels by source file.

Groups benchmarks by their origin (test_flow.py, test_particles.py, diffrax features)
for better readability.
"""

import glob
import json
import os
import re

import dateutil.parser
import matplotlib.pyplot as plt

# Define benchmark categories based on source file and type
CATEGORIES = {
    "Flow (test_flow.py)": [
        "test_flow_performance",
        "test_batched_performance",
        "test_ad_performance",
        "test_sample_n_performance",
        "test_sample_n_and_log_prob_performance",
        "test_hutchinson_performance",
    ],
    "Particles (test_particles.py)": [
        "test_particle_flow_performance",
        "test_particle_flow_batched_performance",
        "test_particle_ad_performance",
        "test_hutchinson_particles_performance",
    ],
    "ODE Solvers": [
        "test_solver_benchmark[Tsit5-solver0]",
        "test_solver_benchmark[Dopri5-solver1]",
        "test_solver_benchmark[Dopri8-solver2]",
    ],
    "Dimension Scaling": [
        "test_dimension_scaling_benchmark[2]",
        "test_dimension_scaling_benchmark[10]",
        "test_dimension_scaling_benchmark[20]",
        "test_dimension_scaling_benchmark[50]",
        "test_hutchinson_dimension_scaling_benchmark[2]",
        "test_hutchinson_dimension_scaling_benchmark[10]",
        "test_hutchinson_dimension_scaling_benchmark[20]",
        "test_hutchinson_dimension_scaling_benchmark[50]",
    ],
    "Batch Scaling": [
        "test_batch_size_scaling[1]",
        "test_batch_size_scaling[10]",
        "test_batch_size_scaling[100]",
        "test_hutchinson_batch_size_scaling[1]",
        "test_hutchinson_batch_size_scaling[10]",
        "test_hutchinson_batch_size_scaling[100]",
    ],
    "Hutchinson Samples": [
        "test_hutchinson_samples_scaling[1]",
        "test_hutchinson_samples_scaling[5]",
        "test_hutchinson_samples_scaling[10]",
        "test_hutchinson_samples_scaling[20]",
    ],
}


def shorten_name(name: str) -> str:
    """Shorten benchmark name for display."""
    name = re.sub(r"^test_", "", name)
    name = re.sub(r"_benchmark$", "", name)
    name = re.sub(r"_performance$", "", name)
    # Keep parameters in brackets but clean up the rest
    name = name.replace("_", " ")
    return name.title()


def main():
    benchmarks_dir = ".benchmarks/Darwin-CPython-3.12-64bit"
    json_files = sorted(glob.glob(os.path.join(benchmarks_dir, "*.json")))

    if not json_files:
        print(f"No benchmark files found in {benchmarks_dir}")
        return

    print(f"Found {len(json_files)} benchmark files.")

    data = {}

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                content = json.load(f)
            timestamp_str = content.get("datetime")
            if not timestamp_str:
                continue

            timestamp = dateutil.parser.isoparse(timestamp_str)

            for bench in content.get("benchmarks", []):
                name = bench.get("name")
                stats = bench.get("stats", {})
                mean = stats.get("mean")
                stddev = stats.get("stddev")

                if name and mean is not None:
                    if name not in data:
                        data[name] = {"dates": [], "means": [], "stddevs": []}

                    data[name]["dates"].append(timestamp)
                    data[name]["means"].append(mean)
                    data[name]["stddevs"].append(stddev if stddev is not None else 0)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    # Create global date index
    all_dates = set()
    for values in data.values():
        all_dates.update(values["dates"])
    sorted_unique_dates = sorted(list(all_dates))
    date_to_index = {d: i for i, d in enumerate(sorted_unique_dates)}

    # Filter categories to only include benchmarks that exist
    active_categories = {}
    for cat_name, cat_benchmarks in CATEGORIES.items():
        existing = [b for b in cat_benchmarks if b in data]
        if existing:
            active_categories[cat_name] = existing

    # Check for uncategorized benchmarks
    all_categorized = set()
    for benchmarks in CATEGORIES.values():
        all_categorized.update(benchmarks)
    uncategorized = [b for b in data.keys() if b not in all_categorized]
    if uncategorized:
        print(f"Warning: {len(uncategorized)} uncategorized benchmarks:")
        for b in uncategorized:
            print(f"  - {b}")

    n_categories = len(active_categories)
    if n_categories == 0:
        print("No benchmarks to plot.")
        return

    # Create figure with subplots (3 columns)
    n_cols = min(3, n_categories)
    n_rows = (n_categories + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)
    fig.suptitle("Benchmark Performance Over Time", fontsize=14, fontweight="bold", y=1.02)

    # Color palettes - distinct for each category
    # Each palette has enough colors and no greys
    palettes = {
        "Flow (test_flow.py)": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        "Particles (test_particles.py)": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"],
        "ODE Solvers": ["#66c2a5", "#fc8d62", "#8da0cb"],
        "Dimension Scaling": [
            "#1b9e77",
            "#d95f02",
            "#7570b3",
            "#e7298a",  # Exact: teal, orange, purple, pink
            "#66a61e",
            "#e6ab02",
            "#a6761d",
            "#666666",  # Hutchinson: green, gold, brown, grey
        ],
        "Batch Scaling": [
            "#e41a1c",
            "#377eb8",
            "#4daf4a",  # Exact: red, blue, green
            "#ff7f00",
            "#984ea3",
            "#a65628",  # Hutchinson: orange, purple, brown
        ],
        "Hutchinson Samples": ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072"],
    }

    # Default fallback palette
    default_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (category, names) in enumerate(active_categories.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        colors = palettes.get(category, default_palette)

        for i, name in enumerate(names):
            if name not in data:
                continue
            values = data[name]
            sorted_pairs = sorted(zip(values["dates"], values["means"], values["stddevs"]))

            x_indices = [date_to_index[p[0]] for p in sorted_pairs]
            means = [p[1] * 1000 for p in sorted_pairs]  # Convert to ms
            stddevs = [p[2] * 1000 for p in sorted_pairs]

            short_name = shorten_name(name)
            ax.errorbar(
                x_indices,
                means,
                yerr=stddevs,
                marker="o",
                markersize=4,
                label=short_name,
                color=colors[i % len(colors)],
                capsize=2,
                linewidth=1.5,
            )

        ax.set_title(category, fontsize=11, fontweight="bold")
        ax.set_xlabel("Benchmark Run", fontsize=9)
        ax.set_ylabel("Time (ms)", fontsize=9)
        ax.set_yscale("log")
        # Consistent x-axis limits across all panels
        ax.set_xlim(-0.5, len(sorted_unique_dates) - 0.5)
        # Legend on the left side, outside the plot
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
        ax.grid(True, which="both", ls="-", alpha=0.3)

        # X-axis ticks
        if len(sorted_unique_dates) <= 10:
            ax.set_xticks(range(len(sorted_unique_dates)))
            ax.set_xticklabels(
                [d.strftime("%m-%d %H:%M") for d in sorted_unique_dates], rotation=45, ha="right", fontsize=7
            )
        else:
            step = max(1, len(sorted_unique_dates) // 5)
            ticks = list(range(0, len(sorted_unique_dates), step))
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [sorted_unique_dates[i].strftime("%m-%d") for i in ticks], rotation=45, ha="right", fontsize=7
            )

    # Hide unused subplots
    for idx in range(len(active_categories), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()

    output_file = ".benchmarks/benchmark_plot.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Benchmark plot saved to {output_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print("LATEST BENCHMARK SUMMARY (sorted by time)")
    print("=" * 70)

    latest = []
    for name, values in data.items():
        sorted_by_date = sorted(zip(values["dates"], values["means"], values["stddevs"]))
        if sorted_by_date:
            _, mean, std = sorted_by_date[-1]
            latest.append((name, mean * 1000, std * 1000))

    latest.sort(key=lambda x: x[1])

    print(f"{'Benchmark':<55} {'Time (ms)':>10} {'Std':>8}")
    print("-" * 75)
    for name, mean, std in latest:
        short = shorten_name(name)[:53]
        print(f"{short:<55} {mean:>10.2f} {std:>8.2f}")


if __name__ == "__main__":
    main()
