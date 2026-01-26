"""Benchmark visualization with dynamic grouping and improved comparison.

This script parses pytest-benchmark JSON output and generates plots comparing
performance over time, automatically grouping benchmarks by test source and parameters.
It specifically highlights "Exact" vs "Hutchinson" estimators.
"""

import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt


def get_benchmark_files(benchmarks_dir: str = ".benchmarks") -> List[str]:
    """recursively find benchmark json files."""
    return sorted(glob.glob(os.path.join(benchmarks_dir, "**", "*.json"), recursive=True))


def shorten_name(name: str) -> str:
    """Shorten benchmark name for display."""
    name = re.sub(r"^test_", "", name)
    name = re.sub(r"_benchmark$", "", name)
    name = re.sub(r"_performance$", "", name)

    # Remove "hutchinson" from name as it will be indicated by marker/style
    name = name.replace("hutchinson_", "")

    # Clean up scaling parameters
    name = re.sub(r"_scaling\[(\d+)\]", r" (\1)", name)
    name = re.sub(r"\[(.*?)\]", r" [\1]", name)

    name = name.replace("_", " ")
    return name.title().strip()


def categorize_benchmark(name: str) -> str:
    """Categorize benchmark based on its name."""
    # Prioritize particle benchmarks (often contain "flow" in name too)
    if "particles" in name or "particle" in name:
        return "Particles"
    elif "solver" in name:
        return "ODE Solvers"
    elif "dimension_scaling" in name:
        return "Dimension Scaling"
    elif "batch_size_scaling" in name:
        return "Batch Size Scaling"
    elif "samples_scaling" in name:
        return "Hutchinson Samples Scaling"
    elif "flow" in name or "sample" in name or "ad" in name:
        return "Flow Operations"
    else:
        return "Other"


def is_hutchinson(name: str) -> bool:
    """Check if benchmark is for Hutchinson estimator."""
    return "hutchinson" in name


def load_benchmark_data(json_files: List[str]) -> Dict[str, Dict]:
    """Load and aggregate benchmark data."""
    data = defaultdict(lambda: {"dates": [], "means": [], "stddevs": []})

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                content = json.load(f)

            timestamp_str = content.get("datetime")
            # If datetime is missing, try to infer from filename (common in pytest-benchmark)
            # Format: ..._YYYYMMDD_HHMMSS_...
            if not timestamp_str:
                match = re.search(r"_(\d{8})_(\d{6})_", os.path.basename(json_file))
                if match:
                    dt_str = f"{match.group(1)}{match.group(2)}"
                    timestamp = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
                else:
                    continue
            else:
                # Handle potentially slightly non-standard ISO strings if needed
                # But pytest-benchmark usually outputs standard ISO 8601
                timestamp = datetime.fromisoformat(timestamp_str)

            for bench in content.get("benchmarks", []):
                name = bench.get("name")
                stats = bench.get("stats", {})
                mean = stats.get("mean")
                stddev = stats.get("stddev")

                if name and mean is not None:
                    data[name]["dates"].append(timestamp)
                    data[name]["means"].append(mean)
                    data[name]["stddevs"].append(stddev if stddev is not None else 0)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    return data


def plot_benchmarks(data: Dict[str, Dict], output_file: str = ".benchmarks/benchmark_plot.png"):
    """Generate benchmark plots."""
    if not data:
        print("No data to plot.")
        return

    # Group benchmarks
    categories = defaultdict(list)
    for name in data.keys():
        cat = categorize_benchmark(name)
        categories[cat].append(name)

    n_cats = len(categories)
    n_cols = min(3, n_cats)
    n_rows = (n_cats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    fig.suptitle("Benchmark Performance History", fontsize=16, fontweight="bold", y=0.98)

    # Get all unique dates for x-axis consistency
    all_dates = set()
    for v in data.values():
        all_dates.update(v["dates"])
    all_dates = sorted(list(all_dates))

    # Color cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    for idx, (category, names) in enumerate(sorted(categories.items())):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # Filter out generic high-level benchmarks if specific ones exist
        # e.g., hide "Dimension Scaling" if "Dimension Scaling [2]" exists
        filtered_names = []
        for name in names:
            # Skip generic "Dimension Scaling" or "Batch Size Scaling" if they don't have parameters
            if "Scaling" in shorten_name(name) and "[" not in name and "(" not in shorten_name(name):
                continue
            filtered_names.append(name)
        names = filtered_names

        # Sort names
        def sort_key(n):
            # Extract numbers if present
            nums = re.findall(r"\d+", n)
            if nums:
                # (is_hutch, has_num=True, num_value, name)
                return (is_hutchinson(n), True, int(nums[-1]), n)
            # (is_hutch, has_num=False, num_value=-1, name)
            # Put Exact first for consistent coloring logic
            return (is_hutchinson(n), False, -1, n)

        names.sort(key=sort_key)

        # Map base names (without "hutchinson") to colors to ensure
        # corresponding exact/hutchinson benchmarks have same color
        base_name_to_color = {}
        color_idx = 0

        for name in names:
            # Determine base name for color matching
            # Remove "hutchinson_" and "test_" prefixes
            base_name = name.replace("hutchinson_", "").replace("test_", "")

            # For scaling benchmarks, we want "scaling[10]" and "hutchinson_scaling[10]" to match
            # The base_name logic above handles this naturally if names are consistent

            if base_name not in base_name_to_color:
                base_name_to_color[base_name] = colors[color_idx % len(colors)]
                color_idx += 1

            color = base_name_to_color[base_name]

            # Style based on type
            is_hutch = is_hutchinson(name)
            marker = "v" if is_hutch else "o"
            linestyle = "--" if is_hutch else "-"

            short_label = shorten_name(name)

            # Clean up label for legend
            # If it's a Hutchinson variant, remove the explicit "Hutchinson" text if we rely on markers
            # OR keep it for clarity. Let's make it cleaner:
            # "Dim [2]" (Exact) vs "Dim [2]" (Hutch)

            # If we stripped "Hutchinson" in shorten_name, add a suffix
            if is_hutch:
                label_suffix = " (Hutch)"
            else:
                label_suffix = " (Exact)"

            # Plot
            values = data[name]
            dates = values["dates"]
            means = [v * 1000 for v in values["means"]]  # ms
            stddevs = [v * 1000 for v in values["stddevs"]]

            # Sort by date
            sorted_pairs = sorted(zip(dates, means, stddevs))
            plot_dates_dt, plot_means, plot_stds = zip(*sorted_pairs)

            # Use ordinal x-axis (0, 1, 2...) but map to dates for consistency
            # Finding indices in the global sorted `all_dates` list
            x_indices = [all_dates.index(d) for d in plot_dates_dt]

            ax.errorbar(
                x_indices,
                plot_means,
                yerr=plot_stds,
                label=short_label + label_suffix,
                color=color,
                marker=marker,
                linestyle=linestyle,
                capsize=3,
                alpha=0.8,
            )

        ax.set_title(category, fontweight="bold")
        ax.set_ylabel("Time (ms)")
        ax.grid(True, linestyle=":", alpha=0.6)

        # Log scale often better for benchmarks
        ax.set_yscale("log")

        # X-axis formatting: Use indices, label with dates
        # Show all ticks if few, or sparsely if many
        n_dates = len(all_dates)
        if n_dates > 0:
            # Limit to max ~10 ticks to avoid clutter
            if n_dates <= 12:
                ticks = range(n_dates)
            else:
                # Pick ~10 points including first and last
                step = n_dates // 10
                ticks = list(range(0, n_dates, step))
                if ticks[-1] != n_dates - 1:
                    ticks.append(n_dates - 1)

            ax.set_xticks(ticks)
            # Format dates as MM-DD
            labels = [all_dates[i].strftime("%m-%d") for i in ticks]
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize="small")

            # Set limits with some padding
            ax.set_xlim(-0.5, n_dates - 0.5)

        # Legend inside the plot, upper left
        ax.legend(fontsize="x-small", loc="upper left", framealpha=0.9)

    # Hide empty subplots
    for idx in range(n_cats, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        fig.delaxes(axes[row, col])

    # Adjust layout to make room for left legends
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


def print_summary(data: Dict[str, Dict]):
    """Print tabular summary of latest results."""
    print("\n" + "=" * 80)
    print(f"{'BENCHMARK':<50} | {'TIME (ms)':<10} | {'STD (ms)':<10}")
    print("-" * 80)

    latest_results = []

    for name, values in data.items():
        if not values["dates"]:
            continue

        # Get latest
        # Sort by date
        zipped = sorted(zip(values["dates"], values["means"], values["stddevs"]))
        last_date, last_mean, last_std = zipped[-1]

        latest_results.append((name, last_mean * 1000, last_std * 1000))

    # Sort by name for readability
    latest_results.sort(key=lambda x: x[0])

    for name, mean, std in latest_results:
        print(f"{shorten_name(name):<50} | {mean:10.4f} | {std:10.4f}")
    print("=" * 80 + "\n")


def main():
    benchmarks_dir = ".benchmarks"  # Root, will search recursively
    files = get_benchmark_files(benchmarks_dir)

    if not files:
        print(f"No benchmark files found in {benchmarks_dir}")
        return

    print(f"Found {len(files)} benchmark files.")

    data = load_benchmark_data(files)
    print_summary(data)

    plot_benchmarks(data)


if __name__ == "__main__":
    main()
