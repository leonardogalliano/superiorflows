import glob
import json
import os

import dateutil.parser
import matplotlib.pyplot as plt


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

    plt.figure(figsize=(12, 8))

    all_dates = set()
    for values in data.values():
        all_dates.update(values["dates"])

    sorted_unique_dates = sorted(list(all_dates))
    date_to_index = {d: i for i, d in enumerate(sorted_unique_dates)}

    for name, values in data.items():
        sorted_pairs = sorted(zip(values["dates"], values["means"], values["stddevs"]))

        x_indices = [date_to_index[p[0]] for p in sorted_pairs]
        means = [p[1] for p in sorted_pairs]
        stddevs = [p[2] for p in sorted_pairs]

        plt.errorbar(x_indices, means, yerr=stddevs, marker="o", label=name, capsize=3)

    plt.title("Benchmark Performance Over Time")
    plt.xlabel("Benchmark Run Index")
    plt.ylabel("Execution Time (s)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="-")

    plt.xticks(
        range(len(sorted_unique_dates)),
        [d.strftime("%Y-%m-%d %H:%M") for d in sorted_unique_dates],
        rotation=45,
        ha="right",
    )
    plt.tight_layout()

    output_file = os.path.join(benchmarks_dir, "benchmark_plot.png")
    output_file = ".benchmarks/benchmark_plot.png"

    plt.savefig(output_file)
    print(f"Benchmark plot saved to {output_file}")


if __name__ == "__main__":
    main()
