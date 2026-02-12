#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import time
from pathlib import Path
from mpi4py import MPI  # Only needed if you plan to use MPI for aggregation

class EnergyMonitor:
    def __init__(self, benchmark_name=None):
        self.jobid = os.environ.get('SLURM_JOBID', 'local_job')
        self.step_id = os.environ.get("SLURM_STEP_ID", "nostep")
        if benchmark_name is None:
            benchmark_name = self.get_calling_filename()
        self.benchmark_name = benchmark_name
        self.output_dir = self.create_output_dir()
        print(f"EnergyMonitor: Using output directory: {self.output_dir}")

    @staticmethod
    def get_calling_filename():
        """Return the base name of the calling script (skipping frames in this module)."""
        current_module = os.path.basename(__file__)
        # Start at frame index 2 to try to skip this module’s frames
        for frame in inspect.stack()[2:]:
            caller_filename = os.path.basename(frame.filename)
            if caller_filename != current_module:
                return os.path.splitext(caller_filename)[0]
        return current_module

    def create_output_dir(self):
        """Creates an output directory based on jobid, benchmark name, and SLURM_STEP_ID."""
        output_dir = Path(f"power_{self.benchmark_name}_{self.jobid}_{self.step_id}")
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def take_snapshot(self, label):
        """Takes an energy snapshot with a given label ('start', 'lap', or 'end')."""
        nodeid = os.environ.get('SLURM_NODEID', '0')
        hostname = os.uname().nodename
        snapshot_file = self.output_dir / f"{nodeid}_snapshot_{label}.json"

        # Define energy counter file paths.
        energy_files = {
            'CPU': '/sys/cray/pm_counters/cpu_energy',
            'Mem': '/sys/cray/pm_counters/memory_energy',
            'Accel0': '/sys/cray/pm_counters/accel0_energy',
            'Accel1': '/sys/cray/pm_counters/accel1_energy',
            'Accel2': '/sys/cray/pm_counters/accel2_energy',
            'Accel3': '/sys/cray/pm_counters/accel3_energy',
            'Node': '/sys/cray/pm_counters/energy'
        }

        # Use a high-resolution timer and record a human-readable timestamp.
        snapshot = {
            "timestamp": time.perf_counter(),
            "timestamp_readable": time.ctime(),
            "hostname": hostname,
            "phase": label
        }

        for key, path in energy_files.items():
            try:
                with open(path) as f:
                    raw_value = f.read().strip()
                    # Extract the first token (numeric energy value)
                    numeric_str = raw_value.split()[0]
                    snapshot[key] = float(numeric_str)
            except Exception as e:
                print(f"DEBUG: Error reading {path}: {e}")
                snapshot[key] = None

        # Write snapshot JSON (using write mode so the file is overwritten if it exists).
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)

        print(f"Snapshot ('{label}') saved to {snapshot_file}")

    def start(self):
        """Record the starting snapshot."""
        self.take_snapshot('start')

    def lap(self):
        """Record an intermediate (lap) snapshot."""
        self.take_snapshot('lap')

    def end(self):
        """Record the ending snapshot."""
        self.take_snapshot('end')

    def report(self):
        """
        Generates a per-node report by reading snapshot files for the current node,
        then computing energy deltas, elapsed time, and average power.
        """
        nodeid = os.environ.get('SLURM_NODEID', '0')
        snapshot_files = sorted(self.output_dir.glob(f"{nodeid}_snapshot_*.json"))
        if not snapshot_files:
            print("No snapshot files found.")
            return

        snapshots = []
        for file in snapshot_files:
            try:
                with open(file) as f:
                    snap = json.load(f)
                    snapshots.append(snap)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        snapshots.sort(key=lambda s: s["timestamp"])

        print("Available snapshots:")
        for snap in snapshots:
            print(f"  Phase: {snap.get('phase', 'unknown')}, Time: {snap['timestamp_readable']}")

        print("\nDetailed Breakdown by Interval:")
        energy_keys = ["CPU", "Mem", "Accel0", "Accel1", "Accel2", "Accel3"]
        overall_total_energy = 0.0

        for i in range(len(snapshots) - 1):
            s1 = snapshots[i]
            s2 = snapshots[i+1]
            interval_time = s2["timestamp"] - s1["timestamp"]
            interval_total_energy = 0.0

            print(f"\nInterval: '{s1.get('phase')}' ({s1['timestamp_readable']}) -> '{s2.get('phase')}' ({s2['timestamp_readable']})")
            print(f"  Elapsed Time: {interval_time:.2f} seconds")

            for key in energy_keys:
                try:
                    delta_value = s2.get(key, 0) - s1.get(key, 0)
                    interval_total_energy += delta_value
                    print(f"  {key}: {delta_value:.2f} Joules")
                except Exception as e:
                    print(f"  Error computing delta for {key}: {e}")

            overall_total_energy += interval_total_energy
            if interval_time > 0:
                interval_avg_power = interval_total_energy / interval_time
            else:
                interval_avg_power = None

            print(f"  Total Energy for interval: {interval_total_energy:.2f} Joules")
            if interval_avg_power is not None:
                print(f"  Average Power for interval: {interval_avg_power:.2f} Watts")
            else:
                print("  Average Power for interval: not available")

        overall_elapsed = snapshots[-1]["timestamp"] - snapshots[0]["timestamp"]
        print("\nOverall Summary (first snapshot to last snapshot):")
        print(f"  Overall Energy Consumed: {overall_total_energy:.2f} Joules")
        print(f"  Overall Elapsed Time: {overall_elapsed:.2f} seconds")
        if overall_elapsed > 0:
            overall_avg_power = overall_total_energy / overall_elapsed
            print(f"  Overall Average Power: {overall_avg_power:.2f} Watts")
        else:
            print("  Overall Average Power: not available")

    def aggregate(self):
        """
        Aggregates snapshot data from all nodes in the output directory,
        computes per-node energy consumption (with breakdown between CPU and GPU),
        and then produces an overall cluster summary.
        """
        snapshot_files = sorted(self.output_dir.glob("*_snapshot_*.json"))
        if not snapshot_files:
            print("No snapshot files found for aggregation.")
            return

        # Group snapshots by node id (assumes filename starts with node id)
        node_snapshots = {}
        for file in snapshot_files:
            node_id = file.name.split('_')[0]
            if node_id not in node_snapshots:
                node_snapshots[node_id] = []
            try:
                with open(file) as f:
                    snap = json.load(f)
                    node_snapshots[node_id].append(snap)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        overall_cluster_energy = 0.0
        overall_cpu_energy = 0.0
        overall_gpu_energy = 0.0
        cluster_start = None
        cluster_end = None

        print("Per-Node Energy Consumption:")
        for node_id, snaps in node_snapshots.items():
            if not snaps:
                continue
            snaps.sort(key=lambda s: s["timestamp"])
            node_start = snaps[0]["timestamp"]
            node_end = snaps[-1]["timestamp"]
            if cluster_start is None or node_start < cluster_start:
                cluster_start = node_start
            if cluster_end is None or node_end > cluster_end:
                cluster_end = node_end

            # CPU energy = CPU + Mem; GPU energy = Accel0 + Accel1 + Accel2 + Accel3
            cpu_energy = ((snaps[-1].get("CPU", 0) - snaps[0].get("CPU", 0)) +
                          (snaps[-1].get("Mem", 0) - snaps[0].get("Mem", 0)))
            gpu_energy = ((snaps[-1].get("Accel0", 0) - snaps[0].get("Accel0", 0)) +
                          (snaps[-1].get("Accel1", 0) - snaps[0].get("Accel1", 0)) +
                          (snaps[-1].get("Accel2", 0) - snaps[0].get("Accel2", 0)) +
                          (snaps[-1].get("Accel3", 0) - snaps[0].get("Accel3", 0)))
            node_total_energy = cpu_energy + gpu_energy
            overall_cluster_energy += node_total_energy
            overall_cpu_energy += cpu_energy
            overall_gpu_energy += gpu_energy

            elapsed_node = node_end - node_start
            if elapsed_node > 0:
                avg_power_node = node_total_energy / elapsed_node
            else:
                avg_power_node = None

            print(f"\nNode {node_id}:")
            print(f"  Total Energy Consumed: {node_total_energy:.2f} Joules")
            print(f"    CPU Energy (CPU + Mem): {cpu_energy:.2f} Joules")
            print(f"    GPU Energy (Accel0 - Accel3): {gpu_energy:.2f} Joules")
            print(f"  Elapsed Time: {elapsed_node:.2f} seconds")
            if avg_power_node is not None:
                print(f"  Average Power: {avg_power_node:.2f} Watts")
            else:
                print("  Average Power: not available")

        if cluster_start is not None and cluster_end is not None:
            cluster_elapsed = cluster_end - cluster_start
        else:
            cluster_elapsed = 0.0

        print("\nOverall Cluster Summary:")
        print(f"  Total Energy Consumed: {overall_cluster_energy:.2f} Joules")
        print(f"    Total CPU Energy (CPU + Mem): {overall_cpu_energy:.2f} Joules")
        print(f"    Total GPU Energy (Accel0 - Accel3): {overall_gpu_energy:.2f} Joules")
        print(f"  Cluster Elapsed Time: {cluster_elapsed:.2f} seconds")
        if cluster_elapsed > 0:
            overall_avg_power = overall_cluster_energy / cluster_elapsed
            print(f"  Overall Average Power: {overall_avg_power:.2f} Watts")
        else:
            print("  Overall Average Power: not available")


# Example of using the EnergyMonitor class
def main():
    parser = argparse.ArgumentParser(description="Energy benchmarking tool (class-based)")
    parser.add_argument("command", choices=["start", "lap", "end", "report", "aggregate"],
                        help="Command to execute (start, lap, end, report, aggregate)")
    parser.add_argument("--benchmark", help="Optional benchmark name", default=None)
    args = parser.parse_args()

    # Instantiate the monitor with the given benchmark name (or let it be auto-detected)
    monitor = EnergyMonitor(benchmark_name=args.benchmark)

    if args.command == "start":
        monitor.start()
    elif args.command == "lap":
        monitor.lap()
    elif args.command == "end":
        monitor.end()
    elif args.command == "report":
        monitor.report()
    elif args.command == "aggregate":
        monitor.aggregate()


if __name__ == "__main__":
    main()
