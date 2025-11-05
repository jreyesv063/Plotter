import os
import re
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

from coffea import util, processor
from collections import defaultdict
from coffea.lookup_tools.dense_lookup import dense_lookup


class BTagEfficiencyProcessor:
    """
    A class to process and combine b-tagging efficiency histograms into lookup tables.
    """

    def __init__(self, output_path: str, output_file: str):
        """
        Initialize the processor with paths.
        
        Args:
            output_path (str): Directory where .pkl files are stored.
            output_file (str): Path where the combined .coffea output will be saved.
        """
        self.output_path = output_path
        self.output_file = output_file
        self.eff_lookup_dict = {}

    def group_files(self):
        """
        Group .pkl files by sample name (removing trailing numbers).
        
        Returns:
            dict: A dictionary mapping sample names to lists of file paths.
        """
        outputs = glob.glob(f"{self.output_path}/*.pkl")
        grouped = defaultdict(list)

        for path in outputs:
            filename = os.path.basename(path).replace(".pkl", "")
            key = re.sub(r'_\d+$', '', filename)
            grouped[key].append(path)

        return grouped

    def process(self):
        """
        Process all .pkl files found in the output_path and save a combined .coffea file.
        """
        grouped = self.group_files()

        for sample, file_list in grouped.items():
            print(f"\n=== Processing {sample} ===")
            histograms = []

            # --- Load histograms from each file ---
            for output in file_list:
                with open(output, "rb") as f:
                    histogram = pickle.load(f)
                    for _, v in histogram.items():
                        histograms.append(v["histograms"])

            # --- Accumulate histograms ---
            btag_efficiency_hist = processor.accumulate(histograms)
            acc_btag_efficiency_hist = btag_efficiency_hist[{"dataset": sum}]
            efficiency = (
                acc_btag_efficiency_hist[{"passWP": True}] /
                acc_btag_efficiency_hist[{"passWP": sum}]
            )

            eff_view = efficiency.view(flow=True)
            np.nan_to_num(eff_view, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # --- Display available axes ---
            print("Axes in efficiency histogram:")
            for ax in efficiency.axes:
                print(f"  - {ax.name} ({type(ax).__name__})")
                if hasattr(ax, "categories"):
                    print(f"    Categories: {ax.categories}")
                elif hasattr(ax, "centers"):
                    print(f"    Centers: {ax.centers}")
                elif hasattr(ax, "edges"):
                    print(f"    Edges: {ax.edges}")

            # --- Calculate efficiencies per jet flavor ---
            flavor_tags = {0: "LightJets", 4: "c-Jets", 5: "b-Jets"}
            print("Efficiencies by jet flavor:")

            efflookup = None

            if "flavor" in [ax.name for ax in efficiency.axes]:
                flavor_axis = efficiency.axes["flavor"]
                flavor_map = {0: 0, 1: 4, 2: 5}

                # Create lookup table
                efflookup = dense_lookup(efficiency.values(), [ax.edges for ax in efficiency.axes])

                # --- Plot results for each flavor ---
                for idx, real_flavor in flavor_map.items():
                    try:
                        eff_flavor = efficiency.values()[..., idx]
                        eff_min = np.min(eff_flavor)
                        eff_max = np.max(eff_flavor)
                        print(f"  → {flavor_tags[real_flavor]}: min = {eff_min:.4f}, max = {eff_max:.4f}")

                        pts = np.linspace(20, 1000)
                        etas = np.linspace(0, 2.5)
                        pt, eta = np.meshgrid(pts, etas)
                        fig, ax = plt.subplots()
                        heatmap = ax.pcolormesh(pt, eta, efflookup(pt, eta, idx),
                                                cmap='viridis', vmin=0, vmax=1)
                        cbar = fig.colorbar(heatmap)
                        ax.set_xlabel('$p_T$ [GeV]')
                        ax.set_ylabel('$|\eta|$')
                        cbar.set_label(f'{flavor_tags[real_flavor]} b-tagging Efficiency')
                        ax.set_title(f"{sample} ({flavor_tags[real_flavor]})")

                        plt.show()
                    except Exception as e:
                        print(f"  ⚠️ Could not compute efficiency for flavor {real_flavor}: {e}")
            else:
                print("⚠️ 'flavor' axis not found in efficiency histogram.")
                continue

            # --- Add lookup to dictionary ---
            if efflookup is not None:
                self.eff_lookup_dict[sample] = efflookup

        # --- Save combined lookup dictionary ---
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        util.save(self.eff_lookup_dict, self.output_file)
        print(f"\n✅ Combined file saved: {self.output_file}")
        print(f"Contains {len(self.eff_lookup_dict)} samples.")

    def verify_output(self, sample_name=None):
        """
        Verify and visualize the contents of the saved .coffea efficiency file.

        Args:
            sample_name (str, optional): Name of a specific sample to plot.
                                         If None, plots all samples.
        """
        print(f"\n🔍 Loading efficiency file: {self.output_file}")
        efflookup_dict = util.load(self.output_file)

        # Internal index -> true jet flavor
        flavor_map = {0: 0, 1: 4, 2: 5}
        flavor_tags = {0: "LightJets", 4: "c-Jets", 5: "b-Jets"}

        # Prepare mesh for visualization
        pts = np.linspace(20, 1000, 200)
        etas = np.linspace(0, 2.5, 100)
        pt, eta = np.meshgrid(pts, etas)

        # Choose samples to visualize
        samples_to_plot = [sample_name] if sample_name else efflookup_dict.keys()

        for sample in samples_to_plot:
            if sample not in efflookup_dict:
                print(f"⚠️ Sample '{sample}' not found in file.")
                continue

            efflookup = efflookup_dict[sample]

            print(f"\n📊 Plotting efficiency maps for sample: {sample}")
            for idx, real_flavor in flavor_map.items():
                try:
                    eff_map = efflookup(pt, eta, np.full_like(pt, idx))
                    eff_min = np.nanmin(eff_map)
                    eff_max = np.nanmax(eff_map)
                    print(f"  → {flavor_tags[real_flavor]}: min = {eff_min:.4f}, max = {eff_max:.4f}")

                    fig, ax = plt.subplots(figsize=(8, 6))
                    heatmap = ax.pcolormesh(pt, eta, eff_map, cmap='viridis', vmin=0, vmax=1)
                    cbar = fig.colorbar(heatmap)
                    ax.set_xlabel('$p_T$ [GeV]')
                    ax.set_ylabel('$|\eta|$')
                    cbar.set_label(f'{flavor_tags[real_flavor]} b-tagging Efficiency')
                    plt.title(f'{sample} – {flavor_tags[real_flavor]} Efficiency Map')
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"  ⚠️ Could not plot {flavor_tags[real_flavor]}: {e}")
