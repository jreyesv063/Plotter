import os
import json
import numpy as np
import pandas as pd
import mplhep as hep
from coffea import processor
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union, Tuple


class HistogramPlotter:
    def __init__(
        self, 
        year: str = "2017", 
        output_dir: str = "plot",
        lepton_flavor: str = "tau", 
        combined_2016: bool = False
    ):

        # General configs
        self.output_dir = output_dir
        self.year = year
      

        plt.style.use(hep.style.ROOT)
        plt.style.use('default')
        plt.close('all')

        # ==============================================
        #  Using json files
        # ==============================================
        # Load lumi labels
        with open("json_files/lumi_labels.json", "r") as f:
            lumi_map = json.load(f)  
            
        if combined_2016:
            self.lumi = lumi_map['2016_full']    
        else:
            self.lumi = lumi_map[year]

        # Load colors and names
        with open("json_files/bgrs.json", "r") as f:
            bgrs_info = json.load(f)
        
        self.sample_map = bgrs_info['names']
        self.sample_colors = bgrs_info['colors']


        # Load variable labels
        with open("json_files/labels.json", "r") as f:
            self.label_map = json.load(f)[lepton_flavor]



    def plot(
        self,
        # Histogram
        distribution: str,
        grouped_histos: Dict[str, np.ndarray],
        binning: Union[List[float], np.ndarray],
        main_bgr: str,
        
        # Style
        events_gev: bool,
        log_scale: bool,       
        cms_loc: float,
        y_axis_range: tuple,
        ratio_axis_range: tuple,

        # Signals and systematic variations
        signals: bool,
        include_systematics,

        # Stat and syst error
        stat_error: Dict[str, np.ndarray] = None,
        syst_error: Dict[str, np.ndarray] = None

    ):
        
        dist_label = self.label_map[distribution]
        bin_widths = binning [1:] - binning [:-1]

        mcs = []
        main_back = []
        data_hist = []

        for sample, hist in grouped_histos.items():
            # data: ["SingleMuon", "SingleElectron", "Tau", "MET"]
            if sample == "data":
                data_hist.append(hist)
                continue


            elif sample not in ["data", *[f"SignalTau_{m}GeV" for m in [300, 400, 600, 750, 1000, 1500, 2000, 3000]]]:
                # Normalize by bin width if requested
                norm_hist = hist / bin_widths if events_gev else hist
                mcs.append(norm_hist)


        # Sum all MC histograms
        total_data = processor.accumulate(data_hist) if data_hist else None
        total_mc = processor.accumulate(mcs) if mcs else None
        total_main_back = processor.accumulate(main_back) if main_back else None


        # Create figure and axes
        fig, axes = self.create_figure(signals)
        self.add_cms_labels(axes, cms_loc)

        self.stat_error = stat_error
        self.syst_error = syst_error
        
        self.draw_main_components(
            grouped_histos, 
            axes, 
            signals,
            events_gev, 
            bin_widths, 
            binning, 
            dist_label, 
            log_scale, 
            y_axis_range, 
            ratio_axis_range
        )
        
        self.save_and_show(distribution)

    # ------------------------------------------
    #    Create canvas
    # -------------------------------------------
    def create_figure(self, signals) -> Tuple[plt.Figure, Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]]:
        """Create figure with appropriate subplot configuration"""
        if signals:
            fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
            return fig, ax
        else:
            fig, (ax, ax_ratio) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(8, 7),
                tight_layout=True,
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True
            )
            return fig, (ax, ax_ratio)

    def add_cms_labels(self, axes, cms_loc) -> None:
        """Add CMS experiment labels and luminosity information"""
        ax = axes[0] if isinstance(axes, tuple) else axes
        
        hep.cms.lumitext(self.lumi, fontsize=14, ax=ax)
        
        hep.cms.text("Preliminary", loc=cms_loc, fontsize=16, ax=ax)
        

       


    def draw_main_components(
        self, 
        grouped_histos, 
        axes, 
        signals,
        events_gev, 
        bin_widths, 
        binning, 
        dist_label, 
        log_scale,
        y_axis_range,
        ratio_axis_range
    ) -> None:
        """Draw the primary histogram components (MC, Data, Signals)"""
        ax, ax_ratio = axes if isinstance(axes, tuple) else (axes, None)

        if ax_ratio and (grouped_histos.get("data", np.array([])).sum() == 0):
            ax_ratio.remove()
            ax_ratio = None
            
        sample_map = {
            "tt": "tt",
            "dy": "DYJetsToLL",
            "st": "SingleTop",
            "vv": "VV",
            "wj": "WJetsToLNu",
            "higgs": "Higgs",
            "qcd": "QCD",
        }
    
        # Excluir Data y Señales del stack
        exclude_keys = ["data"] + [key for key in grouped_histos if key.startswith("Signal")]
    
        # Filtrar MC válidos
        mc_samples_unsorted = [
            s for s in grouped_histos
            if s not in exclude_keys and grouped_histos[s] is not None
        ]
        
        # Ordenar por integral
        sample_integrals = {s: grouped_histos[s].sum() for s in mc_samples_unsorted}
        mc_samples = sorted(sample_integrals, key=sample_integrals.get)
        
        # Preparar histos y colores
        stacked_histos = [grouped_histos[s] for s in mc_samples]
        colors = [self.sample_colors[s] for s in mc_samples]

        # --------------------------------------
        # Total MC
        # --------------------------------------
        total_mc = np.sum(stacked_histos, axis=0)
            
        
        # --------------------------------------
        # Stadistical error
        # --------------------------------------
        if self.stat_error is not None:

            error_label = "stat unc"
            
            stat_bgr_error_down = {}
            stat_bgr_error_up = {}
            
            stat_bgr_error_down["total"] = 0.0
            stat_bgr_error_up["total"] = 0.0

            total2 = 0.0
            
            for bgr in self.stat_error:
                if bgr != "data" and not bgr.startswith("Signal"):
                    total2 = total2 + self.stat_error[bgr]**2

            stat_bgr_error_down["total"] = np.sqrt(total2)
            stat_bgr_error_up["total"] = np.sqrt(total2)

            # --------------------------------------
            # Total error: MC
            # --------------------------------------   
            total_bgr_error_up = np.sqrt(
                stat_bgr_error_up["total"]**2
            )
            total_bgr_error_down = np.sqrt(
                stat_bgr_error_down["total"]**2
            )


        # --------------------------------------
        # Systematic error
        # --------------------------------------        
        if self.syst_error is not None:
            error_label = "stat + syst unc"
            
            syst_bgr_error_down = {}
            syst_bgr_error_up = {}

            syst_bgr_error_down["total"] = 0.0
            syst_bgr_error_up["total"] = 0.0

            total2_up = 0.0
            total2_down = 0.0
            
            for bgr in self.syst_error:
                if bgr != "data" and not bgr.startswith("Signal"):
                    total2_up = total2_up + self.syst_error[bgr]['Up']**2
                    total2_down = total2_down + self.syst_error[bgr]['Down']**2
                
            syst_bgr_error_up["total"] = np.sqrt(total2_up)            
            syst_bgr_error_down["total"] = np.sqrt(total2_down)

            # --------------------------------------
            # Total error: MC
            # --------------------------------------   
            total_bgr_error_up = np.sqrt(
                syst_bgr_error_up["total"]**2 +
                stat_bgr_error_up["total"]**2
            )
            total_bgr_error_down = np.sqrt(
                syst_bgr_error_down["total"]**2 +
                stat_bgr_error_down["total"]**2
            )

        # --------------------------------------
        # Scale histograms
        # --------------------------------------              
        if events_gev:
            scaled_stacked = stacked_histos / bin_widths
            total_mc_scaled = total_mc/bin_widths

            error_down = (total_mc - total_bgr_error_down) / bin_widths
            error_up =  (total_mc + total_bgr_error_up) / bin_widths
            
        else:
            scaled_stacked = stacked_histos

            error_down = (total_mc - total_bgr_error_down)
            error_up =  (total_mc + total_bgr_error_up) 

        # --------------------------------------
        # Draw MC stack + data points + unc bar 
        # --------------------------------------                      
        ax.tick_params(axis='both', labelsize=14)
        hep.histplot(
            scaled_stacked,
            bins=binning,
            ax=ax,
            histtype="fill",
            stack=True,
            color=colors,
            edgecolor="k",
            linewidth=1.0,
            label=[sample_map.get(s, s) for s in mc_samples]
        )

        # --- Error MC ---
        error_down_step = np.repeat(error_down, 2)
        error_up_step = np.repeat(error_up, 2)
        bin_edges_step = np.repeat(binning, 2)[1:-1]
        

        ax.fill_between(
            bin_edges_step,
            error_down_step,
            error_up_step,
            color="lightgray",
            alpha=0.5,
            edgecolor="black",
            hatch="///",
            linewidth=0,
            label=error_label
        )

        

        
        # --- Draw Data  ---
        if "data" in grouped_histos:
            data = grouped_histos["data"]
 
            if events_gev:
                scaled_data = data / bin_widths
                scaled_errors = np.sqrt(data) / bin_widths

            else:
                scaled_data = data
                scaled_errors = np.sqrt(data)

        
        bin_centers = (binning[:-1] + binning[1:]) / 2
        bin_widths = bin_widths / 2
        
        ax.errorbar(
            bin_centers,
            scaled_data,
            xerr=bin_widths,
            yerr=scaled_errors,
            fmt='k.',
            markersize=10,
            linestyle='none',
            capsize=0,
            label="Data"
        )

        # --- Draw Signal ---
        if signals:
            signal_keys = [k for k in grouped_histos if k.startswith("SignalTau_")]
            for signal_key in sorted(signal_keys):
                signal_hist = grouped_histos[signal_key]
                if signal_hist is None:
                    continue
                if events_gev:
                    signal_hist = signal_hist / bin_widths
                    
                hep.histplot(
                    signal_hist,
                    bins=binning,
                    ax=ax,
                    histtype="step",
                    color=self.sample_colors.get(signal_key, "r"),
                    linestyle="--",
                    linewidth=2,
                    label=sample_map.get(signal_key, signal_key)
                )
        


        
        # --------------------------------------
        # Configure style
        # --------------------------------------  
        ax.set_xlim(binning[0], binning[-1])
        ax.set_xlabel(f"{dist_label}", fontsize=16)
        ax.set_ylabel("Events/GeV" if events_gev else "Events", fontsize=16)
        ax.set_yscale("log" if log_scale else "linear")
        ax.set_ylim(y_axis_range)
        
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            fontsize=12,
            frameon=False
        )

        # --------------------------------------       
        # --- Ratio Data/MC  ---
        # --------------------------------------        
        if ax_ratio is not None:
            ratio = np.divide(data, total_mc, out=np.zeros_like(data, dtype=float), where=total_mc != 0)

            # Poisson error
            error_data_down = error_data_up = np.sqrt(data)/data
            error_mc_down  =  error_down / total_mc_scaled
            error_mc_up  =  error_up / total_mc_scaled
            
            yerr = np.vstack([error_data_down, error_data_up])

            ax_ratio.errorbar(
                bin_centers,
                ratio,
                xerr=bin_widths,
                yerr=yerr,
                fmt='ko',
                markersize=5,
                capsize=0
            )
            ax_ratio.fill_between(
                binning,
                np.append(error_mc_down, error_mc_down[-1]),
                np.append(error_mc_up, error_mc_up[-1]),
                step="pre",
                color='lightgray',
                alpha=0.5,
                edgecolor='black',
                hatch='///',
                linewidth=0,
                label="MC stat. unc."
            )
            

            ax_ratio.axhline(1, color='k', linestyle='--')
            ax_ratio.set_ylabel("Data / Total bgr", fontsize=15)
            ax_ratio.set_xlabel(f"{dist_label}", fontsize=16)
            ax_ratio.set_ylim(ratio_axis_range)
            ax_ratio.grid(True)
            

        
    # --------------------------------
    #   Save pdf file
    # --------------------------------
    def save_and_show(self, distribution) -> None:
        """Save plot to file and display it"""

        os.makedirs(self.output_dir, exist_ok=True)

        year_pdf_file = self.year
            
        output_path = f"{self.output_dir}/{distribution}_{year_pdf_file}.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
        plt.show()
        plt.close()
