import os
import numpy as np
import pandas as pd
import mplhep as hep
from coffea import processor
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union, Tuple

from src.intervals import  poisson_interval_v2

def get_hist(
    feature: str,
    pkls: Dict[str, Dict[str, Any]],
    bins: np.ndarray,
    weights_variation: Optional[str] = None,
    consider_overflow: bool = True,
    consider_underflow: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute weighted histograms for a given feature across multiple samples.

    Args:
        feature (str): Variable to histogram.
        pkls (Dict[str, Dict[str, Any]]): Dictionary of samples, each containing a sub-dictionary with variables.
        bins (np.ndarray): Bin edges.
        weights_variation (Optional[str]): Weight key (e.g. "weights" or "weights_L1PrefiringUp").
        consider_overflow (bool): Whether to include overflow in the last bin.
        consider_underflow (bool): Whether to include underflow in the first bin.

    Returns:
        Dict[str, np.ndarray]: Dictionary of histograms by sample.
    """
    histograms = {}
    

    for sample, sample_data in pkls.items():
        if any(key.endswith("_2016APV") for key in pkls.keys()):
            expected_key = sample.rsplit("_", 1)[0]
        else:
            expected_key = sample
            
        if expected_key not in sample_data:
            print(f"⚠️  Subkey '{sample}' not found inside its own entry. Skipping.")
            continue


        arrays = sample_data[expected_key]

        if feature not in arrays:
            #print(f"⚠️  Feature '{feature}' not found in sample '{sample}'. Skipping.")
            continue

        variable = arrays[feature]
        is_data_sample = sample.startswith(("SingleMuon", "SingleElectron", "Tau", "MET"))
        weight = None if is_data_sample else arrays.get(weights_variation)

        if variable is None or (not is_data_sample and weight is None):
            print(f"⚠️  Missing data or weights in '{sample}'. Skipping.")
            continue

        underflow_mask = variable < bins[0]
        overflow_mask = variable > bins[-1]

        if is_data_sample:
            underflow = np.sum(underflow_mask)
            overflow = np.sum(overflow_mask)
            hist, _ = np.histogram(variable, bins=bins)
        else:
            underflow = np.sum(weight[underflow_mask])
            overflow = np.sum(weight[overflow_mask])
            hist, _ = np.histogram(variable, bins=bins, weights=weight)

        if consider_underflow:
            hist[0] += underflow
        if consider_overflow:
            hist[-1] += overflow

        histograms[sample] = hist

    return histograms
    
    
class HistogramPlotter:
    def __init__(self, year: str, lepton_flavor: str = "mu", combined_2016: bool = False, is_signal: bool = False, output_dir: str = "plot"):
        """
        Initialize a histogram plotter for HEP data visualization.
        
        Args:
            year (str): Data-taking year (e.g., "2017", "2018")
            lepton_flavor (str): Lepton channel ("mu", "ele", "tau")
            combined_2016 (bool): Whether to use combined 2016 data
        """
        self.output_dir = output_dir
        self.year = year
        self.lepton_flavor = lepton_flavor
        self.combined_2016 = combined_2016
        self.is_signal = is_signal
        self.sample_colors = self.init_sample_colors()
        self.label_map = self.init_label_map()
        self.sample_map = self.init_sample_map()
        plt.style.use(hep.style.ROOT)
        plt.style.use('default')
        plt.close('all')

    def init_sample_colors(self) -> Dict[str, str]:
        """Define color scheme for different physics processes"""
        return {
            "wj": "#f89c20",              # Orange
            "dy": "#5790fc",              # Blue
            "vv": "#e42536",              # Red
            "tt": "#A9A9A9",              # Gray
            "st": "#8B008B",              # Dark Magenta
            "higgs": "#FFFF00",           # Yellow
            "qcd": "#ffc0cb",             # Pink
            "SignalTau_600GeV": "green",
            "SignalTau_1TeV": "blue",
            "SignalTau_2TeV": "red",
            "SignalTau_3TeV": "orange",
        }

    def init_label_map(self) -> Dict[str, Dict[str, str]]:
        """Define axis labels for different physics variables"""
        return {
            "ele": {
                "jet_pt": r"$p_T$(b-Jet$_{0}$) [GeV]",
                "jet_eta": r"$\eta$(b-Jet$_{0}$)",
                "jet_phi": r"$\phi$(b-Jet$_{0}$)",
                "met": r"$p_T^{miss}$ [GeV]",
                "met_phi": r"$\phi(p_T^{miss})$",
                "lepton_pt": r"$p_T(e)$ [GeV]",
                "lepton_relIso": "$e$ RelIso",
                "lepton_eta": r"$\eta(e)$",
                "lepton_phi": r"$\phi (e)$",
                "lepton_bjet_mass": r"$m(e, $b-Jet$_{0})$ [GeV]",
                "lepton_bjet_dr": r"$\Delta R$($e$, b-Jet$_{0}$)",
                "lepton_met_mass": r"$m_T$($e$, $p_T^{miss}$) [GeV]",
                "lepton_met_delta_phi": r"$\Delta \phi(e, p_T^{miss})$",
                "lepton_met_abs_delta_phi": r"$|\Delta \phi(e, p_T^{miss})|$",
                "lepton_met_bjet_mass": r"$m_T^{tot}(e, $b-Jet$_{0}, p_T^{miss})$ [GeV]",
                "dilepton_mass": r"$m_{ee}$ [GeV]"
            },
            "mu": {
                "jet_pt": r"$p_T$(jet) [GeV]",
                "bjet_pt": r"$p_T$(bjet$_{0}$) [GeV]",
                "jet_eta": r"$\eta$(b-Jet$_{0}$)",
                "jet_phi": r"$\phi$(b-Jet$_{0}$)",
                "met": r"$p_T^{miss}$ [GeV]",
                "met_pt_nomu":  r"$p_T^{miss}(\mu)$ [GeV]",
                "pt_nomu_minus": r"$p_T^{miss}(\mu)$ [GeV]",
                "pt_nomu_plus": r"$p_T^{miss}(\mu)$ [GeV]",
                "recoil_pt":  r"$p_T^{miss}(recoil)$ [GeV]",
                "met_raw":  r"$p_T^{miss}(raw)$ [GeV]",
                "met_phi": r"$\phi(p_T^{miss})$",
                "lepton_pt": r"$p_T(\mu)$ [GeV]",
                "lepton_eta": r"$\eta(\mu)$",
                "lepton_phi": r"$\phi (\mu)$",
                "lepton_bjet_mass": r"$m(\mu, $b-Jet$_{0})$ [GeV]",
                "lepton_bjet_dr": r"$\Delta R$($\mu$, b-Jet$_{0}$)",
                "lepton_met_mass": r"$m_T$($\mu$, $p_T^{miss}$) [GeV]",
                "lepton_met_delta_phi": r"|$\Delta \phi(\mu, p_T^{miss})$|",
                "lepton_met_abs_delta_phi": r"$|\Delta \phi(\mu, p_T^{miss})|$",
                "lepton_met_bjet_mass": r"$m_T^{tot}(\mu, $b-Jet$_{0}, p_T^{miss})$ [GeV]",
                "dilepton_mass": r"$m_{\mu \mu}$ [GeV]",
                "lepton_one_pt": r"$p_{T}(\mu_{1})$ [GeV]",
                
                "mll":  r"$m(\mu\mu)$[GeV]",
                "ptl1": r"$p_T(\mu_{leading})$ [GeV]",
                "ptl2": r"$p_T(\mu_{subleading})$ [GeV]",
                "ptll": r"$p_{T}(\mu\mu)$",
                

                "top_mrec": r"$m_{rec}(top)$ [GeV]",
                
                "njets":  r"$N(j)$", 
                "njets_full":  r"$N(j + b)$", 
                "nbjets": r"$N(b)$",    
                "npvs": r"$npvs$",    
                "nmuons": r"$N(\mu)$",    
                "nelectrons": r"$N(e)$",    
                "ntaus": r"$N(\tau)$",   

                "HT": "HT [GeV]",
                "Z_gen_pt": "Z(gen-level) [GeV]",
                "Z_gen_num": "n[Z(gen-level)]"
                
            },
          "tau": {
                "delta_phi_met_jet": r"$|$Delta$phi(jet, met)|",
                "delta_phi_met_lepton": r"$|$Delta$phi($\tau$, met)|",
                
                "jet_pt": r"$p_T$(jets) [GeV]",
                "bjet_pt": r"$p_T$(bjet) [GeV]",
                "bjet_phi": "$\phi(bjets)$",
                "bjet_eta": r"$\eta$(bjets)",              
                "jet_eta": r"$\eta$(jets)",
                "jet_phi": r"$\phi$(jets)",

                "met": r"$p_T^{miss}$ [GeV]",
                "met_phi": r"$\phi(p_T^{miss})$",
                "lepton_pt": r"$p_T(\tau)$ [GeV]",
                "lepton_relIso": "$\tau$ RelIso",
                "lepton_eta": r"$\eta(\tau)$",
                "lepton_phi": r"$\phi (\tau)$",
                "lepton_bjet_mass": r"$m(\tau, $b-Jet$_{0})$ [GeV]",
                "lepton_bjet_dr": r"$\Delta R$($\tau$, b-Jet$_{0}$)",
                "lepton_met_mass": r"$m_T$($\tau$, $p_T^{miss}$) [GeV]",
                "lepton_met_delta_phi": r"|$\Delta \phi(\tau, p_T^{miss})$|",
                "lepton_met_abs_delta_phi": r"$|\Delta \phi(\tau, p_T^{miss})|$",
                "lepton_met_bjet_mass": r"$m_T^{tot}(\tau, $b-Jet$_{0}, p_T^{miss})$ [GeV]",
                "dilepton_mass": r"$m_{\tau \tau}$ [GeV]",     
                
                "top_mrec": r"$m_{rec}(top)$ [GeV]",
                "w_mrec": r"$m_{rec}(W)$ [GeV]",
                
                "njets":  r"$N(j)$", 
                "njets_old": r"N(j)",
                "njets_full":  r"$N(j + f + b)$", 
                "nbjets": r"$N(b)$",                  
                "npvs": r"$npvs$",    
                "nmuons": r"$N(\mu)$",    
                "nelectrons": r"$N(e)$",    
                "ntaus": r"$N(\tau)$",   
                "njets_no_top_tagger": r"$N(jets-no top)$",
        
                "genPartFlav": r"genPartFlav(\tau)",
                "decayMode": r"decayMode(\tau)",
                "isolation_electrons": r"\tau Vs e",
                "isolation_jets": r"\tau Vs jet",
                "isolation_muons": r"\tau Vs \mu",
        
                "HT": r"HT(j)",
                "ST": r"$ST(\tau, j, f)$",
                "ST_met": r"$ST(\tau, j, f, p_{T}^{miss})$",
                "ST_full": r"$ST(e, \mu, \tau, j, f, p_{T}^{miss})$",
                "ST_met_old":r"$ST(\tau, j, p_{T}^{miss} (old))$",
                "ST_met_top":r"$ST + p_{T}^{miss} (top)$",
                "recoil_pt":  r"$p_T^{miss}(recoil)$ [GeV]",
                "recoil_phi":  r"$\phi(p_T^{miss}(recoil))$"
            }
        }

    def init_sample_map(self) -> Dict[str, str]:
        """Map between sample keys and display names"""
        return {
            "vv": "Diboson",
            "st": "Single Top",
            "wj": r"W$(\ell\nu)$+jets",
            "tt": r"$t\bar{t}$",
            "dy": r"DY$(\ell\ell)$+jets",
            "higgs": "Higgs",
            "qcd": "QCD",
            "SignalTau_300GeV": r"Signal ($m_{\tau}$=300 GeV)",   
            "SignalTau_400GeV": r"Signal ($m_{\tau}$=400 GeV)",               
            "SignalTau_600GeV": r"Signal ($m_{\tau}$=600 GeV)",
            "SignalTau_750GeV": r"Signal ($m_{\tau}$=750 GeV)",               
            "SignalTau_1TeV": r"Signal ($m_{\tau}$=1 TeV)",
            "SignalTau_2TeV": r"Signal ($m_{\tau}$=2 TeV)",
            "SignalTau_3TeV": r"Signal ($m_{\tau}$=3 TeV)",
            "SingleMuon": "Data",
            "SingleElectron": "Data",
            "SingleTau": "Data",
            "MET": "Data"
        }

    def plot(
        self,
        grouped_histos: Dict[str, np.ndarray],
        binning: Union[List[float], np.ndarray],
        feature: str,
        main_bgr_variable: str,
        SF_main_bgr: float,
        events_gev: bool,
        log_scale: bool,
        cms_loc: float,
        y_axis_range: tuple,
        ratio_axis_range: tuple,
        signals: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Main plotting method that orchestrates the full plotting workflow.
        
        Args:
            processed_hists: Dictionary of {sample_name: histogram_values}
            binning: Array of bin edges
            feature: Physics variable being plotted
            **kwargs: Optional plotting parameters
            
        Returns:
            Tuple containing (total_mc, total_main_background) histograms
        """
        # Store configuration
        self.grouped_histos = grouped_histos
        self.binning = np.array(binning)
        self.bin_widths = self.binning[1:] - self.binning[:-1]
        self.feature = feature
        self.main_bgr = main_bgr_variable
        self.SF_main_bgr = SF_main_bgr
        self.log_scale = log_scale
        self.y_axis_range = y_axis_range
        self.ratio_axis_range = ratio_axis_range
        self.signals = signals
        self.cms_loc = cms_loc
        self.events_gev = events_gev

        if self.events_gev:
            self.bin_widths = self.binning[1:] - self.binning[:-1]
        else:
            self.bin_widths =  np.ones(len(self.binning) - 1)
            
        # Create figure and axes
        self.fig, self.axes = self.create_figure()


        self.add_cms_labels()


        
        # Process histograms
        self.total_data, self.total_mc, self.total_main_bgr = self.process_histograms(grouped_histos)

        self.draw_main_components()


        
        # Finalize plot
        self.save_and_show(self.output_dir)
        


    # ------------------------------------------
    #    Create canvas
    # -------------------------------------------
    def create_figure(self) -> Tuple[plt.Figure, Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]]:
        """Create figure with appropriate subplot configuration"""
        if self.signals:
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

    def add_cms_labels(self) -> None:
        """Add CMS experiment labels and luminosity information"""
        ax = self.axes[0] if isinstance(self.axes, tuple) else self.axes
        
        # Get luminosity text
        lumi_text = self.get_lumi_text()
        
        if lumi_text:
            hep.cms.lumitext(lumi_text, fontsize=14, ax=ax)
        
        hep.cms.text("Preliminary", loc=self.cms_loc, fontsize=16, ax=ax)
        


    def get_lumi_text(self) -> str:
        """Generate appropriate luminosity label based on year"""
        if self.combined_2016:
            return "36.3 fb$^{-1}$ (2016, 13 TeV)"
        
        lumi_map = {
            "2017": "41.5 fb$^{-1}$ (2017, 13 TeV)",
            "2018": "59.8 fb$^{-1}$ (2018, 13 TeV)",
            "2016": "16.8 fb$^{-1}$ (2016, 13 TeV)",
            "2016APV": "19.5 fb$^{-1}$ (2016, 13 TeV)"
        }
        
        return lumi_map.get(self.year, "")


       
    # ------------------------------------------
    #    Process data 
    # -------------------------------------------
    def process_histograms(
            self,
            grouped_histos: Dict[str, np.ndarray]
        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process and combine histograms from different samples.
        
        Returns:
            Tuple of (total_mc_histogram, main_background_histogram)
        """
        mcs = []
        main_back = []
        data_hist = []
        
        background_map = {
            "tt": "tt",
            "dy": "DYJetsToLL",
            "st": "SingleTop",
            "vv": "VV",
            "wj": "WJetsToLNu",
            "higgs": "Higgs",
            "qcd": "QCD",
            "data": "Data"
        }
        
        main_background_name = background_map.get(self.main_bgr, "")
        
        for sample, hist in grouped_histos.items():
            if hist is None:
                continue
                
            #if sample in ["SingleMuon", "SingleElectron", "Tau", "MET"]:
            if sample == "data":
                data_hist.append(hist)
                continue
                
                
            # Apply scale factor to main background
            if sample == main_background_name:
                scaled_hist = hist * self.SF_main_bgr
                main_back.append(scaled_hist)
                mcs.append(scaled_hist)
            elif sample not in ["Data", *[f"SignalTau_{m}GeV" for m in [600, 1000, 2000, 3000]]]:
                # Normalize by bin width if requested
                norm_hist = hist / self.bin_widths if self.events_gev else hist
                mcs.append(norm_hist)
        
        # Sum all MC histograms
        total_data = processor.accumulate(data_hist) if data_hist else None
        total_mc = processor.accumulate(mcs) if mcs else None
        total_main_back = processor.accumulate(main_back) if main_back else None

        
        return total_data, total_mc, total_main_back


    def draw_main_components(self) -> None:
        """Draw the primary histogram components (MC, Data, Signals)"""
        
        if isinstance(self.axes, tuple):
            ax, ax_ratio = self.axes
        else:
            ax = self.axes
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

           
    
        # Definir las claves a excluir del stack
        #exclude_keys = ["data"] + [f"SignalTau_{m}GeV" for m in [600, 1000, 2000, 3000]]
        exclude_keys = ["data"] + [key for key in self.grouped_histos if key.startswith("Signal")]

        # Filter MC samples: exclude Data and Signal, and keep only those with non-None histograms
        mc_samples_unsorted = [s for s in self.grouped_histos if s not in exclude_keys and self.grouped_histos[s] is not None]
        
        # Compute integral (total sum of events) for each MC sample
        sample_integrals = {s: self.grouped_histos[s].sum() for s in mc_samples_unsorted}
        
        # Sort MC samples by their integral in ascending order
        # This means samples with fewer events are plotted first (at the bottom of the stack),
        # and samples with more events are plotted last (on top of the stack)
        mc_samples = sorted(sample_integrals, key=sample_integrals.get)
        
        # Extract the histograms and their corresponding colors in the sorted order
        stacked_histos = [self.grouped_histos[s] for s in mc_samples]
        colors = [self.sample_colors[s] for s in mc_samples]


        # Sum all MC histograms bin by bin
        total_mc = np.sum(stacked_histos, axis=0)
        if self.is_signal == False:
            data = self.grouped_histos["data"]
            
        bin_centers = 0.5 * (self.binning[1:] + self.binning[:-1])

        # mc error stat
        stat_bgr_error_down, stat_bgr_error_up = poisson_interval_v2(
            values=total_mc, conf_level=0.95
        )        

        total_bgr_error_down = stat_bgr_error_down
        total_bgr_error_up = stat_bgr_error_up
        
        # Scale MC for events/GeV if applicable
        if self.events_gev:
            scaled_stacked = stacked_histos / self.bin_widths
            error_down = total_bgr_error_down / self.bin_widths
            error_up = total_bgr_error_up / self.bin_widths
        else:
            scaled_stacked = stacked_histos
            error_down = total_bgr_error_down
            error_up = total_bgr_error_up
        
        # Draw MC stack (if available)
        ax.tick_params(axis='both', labelsize=14)
        hep.histplot(
            scaled_stacked,
            bins=self.binning,
            ax=ax,
            histtype="fill",
            stack=True,
            color=colors,
            edgecolor="k",
            linewidth=0.7,
            label=[sample_map.get(s, s) for s in mc_samples]
        )
        
        # Draw MC error bar
        error_down_step = np.repeat(error_down, 2)
        error_up_step = np.repeat(error_up, 2)
        bin_edges_step = np.repeat(self.binning, 2)[1:-1]
        ax.fill_between(
            bin_edges_step,
            error_down_step,
            error_up_step,
            step=None,
            color="lightgray",
            alpha=0.5,
            edgecolor="black",
            hatch="///",
            linewidth=0,
            label="stat unc"
        )
        
        # Draw Data only if it is not a signal
        if "data" in self.grouped_histos and self.grouped_histos["data"] is not None and not self.is_signal:
            if self.events_gev:
                scaled_data = self.grouped_histos["data"] / self.bin_widths
                scaled_errors = np.sqrt(self.grouped_histos["data"]) / self.bin_widths
            else:
                scaled_data = self.grouped_histos["data"]
                scaled_errors = np.sqrt(self.grouped_histos["data"])

            bin_centers = (self.binning[:-1] + self.binning[1:]) / 2
            bin_widths = (self.binning[1:] - self.binning[:-1]) / 2

            ax.errorbar(
                bin_centers,
                scaled_data,
                xerr=bin_widths,
                yerr=scaled_errors,
                fmt='k.',            # black point
                markersize=10,
                linestyle='none',
                capsize=0,           # no caps
                label="Data"
            )

    
        # Draw signal if enabled
        if self.is_signal or self.signals:
            for signal_key in sorted(k for k in self.grouped_histos if k.startswith("SignalTau_")):
                signal_hist = self.grouped_histos[signal_key]
                
                if signal_hist is None:
                    continue

                # Normalize to Events/GeV if needed
                if self.events_gev:
                    signal_hist = signal_hist / self.bin_widths  
        
                hep.histplot(
                    signal_hist,
                    bins=self.binning,
                    ax=ax,
                    histtype="step",
                    color=self.sample_colors.get(signal_key, "r"),  # fallback to red if not in dict
                    linestyle="--",
                    linewidth=2,
                    label=self.sample_map.get(signal_key, signal_key)
                )

            ax.set_xlabel(f"{self.label_map[self.lepton_flavor][self.feature]}", fontsize = 16)
        
                    
    
        ax.set_ylabel("Events/GeV" if self.events_gev else "Events", fontsize=16)
        ax.set_yscale("log" if self.log_scale else "linear")
        ax.set_ylim(self.y_axis_range)
        ax.legend()

        ax.legend(
            loc="upper center",       #  Centered above
            bbox_to_anchor=(0.5, 1.02), 
            ncol=3,                   # Number of columns
            fontsize=12,
            frameon=False
        )

        # Ratio plot
        if ax_ratio is not None and "data" in self.grouped_histos and self.grouped_histos["data"] is not None:

            ax_ratio.tick_params(axis='both', labelsize=14)
            
            # Avoid division by zero
            ratio = np.divide(data, total_mc)

            # Error band only reflects the mc error.
            error_mc_down_ratio = (total_mc -  total_bgr_error_down)/total_mc
            error_mc_up_ratio = (total_bgr_error_up - total_mc)/total_mc

            # Vertical lines in the black point only reflects the data error.
            error_data_down = np.sqrt(data)/data
            error_data_up = np.sqrt(data)/data

            yerr = np.vstack([error_data_down, error_data_up])

            #print(error_mc_down_ratio,  error_mc_up_ratio)
            

            bin_centers = (self.binning[:-1] + self.binning[1:]) / 2
            bin_widths = (self.binning[1:] - self.binning[:-1]) / 2  # half-widths for x error
            
            ax_ratio.errorbar(
                bin_centers,
                ratio,
                xerr=bin_widths,  # Add this line
                yerr=yerr,
                fmt='ko',
                markersize=5,
                capsize=0
            )            

            ax_ratio.fill_between(
                self.binning,
                np.append(1 - error_mc_down_ratio, 1 - error_mc_down_ratio[-1]),
                np.append(1 + error_mc_up_ratio, 1 + error_mc_up_ratio[-1]),
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
            ax_ratio.set_xlabel(f"{self.label_map[self.lepton_flavor][self.feature]}", fontsize = 16)
            ax_ratio.set_ylim(self.ratio_axis_range)
            ax_ratio.grid(True)
          

    
    # --------------------------------
    #   Save pdf file
    # --------------------------------
    def save_and_show(self, output_dir) -> None:
        """Save plot to file and display it"""

        os.makedirs(output_dir, exist_ok=True)

        if self.combined_2016:
            year_pdf_file = "2016_full"
        else:
            year_pdf_file = self.year
            
        output_path = f"{output_dir}/{self.feature}_{year_pdf_file}.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
        plt.show()
        plt.close()
