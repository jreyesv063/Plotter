import os
import numpy as np
import pandas as pd
import mplhep as hep
from coffea import processor
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union, Tuple

from src.intervals import  poisson_interval_v2
from src.utils_errors import calc_bin_eff_error

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

            # Señales: colores distintos y contrastantes
            "SignalTau_300GeV": "#00CED1",  # DarkTurquoise
            "SignalTau_400GeV": "#9400D3",  # DarkViolet
            "SignalTau_600GeV": "#FF4500",  # OrangeRed
            "SignalTau_750GeV": "#1E90FF",  # DodgerBlue
            "SignalTau_1000GeV": "#32CD32", # LimeGreen
            "SignalTau_1500GeV": "#FFD700", # Gold
            "SignalTau_2000GeV": "#FF69B4", # HotPink
            "SignalTau_3000GeV": "#00FA9A", # MediumSpringGreen
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
                "bjet_phi": "$\phi(bjets)$",             
                "bjet_eta": r"$\eta$(bjets)",                   
                "jet_eta": r"$\eta$(Jet$_{0}$)",
                "jet_phi": r"$\phi$(Jet$_{0}$)",
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
                "Z_gen_num": "n[Z(gen-level)]",

                "ST_met": r"$ST(\mu, j, f, p_{T}^{miss})$",
                "ST": r"$ST(\mu, j, f)$",    
                "ST_full": r"$ST(e, \mu, \tau, j, f, p_{T}^{miss})$",       

                "recoil_phi":  r"$\phi(p_T^{miss}(recoil))$",     
                "njets_no_top_tagger": r"$N(jets-no top)$",     
                "njets_full":  r"$N(j + f + b)$",                 
                
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
                "lepton_met_phi": r"$\Delta \phi(\tau, p_T^{miss})$",
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
            "SignalTau_1000GeV": r"Signal ($m_{\tau}$=1.0 TeV)",
            "SignalTau_1500GeV": r"Signal ($m_{\tau}$=1.5 TeV)",
            "SignalTau_2000GeV": r"Signal ($m_{\tau}$=2.0 TeV)",
            "SignalTau_3000GeV": r"Signal ($m_{\tau}$=3.0 TeV)",
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
        signals: bool,
        denominator,
        include_systematics,
        systematics,
        qcd_estimation_error
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
        self.qcd_estimation_error = qcd_estimation_error

        self.include_systematics = include_systematics
        if self.include_systematics:
            self.systematics = systematics

        # Denominator to calculate the stadistical error
        rename_map = {
            "DrellYan+jets": "DYJetsToLL",
            "W+jets": "WJetsToLNu",
        }
        
        self.denominator = denominator.iloc[0].rename(index=rename_map)

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
    
        # Excluir Data y Señales del stack
        exclude_keys = ["data"] + [key for key in self.grouped_histos if key.startswith("Signal")]
    
        # Filtrar MC válidos
        mc_samples_unsorted = [
            s for s in self.grouped_histos
            if s not in exclude_keys and self.grouped_histos[s] is not None
        ]
        
        # Ordenar por integral
        sample_integrals = {s: self.grouped_histos[s].sum() for s in mc_samples_unsorted}
        mc_samples = sorted(sample_integrals, key=sample_integrals.get)
        
        # Preparar histos y colores
        stacked_histos = [self.grouped_histos[s] for s in mc_samples]
        colors = [self.sample_colors[s] for s in mc_samples]
    
        # -------------------------------------
        # Estadístico: error de MC
        # -------------------------------------
        mapped_denominator_sumw = {}
        for short, long in sample_map.items():
            if long in self.denominator:
                mapped_denominator_sumw[short] = self.denominator[long]
            elif short == "qcd":
                # Preliminar version
                data_val = self.denominator.get("Data (MET)", 0.0)
                total_val = self.denominator.get("Total", 0.0)
                mapped_denominator_sumw[short] = abs(data_val - total_val)
            else:
                mapped_denominator_sumw[short] = 0.0
    
        self.stat_errors = {"bkg": None}
        for sample, hist in self.grouped_histos.items():
            if sample == "data":
                continue
            denom = mapped_denominator_sumw.get(sample, 0.0)
            eff_err = [calc_bin_eff_error(num, denom) for num in hist]

               
            tot_err = [denom * de for de in eff_err]

            if sample == "qcd" and self.qcd_estimation_error is not None and len(self.qcd_estimation_error) > 0:
                self.stat_errors[sample] = self.qcd_estimation_error
            else:
                self.stat_errors[sample] = np.array(tot_err)
                
            if self.stat_errors["bkg"] is None:
                self.stat_errors["bkg"] = np.array(tot_err) ** 2
            else:
                self.stat_errors["bkg"] += np.array(tot_err) ** 2
        self.stat_errors["bkg"] = np.sqrt(self.stat_errors["bkg"])
    
        # --------------------------------------
        # Total MC y errores
        # --------------------------------------
        total_mc = np.sum(stacked_histos, axis=0)
        bin_centers = 0.5 * (self.binning[1:] + self.binning[:-1])
        # Stadistical error       
        stat_bgr_error_down, stat_bgr_error_up = np.subtract(total_mc,self.stat_errors["bkg"]) , np.add(total_mc,self.stat_errors["bkg"])
        
        if self.include_systematics:
            syst_total_bgr = {}
        
            for process, (_, df_total) in self.systematics.items():
                # df_total es el segundo DataFrame de la tupla
                if "Total" in df_total.index:
                    syst_total_bgr[process] = df_total.loc["Total"]
        
            syst_bgr_error_up = {}
            syst_bgr_error_down = {}
            
            for process, series in syst_total_bgr.items():
                up_vals, down_vals = [], []
            
                for key in series.index:
                    if key.endswith("_nominal") and not key.startswith("Total"):
                        bin_prefix = key.replace("_nominal", "")
        
                        up = series[f"{bin_prefix}_|Nom-Up|"]
                        down = series[f"{bin_prefix}_|Nom-Down|"]
        
                        up_vals.append(up)
                        down_vals.append(down)
        
                syst_bgr_error_up[process] = up_vals
                syst_bgr_error_down[process] = down_vals
        
            # --- Calcular total en cuadratura ---
            # Número de bins = longitud de cualquier proceso
            n_bins = len(next(iter(syst_bgr_error_up.values())))
            up_total, down_total = [], []
        
            for i in range(n_bins):
                up_sq = sum((syst_bgr_error_up[p][i])**2 for p in syst_bgr_error_up)
                down_sq = sum((syst_bgr_error_down[p][i])**2 for p in syst_bgr_error_down)
                up_total.append(np.sqrt(up_sq))
                down_total.append(np.sqrt(down_sq))
        
            syst_bgr_error_up["total"] = up_total
            syst_bgr_error_down["total"] = down_total

            # ============================================
            # COmbinación estadisticos + systematicos
            # ============================================
            
            # Estadisticos
            stat_err_up = stat_bgr_error_up - total_mc
            stat_err_down = total_mc - stat_bgr_error_down

            # Systematicos
            syst_up_total = syst_bgr_error_up["total"]
            syst_down_total = syst_bgr_error_up["total"]
            
            total_err_up = np.sqrt(stat_err_up**2 + np.array(syst_up_total)**2)
            total_err_down = np.sqrt(stat_err_down**2 + np.array(syst_down_total)**2)
            
            total_bgr_error_up = total_mc + total_err_up
            total_bgr_error_down = total_mc - total_err_down


        else:
            total_bgr_error_down, total_bgr_error_up = stat_bgr_error_down, stat_bgr_error_up

        
        if self.events_gev:
            scaled_stacked = stacked_histos / self.bin_widths
            error_down = total_bgr_error_down / self.bin_widths
            error_up = total_bgr_error_up / self.bin_widths
        else:
            scaled_stacked = stacked_histos
            error_down = total_bgr_error_down
            error_up = total_bgr_error_up
    
        # --- Dibujar MC stack ---
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
    
        # --- Error MC ---
        error_down_step = np.repeat(error_down, 2)
        error_up_step = np.repeat(error_up, 2)
        bin_edges_step = np.repeat(self.binning, 2)[1:-1]

        if self.include_systematics:
            error_label = "stat + syst"
        else:
            error_label = "stat"
            
        ax.fill_between(
            bin_edges_step,
            error_down_step,
            error_up_step,
            color="lightgray",
            alpha=0.5,
            edgecolor="black",
            hatch="///",
            linewidth=0,
            label=f"{error_label} unc"
        )
    
        # --- Data (si existe) ---
        if "data" in self.grouped_histos and self.grouped_histos["data"] is not None:
            data = self.grouped_histos["data"]
            if self.events_gev:
                scaled_data = data / self.bin_widths
                scaled_errors = np.sqrt(data) / self.bin_widths
            else:
                scaled_data = data
                scaled_errors = np.sqrt(data)
    
            bin_centers = (self.binning[:-1] + self.binning[1:]) / 2
            bin_widths = (self.binning[1:] - self.binning[:-1]) / 2
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
    
        # --- Señales (si existen) ---
        signal_keys = [k for k in self.grouped_histos if k.startswith("SignalTau_")]
        for signal_key in sorted(signal_keys):
            signal_hist = self.grouped_histos[signal_key]
            if signal_hist is None:
                continue
            if self.events_gev:
                signal_hist = signal_hist / self.bin_widths
            hep.histplot(
                signal_hist,
                bins=self.binning,
                ax=ax,
                histtype="step",
                color=self.sample_colors.get(signal_key, "r"),
                linestyle="--",
                linewidth=2,
                label=self.sample_map.get(signal_key, signal_key)
            )
    
        # --- Estilo ---
        ax.set_xlabel(f"{self.label_map[self.lepton_flavor][self.feature]}", fontsize=16)
        ax.set_ylabel("Events/GeV" if self.events_gev else "Events", fontsize=16)
        ax.set_yscale("log" if self.log_scale else "linear")
        ax.set_ylim(self.y_axis_range)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            fontsize=12,
            frameon=False
        )
    
        # --- Ratio Data/MC (si hay Data) ---
        if ax_ratio is not None and "data" in self.grouped_histos and self.grouped_histos["data"] is not None:
            ratio = np.divide(data, total_mc, out=np.zeros_like(data, dtype=float), where=total_mc != 0)
            error_mc_down_ratio = (total_mc - total_bgr_error_down) / total_mc
            error_mc_up_ratio = (total_bgr_error_up - total_mc) / total_mc
            error_data_down = np.sqrt(data) / data
            error_data_up = np.sqrt(data) / data
            yerr = np.vstack([error_data_down, error_data_up])
    
            bin_centers = (self.binning[:-1] + self.binning[1:]) / 2
            bin_widths = (self.binning[1:] - self.binning[:-1]) / 2
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
            ax_ratio.set_xlabel(f"{self.label_map[self.lepton_flavor][self.feature]}", fontsize=16)
            ax_ratio.set_ylim(self.ratio_axis_range)
            ax_ratio.grid(True)

          
    # --------------------------------
    #  Systematic error
    # --------------------------------
    def errors_plot(self):
        return self.stat_errors
        
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
