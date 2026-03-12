import os
import ROOT
import json
import array
import pickle
import uproot
import numpy as np
import pandas as pd
from pathlib import Path


# Plots
import mplhep as hep
import boost_histogram as bh
import matplotlib.pyplot as plt

# Local 
from src.utils_hist import load_pkl_files
from src.utils_metadata import load_json_files

# Utils
from src.utils_plot import HistogramPlotter
from src.utils import group_samples, get_rename_map
from src.utils_errors import compute_statistical_error, compute_systematic_error


class Plotter:
    
    def __init__(
        self,
        input_path: str = "", 
        output_path: str = "", 
        year: str = "",
        lepton_flavor: str = "",
        combine_2016: bool = False
    ):

        output_dir = Path(output_path)

        self.year=year
        self.lepton_flavor=lepton_flavor
        self.combine_2016=combine_2016
        self.DATA = [
            "MET", 
            "SingleMuon", 
            "SingleElectron", 
            "Tau"
        ]
        self.syst_object_level_names = [
            'CMS_scale_e_13TeV_tt_tau_2017Up', 'CMS_scale_e_13TeV_tt_tau_2017Down', 
            'CMS_scale_m_tt_tau_2017Up', 'CMS_scale_m_tt_tau_2017Down', 
            'CMS_scale_t_tt_tau_2017Up', 'CMS_scale_t_tt_tau_2017Down', 
            'CMS_scale_met_unclustered_energy_tt_tau_2017Up', 'CMS_scale_met_unclustered_energy_tt_tau_2017Down', 
            'CMS_scale_j_tt_tau_2017Up', 'CMS_scale_j_tt_tau_2017Down', 
            'CMS_res_j_tt_tau_2017Up', 'CMS_res_j_tt_tau_2017Down', 
            'CMS_scale_fj_tt_tau_2017Up', 'CMS_scale_fj_tt_tau_2017Down', 
            'CMS_res_fj_tt_tau_2017Up', 'CMS_res_fj_tt_tau_2017Down'
        ]
        self.BCD_REGIONS = ["cr_b", "cr_c", "cr_d"]

        if not output_dir.exists():    
            print(":::::::::::::::::::::::::::::::::::")
            print("Loading pkl files")
            load_pkl_files(input_path, output_path)
    
            print("\n Loading json files")
            load_json_files(input_path, output_path)
            print(":::::::::::::::::::::::::::::::::::")

        else:
            print(f"Output path already exists → skipping load pkl and json files")


        with open("json_files/luminosity.json") as f:
            self.lumi = json.load(f)[year]


        with open("json_files/DAS_xsec.json") as f:
            self.xsec = json.load(f)

        
        # ====================================================
        #  Load pkl information
        # ====================================================
        self.hist = {}
    
        for name in os.listdir(output_path):
            if name.endswith(".pkl"):   
                path = os.path.join(output_path, name)
        
                with open(path, "rb") as f:
                    key = Path(name).stem  # Remove .pkl in the name
                    self.hist[key] = pickle.load(f)
    
        # ====================================================
        #  Load metadata information
        # ===================================================
        metadata_path = os.path.join(output_path, "metadata")
        self.metadata = {}
    
        for name in os.listdir(metadata_path):
            if name.endswith(".json"):
                path = os.path.join(metadata_path, name)
        
                with open(path, "r", encoding="utf-8") as f:
                    key = Path(name).stem  # Remove .json in the name
                    self.metadata[key] = json.load(f)

        # ===================================================
        #           Auxiliar variables
        # ===================================================

        # xsec * Lumi / sumw
        self.normalization = {}
        self.sumw = {}
        
        for sample in self.metadata:
            if sample in self.DATA: 
                self.sumw[sample] = 1.0
                self.normalization[sample] = 1.0
                continue
            self.sumw[sample] = self.metadata[sample]['main']['sumw']
            self.normalization[sample] = (self.xsec[sample] * self.lumi)/self.sumw[sample]

        # group samples
        self.grouped_samples = group_samples(self.metadata.keys())
        self.get_rename_map = get_rename_map(self.grouped_samples)

        # ===================================================
        #          Identify syst variations
        # ===================================================
        self.has_syst = True

        for sample, content in self.hist.items():
        
            if sample in self.DATA:
                continue   
        
            keys = content['hist']['main'].keys()
        
            sample_has_syst = any(
                k.endswith("Up") or k.endswith("Down")
                for k in keys
            )
        
            self.has_syst &= sample_has_syst

        if self.has_syst:
            
            self.syst_by_sample = {}

            for sample, content in self.hist.items():
                
                if sample in self.DATA:
                    continue
            
                keys = content['hist']['main'].keys()
            
                up   = [k for k in keys if k.endswith("Up")]
                down = [k for k in keys if k.endswith("Down")]
            
                self.syst_by_sample[sample] = {
                    "Up": up,
                    "Down": down
                }
    
    
    def get_cutflow(
        self, 
        CR: str = "main", 
        table_name: str = "cutflow", 
        group: bool = True
    ):

        # Set two decimal places of precision.
        pd.options.display.float_format = '{:.2f}'.format  

        # =============================================
        #  1. Create the dataframe in pandas
        # =============================================
        base_table = "cutflow"  if CR == "main" else f"cutflow_{CR}"# In data is always used cutflow
        
        df_cutflow = pd.DataFrame({
            name: info["main"].get(table_name, info[CR][base_table])
            for name, info in self.metadata.items()
        })

        
        # ===================================================
        #  2. Normalize by lumi and xsec
        # ===================================================
        for bgr in df_cutflow.columns:
            df_cutflow[bgr] = df_cutflow[bgr] * self.normalization[bgr]


        # ===================================================
        #  3.  Group samples, e.g:
        #  tt = TTToSemiLeptonic + TTToHadronic + TTTo2L2Nu
        # ===================================================
        if group:
            df_cutflow = pd.DataFrame({
                group: df_cutflow[bgr].sum(axis=1)
                for group, bgr in self.grouped_samples.items()
            })

        # ===================================================
        #  4. Calculate total bgr
        # ===================================================
        columns_to_sum = [
            column_name for column_name in df_cutflow.columns
            if column_name != "data" and not column_name.startswith("Signal")
        ]

        df_cutflow["total"] = df_cutflow[columns_to_sum].sum(axis=1)

        
        # ===================================================
        # 5 Ratio Data / Total BKG
        # ===================================================
        df_cutflow["Data/Bgr"] = df_cutflow["data"].astype(float) / df_cutflow["total"].astype(float)


        # ===================================================
        # 6 Rename columns
        # ===================================================
        df_cutflow = df_cutflow.rename(columns=self.get_rename_map)
        
        
        return df_cutflow

    def get_histograms(
        self, 
        distribution: str = "lepton_met_mass",
        cut: str = "fail_top_tagger",
        CR: str = "main",              # cr_b; cr_c; cr_d
        variation: str = "nominal",    # 
        main_bgr: list = "tt",
        SF: dict[str, float] = {"tt": 1.0, "st": 1.0, "wj": 1.0, "dy": 1.0, "vv": 1.0, "qcd": 1.0},

        output_dir:str =  "",
        
        # Boolean
        events_per_gev: bool = True,
        log_scale: bool = True,
        include_signals: bool = False,
        include_syst_var: bool = False,

        # Style
        cms_loc_text: float = 0.0,
        y_range=(0,10000),
        ratio_range=(0.5,1.5),

    ):

        
        histos = {}
        sumw = {}
        for sample in self.hist:
            histos[sample] = {}

            if sample not in self.DATA:
                base = self.hist[sample]['hist'][CR][variation]
                
                if variation != "nominal":
                    if "top_tagger" not in cut:
                        
                        raise ValueError(
                            f"No systematics are defined for the cut '{cut}'. "
                        )
                        
                    else:
                        base = base['hist']

            else:
                base = self.hist[sample]['hist'][CR]["nominal"]
                
        
            if variation in self.syst_object_level_names and sample not in self.DATA:
                cut_tmp = "pass_top_tagger_(AK4_JER)"
            else:
                cut_tmp = cut

            norm = np.abs(base['top_tagger_cases'][cut_tmp]['edges'][0])

            # Binning
            edges = base[distribution][cut_tmp]["edges"] / norm
        
            # Sum of weights
            sumw_val = base[distribution][cut_tmp]["sumw"]

        
            if sample not in self.DATA:
                eff_denominator = self.sumw[sample]
                
                histos[sample]["sumw"] = (
                    (sumw_val / eff_denominator) *
                    self.xsec[sample] *
                    self.lumi
                )

                # Statistical error
                histos[sample]["stat_error"] = compute_statistical_error(
                    numerator = sumw_val, 
                    denominator = eff_denominator
                ) * self.xsec[sample] * self.lumi

                # Systematic error
                if self.has_syst and "top_tagger" in cut:
                    histos[sample]["syst_error"] = {}
                    up_var_tmp, down_var_tmp = compute_systematic_error(
                        histos = self.hist[sample]['hist'][CR], 
                        list_syst_var = self.syst_by_sample[sample], 
                        distribution = distribution, 
                        cut = cut
                    ) 

                    histos[sample]["syst_error"]["Up"] =  up_var_tmp * self.xsec[sample] * self.lumi
                    histos[sample]["syst_error"]["Down"] = down_var_tmp * self.xsec[sample] * self.lumi
                    
            
                
            else:
                histos[sample]["sumw"] = sumw_val
                    
                    
                
                
        grouped_histos = {}
        grouped_histos_stat_error = {}
        grouped_histos_syst_error = {}

        for group, samples in self.grouped_samples.items():
            grouped_histos[group] = 0
            grouped_histos_stat_error[group] = None
            grouped_histos_syst_error[group] = {}
            grouped_histos_syst_error[group]['Up'] = None
            grouped_histos_syst_error[group]['Down'] = None
            
            for sample in samples:
                grouped_histos[group] += histos[sample]["sumw"]

                if sample not in self.DATA:
                    if grouped_histos_stat_error[group] is None:
                        grouped_histos_stat_error[group] = histos[sample]["stat_error"]**2

                        # Only pass and fail top tagger cuts have available de systematic variations
                        if "top_tagger" in cut:
                            contain_syst_variation = True
                            
                            if grouped_histos_syst_error[group]['Up'] is None:
                                grouped_histos_syst_error[group]['Up'] = histos[sample]["syst_error"]["Up"]**2
                                grouped_histos_syst_error[group]['Down'] = histos[sample]["syst_error"]["Down"]**2
                                
                        else:
                            contain_syst_variation = False
                                               
                    else:
                        
                        grouped_histos_stat_error[group] = grouped_histos_stat_error[group] + histos[sample]["stat_error"]**2  
                        if contain_syst_variation:
                            grouped_histos_syst_error[group]['Up'] = grouped_histos_syst_error[group]['Up']  + histos[sample]["syst_error"]["Up"]**2
                            grouped_histos_syst_error[group]['Down'] = grouped_histos_syst_error[group]['Down'] + histos[sample]["syst_error"]["Down"]**2  
                        
                            


            if group != "data":
                grouped_histos_stat_error[group] = np.sqrt(grouped_histos_stat_error[group])
                if contain_syst_variation:
                    grouped_histos_syst_error[group]['Up'] = np.sqrt(grouped_histos_syst_error[group]['Up'])
                    grouped_histos_syst_error[group]['Down'] = np.sqrt(grouped_histos_syst_error[group]['Down'])
                    

            
        # ===========================================
        #  Include BCD information
        # ===========================================
        qcd_data_driven = all(
            all(region in proc["hist"] for region in self.BCD_REGIONS)
            for proc in self.hist.values()
        )

        if qcd_data_driven and CR not in self.BCD_REGIONS: 
            histos_data_driven = {}
            normalization_data_driven = {}

            for cr in self.BCD_REGIONS: 
                histos_data_driven[cr] = {}
                normalization_data_driven[cr] = {}

                for sample in self.hist:
                    histos_data_driven[cr][sample] = {}

                    if sample in self.DATA: 
                        normalization_data_driven[cr][sample] =  1.0

                    else:
                        normalization_data_driven[cr][sample] =  (self.xsec[sample] * self.lumi) / self.hist[sample]['hist'][cr][variation]["sumw_all_weights"]

                    histos_data_driven[cr][sample]["sumw"] = (self.hist[sample]['hist'][cr][variation][distribution][f"{cut}_{cr}"]["sumw"]) * normalization_data_driven[cr][sample]
                
                
            grouped_histos_data_driven = {}
            for cr in self.BCD_REGIONS: 
                grouped_histos_data_driven[cr] = {}
                for group, samples in self.grouped_samples.items():
                    grouped_histos_data_driven[cr][group] = 0
                    first_sample = samples[0]

                    for sample in samples:
                        grouped_histos_data_driven[cr][group] += histos_data_driven[cr][sample]["sumw"]
                        
            for cr in grouped_histos_data_driven:
                grouped_histos_data_driven[cr]["total"] = 0
        
                for bgr in grouped_histos_data_driven[cr]:
                    if bgr in ["data"]:
                        qcd_data = grouped_histos_data_driven[cr][bgr]
                        continue
                    if bgr.startswith("Signal"):
                        continue
                    elif bgr == "total":
                        continue
                    grouped_histos_data_driven[cr]["total"] +=  grouped_histos_data_driven[cr][bgr]
                grouped_histos_data_driven[cr]["qcd"] =  qcd_data - grouped_histos_data_driven[cr]["total"]    
                
            grouped_histos['qcd'] =  grouped_histos_data_driven['cr_b']["qcd"] * (grouped_histos_data_driven['cr_c']["qcd"]/grouped_histos_data_driven['cr_d']["qcd"])

            # Revisar estadisticos de QCD data-driven
            grouped_histos_stat_error["qcd"] = 0.1 * grouped_histos['qcd'] 
            grouped_histos_syst_error["qcd"] = {}
            grouped_histos_syst_error["qcd"]['Up'] = 0.1 * grouped_histos['qcd'] 
            grouped_histos_syst_error["qcd"]['Down'] = 0.1 * grouped_histos['qcd'] 


            
            for cr in grouped_histos_data_driven:
                print(f" ::::::::::::::::::::: {cr} :::::::::::::::::::::::: ")
                print(np.sum(grouped_histos_data_driven[cr]["qcd"]))
                print(grouped_histos_data_driven[cr]["qcd"])
                print((np.sum(grouped_histos_data_driven[cr]["qcd"])/np.sum(grouped_histos_data_driven[cr]["data"]))*100)
                
            
            
            
        # ===========================================
        #  Create stack
        # ===========================================        
        hist_plotter = HistogramPlotter(
            year=self.year, 
            lepton_flavor = self.lepton_flavor, 
            combined_2016 = self.combine_2016, 
            output_dir = output_dir
        )


        # Scale mc contribution
        for bgr in grouped_histos:
            if bgr != "data":
                print(
                    f"Background: {bgr:10s} | "
                    f"binning: {grouped_histos[bgr]}"
                    f"Events: {np.sum(grouped_histos[bgr]):8.2f} | "
                    f"Stat: {np.linalg.norm(grouped_histos_stat_error[bgr]):8.2f} | "
                    f"Syst Up: {np.linalg.norm(grouped_histos_syst_error[bgr]['Up']):8.2f} | "
                    f"Syst Down: {np.linalg.norm(grouped_histos_syst_error[bgr]['Down']):8.2f}"
                )
                if not bgr.startswith("Signal"):
                    grouped_histos[bgr] = SF[bgr] * grouped_histos[bgr]
                
            else:
                print(
                    f"Background: {bgr:10s} | "
                    f"Events: {np.sum(grouped_histos[bgr]):8.2f} | "
                )

            print("\t\t")
 
        

        
        hist_plotter.plot(
            # Histograms
            distribution = distribution,
            grouped_histos = grouped_histos,
            binning = edges,
            main_bgr = main_bgr,
        
            # Presentation
            events_gev = events_per_gev,
            log_scale = log_scale,
            cms_loc = cms_loc_text,
            y_axis_range =  y_range,
            ratio_axis_range = ratio_range, 
        
            # Include signals
            signals = include_signals,
        
            # Systematic variations
            include_systematics = include_syst_var,
            stat_error = grouped_histos_stat_error,
            syst_error = grouped_histos_syst_error
            
        )

        return grouped_histos, grouped_histos_stat_error #, grouped_histos_syst_error
    
    def get_2D_histograms(self, bgr, ST_bins, nj_bins):
        hist_2D = {}
        for sample in self.hist:
            hist_2D[sample] = {}
            edges_nj = self.hist[sample]['hist']['main']['ttbar_boost_weight']['nj_edges']/self.hist[sample]['hist']['main']['ttbar_boost_weight']['nj_edges'][1]
            edges_ST = self.hist[sample]['hist']['main']['ttbar_boost_weight']['ST_edges']/self.hist[sample]['hist']['main']['ttbar_boost_weight']['nj_edges'][1]
            
            hist_2D[sample]["sumw"] = self.hist[sample]['hist']['main']['ttbar_boost_weight']['hist'] * self.normalization[sample]
                
                
               
        grouped_histos = {}
        grouped_histos["total"] = 0  


        for group, samples in self.grouped_samples.items():
            grouped_histos[group] = 0
            
            for sample in samples:
                grouped_histos[group] += hist_2D[sample]["sumw"]
                
                if sample not in self.DATA:
                    grouped_histos["total"] += hist_2D[sample]["sumw"]    

        
        def rebin2d_with_flow(H, old_x, old_y, new_x, new_y):
            """
            Rebin 2D histogram conserving counts.
            Underflow → first bin
            Overflow → last bin
            """
        
            Hnew = np.zeros((len(new_x)-1, len(new_y)-1))
        
            for i in range(len(old_x)-1):
                for j in range(len(old_y)-1):
        
                    content = H[i, j]
                    if content == 0:
                        continue
        
                    cx = 0.5 * (old_x[i] + old_x[i+1])
                    cy = 0.5 * (old_y[j] + old_y[j+1])
        
                    ix = np.searchsorted(new_x, cx) - 1
                    iy = np.searchsorted(new_y, cy) - 1
        
                    # underflow
                    if ix < 0:
                        ix = 0
                    if iy < 0:
                        iy = 0
        
                    # overflow
                    if ix >= Hnew.shape[0]:
                        ix = Hnew.shape[0] - 1
                    if iy >= Hnew.shape[1]:
                        iy = Hnew.shape[1] - 1
        
                    Hnew[ix, iy] += content
        
            return Hnew

        def histogram2d_to_table(H, xbins, ybins):
            rows = []
        
            for i in range(len(xbins) - 1):
                for j in range(len(ybins) - 1):
        
                    rows.append({
                        "ST_min": xbins[i],
                        "ST_max": xbins[i+1],
                        "nj_min": ybins[j],
                        "nj_max": ybins[j+1],
                        "events": H[i, j]
                    })
        
            return pd.DataFrame(rows)
    
        H_rebinned = rebin2d_with_flow(
            grouped_histos[bgr],
            edges_ST,
            edges_nj,
            ST_bins,
            nj_bins
        )

        table = histogram2d_to_table(
            H_rebinned,
            ST_bins,
            nj_bins
        )

        plt.figure(figsize=(10, 7))
        im = plt.pcolormesh(
            ST_bins, 
            nj_bins, 
            H_rebinned.T, 
            shading="auto", 
            cmap="viridis" # Puedes cambiar el mapa de colores
        )

        # Añadimos los números
        for i in range(len(ST_bins)-1):
            for j in range(len(nj_bins)-1):
                val = H_rebinned[i, j]
                
                # Solo dibujamos si el valor es mayor a 0 (opcional, para limpiar el gráfico)
                if val > 0:
                    # Calculamos el centro de la celda
                    x_center = 0.5 * (ST_bins[i] + ST_bins[i+1])
                    y_center = 0.5 * (nj_bins[j] + nj_bins[j+1])
                    
                    # Lógica de color: texto negro en celdas claras, blanco en oscuras
                    color = "white" if val < (H_rebinned.max() * 0.5) else "black"
                    
                    plt.text(
                        x_center, 
                        y_center, 
                        f"{val:.1f}", 
                        ha="center", 
                        va="center", 
                        color=color,
                        fontsize=8,
                        fontweight='bold'
                    )        

        plt.xlabel("ST [GeV]", fontsize=12)
        plt.ylabel("N jets", fontsize=12)
        plt.title(f"Distribution for group: {bgr}", fontsize=14)
        plt.colorbar(im, label="Events")
        
        plt.tight_layout()
        plt.show() 

        return table

    
    def obtain_root_files(self, output_dir, CR, distribution, cut):
        """
        Genera archivos ROOT compatibles con Combine usando PyROOT.
        Evita archivos corruptos (zombies) al cerrar correctamente los streams de EOS.
        """
        sample_map = {
            "tt": f"tt_{CR}_{self.lepton_flavor}_{self.year}",           
            "wj": f"WJetToLNu_{CR}_{self.lepton_flavor}_{self.year}",           
            "dy": f"DYJetsToLNu_{CR}_{self.lepton_flavor}_{self.year}",           
            "vv": f"Diboson_{CR}_{self.lepton_flavor}_{self.year}",           
            "st": f"SingleTop_{CR}_{self.lepton_flavor}_{self.year}",           
            "higgs": f"Higgs_{CR}_{self.lepton_flavor}_{self.year}",   
            "qcd": f"QCD_{CR}_{self.lepton_flavor}_{self.year}",
            "data": f"data_obs_{CR}_{self.lepton_flavor}_{self.year}",
            "SignalTau_300GeV": f"M_300_{CR}_{self.lepton_flavor}_{self.year}",
            "SignalTau_400GeV": f"M_400_{CR}_{self.lepton_flavor}_{self.year}",
            "SignalTau_600GeV": f"M_600_{CR}_{self.lepton_flavor}_{self.year}",
            "SignalTau_750GeV": f"M_750_{CR}_{self.lepton_flavor}_{self.year}",
            "SignalTau_1000GeV": f"M_1000_{CR}_{self.lepton_flavor}_{self.year}",
            "SignalTau_1500GeV": f"M_1500_{CR}_{self.lepton_flavor}_{self.year}",
            "SignalTau_2000GeV": f"M_2000_{CR}_{self.lepton_flavor}_{self.year}",
            "SignalTau_3000GeV": f"M_3000_{CR}_{self.lepton_flavor}_{self.year}"
        }
    
        os.makedirs(output_dir, exist_ok=True)
    
        # Contenedores para agrupar
        grouped_var = {}
        grouped_edges = {} 
    
        for group, samples in self.grouped_samples.items():
            grouped_var[group] = {}
            grouped_edges[group] = {}
            
            for sample in samples:
                # Acceso a los datos originales (asumiendo que vienen de un procesador de Python/Coffea)
                base = self.hist[sample]['hist']['main']
                
                for var in base:
                    if var == 'ttbar_boost_weight':
                        continue
                    
                    # --- Lógica de extracción de datos ---
                    if var == "nominal":
                        current_hist = base[var][distribution][cut]
                        norm_val = np.abs(base[var]['top_tagger_cases'][cut]['edges'][0])
                    else:
                        cut_key = next((k for k in base[var]['hist'][distribution] if k.startswith(cut)), None)
                        if cut_key is None: continue
                        
                        current_hist = base[var]['hist'][distribution][cut_key]
                        norm_val = np.abs(base[var]['hist']['top_tagger_cases'][cut_key]['edges'][0])
    
                    sumw = current_hist['sumw'] * self.normalization[sample]
                    edges = current_hist['edges'] / norm_val
    
                    # --- Acumulación por Grupo ---
                    if var not in grouped_var[group]:
                        grouped_var[group][var] = sumw.copy()
                        grouped_edges[group][var] = edges # Guardar bordes una vez
                    else:
                        grouped_var[group][var] += sumw
    
        # =========================================
        #  Escritura de archivos usando PyROOT
        # =========================================
        for bgr in grouped_var:
            if bgr not in sample_map: continue
            
            output_path = os.path.join(output_dir, f"{sample_map[bgr]}.root")
            
            # Abrir archivo con PyROOT
            f_out = ROOT.TFile.Open(output_path, "RECREATE")
            
            for var in grouped_var[bgr]:
                # 0. Limpiamos la variable name en cada vuelta para evitar "fantasmas" del pasado
                name = ""
    
                # 1. Caso DATA nominal
                if bgr == "data" and var == "nominal":
                    name = f"data_obs_{CR}_{self.lepton_flavor}_{self.year}_nom"

                # 2. Caso Background/Signal nominal
                elif bgr != "data" and var == "nominal":
                    name = f"{sample_map[bgr]}_nom"

                # 3. Si la variable ya contiene el CR, la usamos tal cual
                elif CR in var:
                    name = var

                # 4. Si no tiene el CR, aplicamos el reemplazo del sufijo
                else:

                    suffix_to_find = f"_{self.year}"
                    new_suffix = f"_{CR}_{self.lepton_flavor}_{self.year}"
                    
                    if suffix_to_find in var:
                        name = var.replace(suffix_to_find, new_suffix)
                    else:
                        name = var
                    

                # Datos del histograma
                data_points = grouped_var[bgr][var]
                bins_edges = grouped_edges[bgr][var]
                n_bins = len(data_points)
    
                # Crear TH1D (necesita un array de tipo double para los edges)
                edges_array = array.array('d', bins_edges)
                h = ROOT.TH1D(name, name, n_bins, edges_array)
                
                # Llenar el histograma
                for i in range(n_bins):
                    # ROOT usa índice 1 para el primer bin
                    h.SetBinContent(i + 1, data_points[i])
                    # Opcional: h.SetBinError(i + 1, np.sqrt(variance[i])) si tienes varianza
    
                # Verificación de seguridad
                if h.Integral() < 0:
                    print(f"⚠️ Warning: {name} en {bgr} tiene integral negativa.")
    
                # Escribir objeto en el archivo
                h.Write()
            
            # Cerrar archivo ROOT (Vital para evitar el estado zombie)
            f_out.Close()
            print(f"✅ Archivo generado exitosamente: {output_path}")

        return grouped_var
    

    def aux_func(self):
        return self.lumi, self.xsec, self.metadata, self.hist, self.grouped_samples, self.normalization
        