import os
import math
import json
import numpy as np
import pandas as pd
import mplhep as hep
import matplotlib.pyplot as plt
from collections import defaultdict

# Local libraries
from src.utils_pkl import load_pkl_files
from src.utils_2D_weights import weights_2D
from src.utils_metadata import load_json_files
from src.utils_plot import get_hist, HistogramPlotter
from src.utils_errors import get_table_cutflow_unscaled, compute_eff_cutflow 
from src.utils_plot_2D import get_group_hist2d, plot_2d_hist, get_binning_table 
from src.utils_qcd import QCD_squema_plot, qcd_estimation, get_qcd_estimation, get_qcd_estimation_shape
from src.utils_systematic_variations import load_systematic_variations, load_systematic_variation_per_bgr, process_systematics_table, systematic_error_table_report
from src.utils import ensure_directory, load_all_pickles, load_all_jsons, get_weights, group_samples, get_rename_map


class Plotter:
    def __init__(
        self,
        
        # General configs
        year: str = "2017",                      # Data year to analyze (e.g., "2016", "2016APV", "2017", "2018")
        samples_folder: str = "..",              # Folder with pkl and json files
        output_folder: str = "..",               # Output directory path for saving plots
        lepton_flavor: str = "tau",              # Lepton type to analyze ("tau", "muon", "electron")
        control_region: str = "",

        # 2016APV and 2016 combined
        combined_2016: bool = False,             # Combine 2016 data (pre- and post-VFP) for analysis

        # Merge samples
        merge_samples: bool = True,

        
        # Efficiency and stadistical error
        systematic_error: bool = False,
        normalized_to: str = "sumw",
        stadistical_error_using: str = "cutflow",

        # root files: limit studies
        created_root_files: bool = False,
        root_files_folder: str = "",

        # Signal samples
        is_SR: bool = False,
        signal_superposition: bool = False,
        include_signal_samples: bool = False,

        # QCD estimation
        applied_data_driven: bool = False,
        cr_B_folder: str = "",
        cr_C_folder: str = "",
        cr_D_folder: str = "",

        qcd_shape: str = "cr_b",
        qcd_ratio: str = ["cr_c", "cr_d"],   # cr_c/cr_d
        qcd_ratio_integrated: bool = False
        
    ) -> None:

        # Variables used in the plotter methods
        self.year = year
        self.lepton_flavor = lepton_flavor
        self.combined_2016 = combined_2016
        self.control_region = control_region
    

        
        self.stadistical_error_using = stadistical_error_using
        self.systematic_error = systematic_error


        self.signal = is_SR 
        self.signal_superposition = signal_superposition



        self.output_folder = output_folder
        self.root_files_folder = root_files_folder

        self.applied_data_driven = applied_data_driven
        self.cr_B_folder = cr_B_folder
        self.cr_C_folder = cr_C_folder
        self.cr_D_folder = cr_D_folder

        self.qcd_shape = qcd_shape
        self.qcd_ratio = qcd_ratio
        self.qcd_ratio_integrated = qcd_ratio_integrated

    


        # ------------------------------------------------
        #                  Step 1  
        #      Verify if the sample folders exist
        #   Create output directory and root_file directory
        # ------------------------------------------------

        ensure_directory(samples_folder, must_exist=True, description="Samples directory")
        ensure_directory(output_folder, description="Output directory")

        if created_root_files:
            ensure_directory(root_files_folder, description="ROOT files directory")
        

        # ------------------------------------------------
        #                  Step 2  
        #      Load xsections and luminosities
        # ------------------------------------------------

        try:
            with open("jsons/DAS_xsec.json") as f:
                xsecs = json.load(f)

        
            with open("jsons/luminosity.json") as f:
                luminosity = json.load(f)
        
            if self.combined_2016:
                lumi_2016 = luminosity["2016"]
                lumi_2016APV = luminosity["2016APV"]
            else:
                lumi = luminosity[self.year]
        
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Required JSON file not found: {e}\n"
                "Make sure 'jsons/DAS_xsec.json' and 'jsons/luminosity.json' exist."
            ) from e
        except KeyError as e:
            raise KeyError(
                f"Missing expected key in luminosity.json: {e}\n"
                f"Available keys: {list(luminosity.keys())}"
            ) from e

        self.lumi_map = {
            "2017": "41.5 fb$^{-1}$ (2017, 13 TeV)",
            "2018": "59.8 fb$^{-1}$ (2018, 13 TeV)",
            "2016": "16.8 fb$^{-1}$ (2016, 13 TeV)",
            "2016APV": "19.5 fb$^{-1}$ (2016APV, 13 TeV)"
        }
        
        # ------------------------------------------------
        #     Merge json (metadata) and pkl files
        # ------------------------------------------------
        if merge_samples:

            if self.combined_2016:

                print("\n Loading pkl files for 2016")
                print(f"\n Luminosity {self.lumi_map['2016']}") 

                base_folder = os.path.dirname(samples_folder.rstrip("/"))
                
                samples_folder_2016 = os.path.join(base_folder, "2016")
                samples_folder_2016APV = os.path.join(base_folder, "2016APV")

                # pkl merge
                load_pkl_files(samples_folder = samples_folder_2016)
                print("\n")
                load_json_files(samples_folder= samples_folder_2016)
                

                print("\n")                
                print("Loading pkl files for 2016APV")
                print(f"\n Luminosity {self.lumi_map['2016APV']}") 
                
                load_pkl_files(samples_folder = samples_folder_2016APV)
                print("\n\n")
                load_json_files(samples_folder= samples_folder_2016APV)

                if self.applied_data_driven:

                    folders = {
                        'CR_B': cr_B_folder,
                        'CR_C': cr_C_folder,
                        'CR_D': cr_D_folder,
                    }
                
                    for region, folder in folders.items():
                        print(f"\nProcesing región: {region}")
                        
                        base_folder_region = os.path.dirname(folder.rstrip("/"))
        
                        samples_folder_2016 = os.path.join(base_folder_region, "2016")
                        samples_folder_2016APV = os.path.join(base_folder_region, "2016APV")

                        
                        print(f"\n Loading pkl files for {region} 2016APV")
                        load_pkl_files(samples_folder=samples_folder_2016APV)
                        print("\n\n")
                        load_json_files(samples_folder=samples_folder_2016APV)
                        
                        print(f"\n Loading pkl files for {region} 2016")
                        load_pkl_files(samples_folder=samples_folder_2016)
                        print("\n\n")
                        load_json_files(samples_folder=samples_folder_2016)
    
            else:

                print(f"\n Loading pkl files for {self.year}")
                print(f"\n Luminosity {self.lumi_map[self.year]}") 
                
                load_pkl_files(samples_folder= samples_folder)
                print("\n\n")
                load_json_files(samples_folder= samples_folder)

                if self.applied_data_driven:
                    
                    print(f"\n Loading pkl files for CR_B {self.lumi_map[self.year]}")
                    load_pkl_files(samples_folder= cr_B_folder)
                    print("\n\n")
                    load_json_files(samples_folder= cr_B_folder)

                    print(f"\n Loading pkl files for CR_C {self.lumi_map[self.year]}")
                    load_pkl_files(samples_folder= cr_C_folder)
                    print("\n\n")
                    load_json_files(samples_folder= cr_C_folder)

                    print(f"\n Loading pkl files for CR_D {self.lumi_map[self.year]}")
                    load_pkl_files(samples_folder= cr_D_folder)
                    print("\n\n")
                    load_json_files(samples_folder= cr_D_folder)        
                    
        # ------------------------------------------------
        #     Load pkl, json and normalization
        # ------------------------------------------------        
        if self.combined_2016:

            base_folder = os.path.dirname(samples_folder.rstrip("/"))

            periods = {
                "2016": {
                    "folder": os.path.join(base_folder, "2016"),
                    "lumi": lumi_2016
                },
                "2016APV": {
                    "folder": os.path.join(base_folder, "2016APV"),
                    "lumi": lumi_2016APV
                }
            }

            self.pkl_map = {}
            self.json_map = {}
            self.normalization = {}
            self.sumw = {}


            for suffix, info in periods.items():
                pkl_folder = os.path.join(info["folder"], "summary", "pkl")
                json_folder = os.path.join(info["folder"], "summary", "metadata")
        
                print(f"Reading files for {suffix}: pkl -> {pkl_folder}, json -> {json_folder}")
        
                pkl_map = load_all_pickles(pkl_folder)
                json_map = load_all_jsons(json_folder)
                normalization, sumw = get_weights(
                    luminosity=info["lumi"],
                    xsecs=xsecs,
                    pkls=pkl_map,
                    jsons=json_map,
                    normalized_to=normalized_to
                )
        
                # Agregar sufijo a las claves y combinar
                self.pkl_map.update({f"{k}_{suffix}": v for k, v in pkl_map.items()})
                self.json_map.update({f"{k}_{suffix}": v for k, v in json_map.items()})
                self.normalization.update({f"{k}_{suffix}": v for k, v in normalization.items()})
                self.sumw.update({f"{k}_{suffix}": v for k, v in sumw.items()})

            
            if self.applied_data_driven:
                # --- Diccionarios de CR ---
                cr_folders = {
                    "cr_b": self.cr_B_folder,
                    "cr_c": self.cr_C_folder,
                    "cr_d": self.cr_D_folder
                }
            
                # --- Inicializar resultados ---
                self.pkl_files_qcd = {}
                self.json_files_qcd = {}
                self.normalizations_qcd = {}
                self.sumws_qcd = {}
            
                # --- Periodos ---
                periods = {
                    "2016": {"lumi": lumi_2016},
                    "2016APV": {"lumi": lumi_2016APV}
                }
            
                # --- Iterar sobre cada CR ---
                for cr_name, cr_folder in cr_folders.items():
                    # --- Quitar el año precargado ---
                    cr_base_folder = os.path.dirname(cr_folder.rstrip("/"))
            
                    self.pkl_files_qcd[cr_name] = {}
                    self.json_files_qcd[cr_name] = {}
                    self.normalizations_qcd[cr_name] = {}
                    self.sumws_qcd[cr_name] = {}
            
                    for suffix, info in periods.items():
                        # --- Carpeta específica del CR y periodo ---
                        folder = os.path.join(cr_base_folder, suffix)
                        pkl_folder = os.path.join(folder, "summary", "pkl")
                        json_folder = os.path.join(folder, "summary", "metadata")
            
                        # --- Cargar archivos ---
                        pkls = load_all_pickles(pkl_folder)
                        jsons = load_all_jsons(json_folder)
            
                        # --- Separar Data/MET de MC ---
                        data_keys = [k for k in pkls.keys() if k.lower() in ["met", "data", "met_merged"]]
                        mc_keys = [k for k in pkls.keys() if k not in data_keys]
            
                        # --- Normalización MC ---
                        mc_norm, mc_sumw = get_weights(
                            luminosity=info["lumi"],
                            xsecs=xsecs,
                            pkls={k: pkls[k] for k in mc_keys},
                            jsons={k: jsons[k] for k in mc_keys},
                            normalized_to=normalized_to
                        )
            
                        # --- Agregar sufijos a MC y Data ---
                        mc_norm = {f"{k}_{suffix}": v for k, v in mc_norm.items()}
                        mc_sumw = {f"{k}_{suffix}": v for k, v in mc_sumw.items()}
                        pkls = {f"{k}_{suffix}": v for k, v in pkls.items()}
                        jsons = {f"{k}_{suffix}": v for k, v in jsons.items()}
            
                        # --- Combinar en el CR sin sobrescribir ---
                        self.pkl_files_qcd[cr_name].update(pkls)
                        self.json_files_qcd[cr_name].update(jsons)
                        self.normalizations_qcd[cr_name].update(mc_norm)
                        self.sumws_qcd[cr_name].update(mc_sumw)

        else:

            pkl_folder = os.path.join(samples_folder, "summary", "pkl")
            json_folder = os.path.join(samples_folder, "summary", "metadata")

            print(f" Reading pkl files from: {pkl_folder}")
            print(f" Reading json files from: {json_folder}")
            
            self.pkl_map = load_all_pickles(pkl_folder)
            self.json_map = load_all_jsons(json_folder)
            self.normalization, self.sumw = get_weights(luminosity = lumi, xsecs = xsecs, pkls = self.pkl_map, jsons = self.json_map, normalized_to = normalized_to)


            
            if self.applied_data_driven:

                cr_folders = {
                    "cr_b": self.cr_B_folder,
                    "cr_c": self.cr_C_folder,
                    "cr_d": self.cr_D_folder
                }

                # Diccionarios para almacenar resultados
                self.json_files_qcd = {}
                self.pkl_files_qcd = {}
                self.normalizations_qcd = {}
                self.sumws_qcd = {}


                for cr_name, folder in cr_folders.items():
                    jsons = load_all_jsons(os.path.join(folder, "summary", "metadata"))
                    pkls = load_all_pickles(os.path.join(folder, "summary", "pkl"))
                
                    norm, sumw = get_weights(
                        luminosity=lumi, 
                        xsecs=xsecs, 
                        pkls=pkls, 
                        jsons=jsons, 
                        normalized_to=normalized_to
                    )
                
                    self.json_files_qcd[cr_name] = jsons
                    self.pkl_files_qcd[cr_name] = pkls
                    self.normalizations_qcd[cr_name] = norm
                    self.sumws_qcd[cr_name] = sumw

        # ------------------------------------------------
        #     Grouped samples
        # ------------------------------------------------
        self.grouped_samples = group_samples(self.json_map.keys())


        # -------------------------------------------------
        #  Plot QCD squema
        # -------------------------------------------------
        if self.applied_data_driven:
            QCD_squema_plot(self.control_region, self.qcd_shape)

        # -------------------------------------------------------
        # Remove signal samples: include_signal_samples = False
        # -------------------------------------------------------

        if not include_signal_samples:
            # Obtener todas las claves de señales a eliminar (de cualquier diccionario)
            all_signal_keys = set()
            for d in [self.pkl_map, self.json_map, self.normalization, self.sumw, self.grouped_samples]:
                all_signal_keys.update(k for k in d if k.startswith("Signal"))
        
            # Eliminar esas claves de todos los diccionarios
            for d in [self.pkl_map, self.json_map, self.normalization, self.sumw, self.grouped_samples]:
                for k in all_signal_keys:
                    d.pop(k, None)  # pop con None evita error si la clave no existe
        
            if all_signal_keys:
                print(f"Following signals will be ignored: {sorted(all_signal_keys)}")
        

        """
        if not include_signal_samples:
            for d in [self.pkl_map, self.json_map, self.normalization, self.sumw, self.grouped_samples]:
                keys_to_remove = [k for k in d if k.startswith("Signal")]
                for k in keys_to_remove:
                    d.pop(k)
        """
       
    
    def get_table_cutflow(self, variation, combined_samples=False):

        # ---------------------------
        pd.set_option('display.float_format', '{:.2f}'.format)
    
        result_map, scaled_error_df = compute_eff_cutflow(
            cutflow_table=get_table_cutflow_unscaled(self.json_map, self.stadistical_error_using),
            normalization=self.normalization
        )

        if self.combined_2016:
            allowed_variations = [key for key in self.json_map['ST_tW_top_5f_inclusiveDecays_2016'].keys() if key.startswith("cutflow")]
        else:
            allowed_variations = [key for key in self.json_map['ST_tW_top_5f_inclusiveDecays'].keys() if key.startswith("cutflow")]
            
        print("📋 Available variations:", allowed_variations)
    
        is_cutflow = variation.startswith("cutflow")
        
        cutflow_scaled = {}
    
        try:
            base_cuts = next(
                list(self.json_map[ds][variation].keys()) for ds in self.json_map if variation in self.json_map[ds]
            )
        except StopIteration:
            raise ValueError(f"❌ Ningún dataset contiene la variation '{variation}'")
    
        for dataset in self.json_map:
            norm = float(self.normalization.get(dataset, 1.0))
            cutflow_nominal = self.json_map[dataset].get("cutflow", {})
    
            if dataset in ['SingleElectron', 'SingleMuon', 'Tau', 'MET']:
                print(f"\n🔎 Revisando dataset tipo data: {dataset}")
                scaled = {}
                for cut in base_cuts:
                    value = cutflow_nominal.get(cut)
                    fallback_used = False
    
                    if value is None and "_" in variation:
                        suffix = "_" + variation.split("cutflow")[-1].lstrip("_")
                        if cut.endswith(suffix):
                            cut_base = cut.removesuffix(suffix)
                            value = cutflow_nominal.get(cut_base)
                            fallback_used = True
    
                    if value is not None:
                        try:
                            scaled[cut] = float(value) * norm
                        except (ValueError, TypeError):
                            scaled[cut] = None
                    else:
                        print(f"⚠️  Campo '{cut}' no encontrado en dataset '{dataset}' (tipo data)")
                        scaled[cut] = None
    
                    if fallback_used:
                        print(f"ℹ️  Usando campo análogo '{cut_base}' en lugar de '{cut}' en dataset '{dataset}'")
    
                cutflow_scaled[dataset] = scaled
                continue
    
            cutflow_source = self.json_map[dataset].get(variation, {})
            scaled = {}
    
            for cut in base_cuts:
                value = cutflow_source.get(cut)
                if value is None:
                    base_cut = cut.rsplit("_", 1)[0] if "_" in cut else cut
                    value = cutflow_nominal.get(base_cut)
    
                try:
                    scaled[cut] = float(value) * norm if value is not None else None
                except (ValueError, TypeError):
                    scaled[cut] = None
    
            cutflow_scaled[dataset] = scaled
    
        df = pd.DataFrame.from_dict(cutflow_scaled, orient="index").transpose()
        sumw_row = df.loc["sumw"].copy() if "sumw" in df.index else None
    
        if not is_cutflow:
            if combined_samples:
                grouped_cutflows = defaultdict(lambda: defaultdict(float))
                for group_name, samples in self.grouped_samples.items():
                    for sample in samples:
                        for cut in df.index:
                            value = df.get(sample, {}).get(cut)
                            if value is not None:
                                grouped_cutflows[group_name][cut] += value
    
                df_grouped = pd.DataFrame.from_dict(grouped_cutflows, orient="index").transpose()
                rename_columns = get_rename_map(self.grouped_samples)
                df_grouped = df_grouped.rename(columns=rename_columns)
                df_grouped["Total"] = df_grouped.sum(axis=1, numeric_only=True)
    
                if sumw_row is not None:
                    # Agregar fila sumw para muestras combinadas
                    rename_columns = get_rename_map(self.grouped_samples)
                    sumw_row_grouped = pd.Series(dtype="object")
    
                    for group_name, group_col in rename_columns.items():
                        matching_samples = self.grouped_samples.get(group_name, [])
                        total_sumw = sum([sumw_row[sample] for sample in matching_samples if sample in sumw_row])
                        sumw_row_grouped[group_col] = total_sumw
    
                    bkg_cols = [col for col in sumw_row_grouped.index if not col.startswith("Data")]
                    sumw_row_grouped["Total"] = sum([sumw_row_grouped[col] for col in bkg_cols if pd.notna(sumw_row_grouped[col])])
    
                    df_grouped.loc["sumw"] = sumw_row_grouped
    
                return df_grouped.round(2), scaled_error_df
    
            else:
                if sumw_row is not None:
                    df.loc["sumw"] = sumw_row
                return df.round(2), scaled_error_df

        # --- cutflow con errores ---
        if combined_samples:
            grouped_cutflows = defaultdict(lambda: defaultdict(float))
            grouped_errors = defaultdict(lambda: defaultdict(float))
    
            for group_name, samples in self.grouped_samples.items():
                for sample in samples:
                    for cut in base_cuts:
                        value = cutflow_scaled[sample].get(cut)
                        error = scaled_error_df.get(sample, {}).get(cut)
                        if value is not None:
                            grouped_cutflows[group_name][cut] += value
                        if error is not None:
                            grouped_errors[group_name][cut] += error ** 2
    
            for group_name in grouped_errors:
                for cut in grouped_errors[group_name]:
                    grouped_errors[group_name][cut] = np.sqrt(grouped_errors[group_name][cut])
    
            df_grouped = pd.DataFrame.from_dict(grouped_cutflows, orient="index").transpose()
            error_grouped_df = pd.DataFrame.from_dict(grouped_errors, orient="index").transpose()
    
            rename_columns = get_rename_map(self.grouped_samples)
            df_grouped = df_grouped.rename(columns=rename_columns)
            error_grouped_df = error_grouped_df.rename(columns=rename_columns)
    
            bkg_cols = [col for col in df_grouped.columns if not col.startswith("Data")]
            df_grouped["Total"] = df_grouped[bkg_cols].sum(axis=1)
            error_grouped_df["Total"] = np.sqrt(np.square(error_grouped_df[bkg_cols]).sum(axis=1))
    
            self.df_with_errors = df_grouped.copy()
            for col in df_grouped.columns:
                for cut in df_grouped.index:
                    val = df_grouped.at[cut, col]
                    err = None
                    if cut in error_grouped_df.index and col in error_grouped_df.columns:
                        err = error_grouped_df.at[cut, col]
    
                    if cut == "sumw":
                        self.df_with_errors.at[cut, col] = f"{val:.2f}" if pd.notna(val) else ""
                    elif col.startswith("Data"):
                        self.df_with_errors.at[cut, col] = f"{val:.2f}" if pd.notna(val) else ""
                    elif pd.notna(val) and pd.notna(err):
                        self.df_with_errors.at[cut, col] = f"{val:.2f} ± {err:.2f}"
                    else:
                        self.df_with_errors.at[cut, col] = ""
    
            if sumw_row is not None:
                # Agregar sumw para tabla combinada
                sumw_row_grouped = pd.Series(dtype="object")
                for group_name, group_col in rename_columns.items():
                    matching_samples = self.grouped_samples.get(group_name, [])
                    total_sumw = sum([sumw_row[sample] for sample in matching_samples if sample in sumw_row])
                    sumw_row_grouped[group_col] = total_sumw
    
                bkg_cols = [col for col in sumw_row_grouped.index if not col.startswith("Data")]
                sumw_row_grouped["Total"] = sum([sumw_row_grouped[col] for col in bkg_cols if pd.notna(sumw_row_grouped[col])])
    
                self.df_with_errors.loc["sumw"] = sumw_row_grouped

            # ---- QCD data driven ------
            if self.applied_data_driven:
                self.qcd = qcd_estimation( 
                                    variation = variation,
                                    grouped_samples=self.grouped_samples,
                                    combined_samples=combined_samples, combined_2016=self.combined_2016,
                                    cr_BCD_normalization = self.normalizations_qcd,
                                    cr_BCD_jsons =  self.json_files_qcd,
                                    shape_region=self.qcd_shape, ratio_regions=self.qcd_ratio
                )
                # normalization = self.normalization,
                return self.df_with_errors, scaled_error_df, self.qcd
            else:
                return self.df_with_errors, scaled_error_df  #, self.qcd
    
        else:
            df_with_errors = df.copy()
            for col in df.columns:
                for cut in df.index:
                    val = df.at[cut, col]
                    err = scaled_error_df.get(col, {}).get(cut)
    
                    if cut == "sumw":
                        df_with_errors.at[cut, col] = f"{val:.2f}" if pd.notna(val) else ""
                    elif col.startswith("Data"):
                        df_with_errors.at[cut, col] = f"{val:.2f}" if pd.notna(val) else ""
                    elif pd.notna(val) and pd.notna(err):
                        df_with_errors.at[cut, col] = f"{val:.2f} ± {err:.2f}"
                    else:
                        df_with_errors.at[cut, col] = ""
    
            if sumw_row is not None:
                df_with_errors.loc["sumw"] = sumw_row
    
            return df_with_errors, scaled_error_df

    def get_table_report(self):   
        qcd_estimated = None
        qcd_estimated_error = None
    
        # ==============================
        # QCD data-driven
        # ==============================
        if self.applied_data_driven:
            valor_str = self.qcd['QCD Estimated'].iloc[-1]
            qcd_estimated = float(valor_str.split('±')[0].strip())
            qcd_estimated_error = float(valor_str.split('±')[1].strip())

            ratio_str = self.qcd['Ratio (X/Y)'].iloc[-1]
            ratio_estimated =  float(ratio_str.split('±')[0].strip())
            ratio_estimated_error =  float(ratio_str.split('±')[1].strip())

    
        report_map = {}
    
        # ==============================
        # Cargar yields esperados
        # ==============================
        for sample, info in self.json_map.items():
            n_events = info.get("weighted_final_nevents", None)
            if n_events is None:
                print(f"⚠️  No se encontró 'weighted_final_nevents' para la muestra {sample}")
                continue
    
            norm = self.normalization.get(sample, 1.0)
            expected = float(n_events) * norm
            report_map[sample] = expected
    
        # ==============================
        # Agrupar según self.grouped_samples
        # ==============================
        grouped_report = defaultdict(float)
        for group_name, samples in self.grouped_samples.items():
            for sample in samples:
                if sample in report_map:
                    grouped_report[group_name] += report_map[sample]
    
        # Renombrar con etiquetas legibles
        rename_columns = get_rename_map(self.grouped_samples)
        renamed_grouped_report = {rename_columns.get(k, k): v for k, v in grouped_report.items()}
    
        # ==============================
        # Crear DataFrame principal
        # ==============================
        df_report = pd.DataFrame([renamed_grouped_report], index=["Events"]).transpose()
    
        # Agregar QCD (Data-driven) antes de cualquier fila que empiece con 'Data'
        if qcd_estimated is not None:
            qcd_row_name = "QCD (Data-driven)"
            insert_position = next(
                (i for i, idx in enumerate(df_report.index) if str(idx).startswith("Data")),
                len(df_report)
            )
            formatted_qcd = f"{qcd_estimated:.2f} ± {qcd_estimated_error:.2f}"
            df_report = pd.concat([
                df_report.iloc[:insert_position],
                pd.DataFrame({"Events": [formatted_qcd]}, index=[qcd_row_name]),
                df_report.iloc[insert_position:]
            ])
    
        # ==============================
        # Identificar filas de fondo
        # ==============================
        bkg_rows = [
            idx for idx in df_report.index
            if not str(idx).startswith("Data")
            and idx not in ["Total bgr", "Data/Total bgr"]
            and not str(idx).startswith("Signal")
        ]
        non_bkg_rows = [idx for idx in df_report.index if idx not in bkg_rows]
    
        # ==============================
        # Fila Total (solo fondos)
        # ==============================
        if len(bkg_rows) > 0:
            total = df_report.loc[bkg_rows, "Events"].apply(
                lambda x: float(str(x).split('±')[0].strip()) if isinstance(x, str) else x
            ).sum()
            df_report.loc["Total bgr"] = total
        else:
            df_report.loc["Total bgr"] = 0.0
    
        # ==============================
        # Fila Data/Total
        # ==============================
        data_rows = df_report.index[df_report.index.str.startswith("Data")]
        if len(data_rows) > 0:
            data_total = df_report.loc[data_rows, "Events"].apply(
                lambda x: float(str(x).split('±')[0].strip()) if isinstance(x, str) else x
            ).sum()
            total_val = float(str(df_report.loc["Total bgr", "Events"]).split('±')[0].strip())
            ratio = data_total / total_val if total_val > 0 else float("nan")
            df_report.loc["Data/Total bgr"] = ratio
        else:
            df_report.loc["Data/Total bgr"] = float("nan")
    
        # ==============================
        # Columna contribución (%)
        # ==============================
        contribution = []
        total_val = float(str(df_report.loc["Total bgr", "Events"]).split('±')[0].strip())
        for idx in df_report.index:
            if idx in ["Total bgr", "Data/Total bgr"] or str(idx).startswith("Data") or str(idx).startswith("Signal"):
                contribution.append(float("nan"))
            else:
                contrib_val = df_report.loc[idx, "Events"]
                contrib_val = float(str(contrib_val).split('±')[0].strip()) if isinstance(contrib_val, str) else contrib_val
                contrib = 100 * contrib_val / total_val if total_val > 0 else float("nan")
                contribution.append(contrib)
    
        df_report["Contribution (%)"] = contribution
    
        # ==============================
        # Ordenar por contribución
        # ==============================
        df_bkg_sorted = df_report.loc[bkg_rows].sort_values("Contribution (%)", ascending=False) if bkg_rows else pd.DataFrame()
        df_rest = df_report.loc[non_bkg_rows]
    
        special_rows = []
        for special in ["Total bgr", "Data/Total bgr"]:
            if special in df_report.index:
                special_rows.append(df_report.loc[[special]])
    
        df_report = pd.concat([df_bkg_sorted, df_rest] + special_rows)
    
        # ==============================
        # Añadir errores
        # ==============================
        if self.applied_data_driven:
            cutflow_table, _, _ = self.get_table_cutflow(variation="cutflow", combined_samples=True)
        else:
            cutflow_table, _ = self.get_table_cutflow(variation="cutflow", combined_samples=True)
    
        last_cut = cutflow_table.iloc[-1]
    
        event_strings = []
        errors_for_total = []
        for idx in df_report.index:
            event_val = df_report.loc[idx, "Events"]
    
            if isinstance(event_val, str) and '±' in event_val:
                event_strings.append(event_val)
                err = float(event_val.split('±')[1].strip())
                if idx not in ["Data/Total bgr"] and not str(idx).startswith("Signal"):
                    errors_for_total.append(err)
                else:
                    errors_for_total.append(0.0)
                continue
    
            if idx == "QCD (Data-driven)" and qcd_estimated is not None:
                formatted = f"{qcd_estimated:.2f} ± {qcd_estimated_error:.2f}"
                event_strings.append(formatted)
                errors_for_total.append(qcd_estimated_error)
                continue
    
            if str(idx).startswith("Signal"):
                val_err_str = last_cut.get(idx)
                if isinstance(val_err_str, str) and '±' in val_err_str:
                    v, e = val_err_str.split("±")
                    val_num = float(v.strip())
                    err_val = float(e.strip())
                    formatted = f"{val_num:.2f} ± {err_val:.2f}"
                    errors_for_total.append(0.0)
                else:
                    val_num = float(str(event_val).split('±')[0].strip())
                    formatted = f"{val_num:.2f} ± 0.00"
                    errors_for_total.append(0.0)
                event_strings.append(formatted)
                continue
    
            val_err_str = last_cut.get(idx)
            if isinstance(val_err_str, str) and '±' in val_err_str:
                val_part, err_part = val_err_str.split('±')
                val_num = float(val_part.strip())
                err_val = float(err_part.strip())
                event_strings.append(f"{val_num:.2f} ± {err_val:.2f}")
                errors_for_total.append(err_val)
            else:
                val_num = float(event_val)
                event_strings.append(f"{val_num:.2f}")
                errors_for_total.append(0.0)
    
        df_report["Events"] = event_strings
    
        # ==============================
        # Sumar errores en cuadratura para 'Total bgr'
        # ==============================
        if "Total bgr" in df_report.index:
            total_bkg_val = df_report.loc["Total bgr", "Events"]
            if isinstance(total_bkg_val, str) and '±' in total_bkg_val:
                total_bkg_val = float(total_bkg_val.split('±')[0].strip())
            else:
                total_bkg_val = float(total_bkg_val)
    
            total_bkg_error = (np.array(errors_for_total) ** 2).sum() ** 0.5
            df_report.loc["Total bgr", "Events"] = f"{total_bkg_val:.2f} ± {total_bkg_error:.2f}"
    
        df_report["Contribution (%)"] = df_report["Contribution (%)"].round(2)
        df_report.rename(columns={"Events": "Events ± stat"}, inplace=True)
        df_report.columns.name = "Samples"
    
        # ==============================
        # Calcular Rtt
        # ==============================
        if bkg_rows:
            main_bgr = df_report.loc[bkg_rows, "Contribution (%)"].idxmax()
        else:
            main_bgr = None
    
        data_val, data_err = 0.0, 0.0
        for row in df_report.index[df_report.index.str.startswith("Data")]:
            val = df_report.loc[row, "Events ± stat"]
            if isinstance(val, str) and "±" in val:
                v, e = val.split("±")
                data_val += float(v.strip())
                data_err = np.sqrt(data_err**2 + float(e.strip())**2)
            else:
                data_val += float(str(val).split("±")[0].strip())
    
        if main_bgr is not None:
            main_val = df_report.loc[main_bgr, "Events ± stat"]
            if isinstance(main_val, str) and "±" in main_val:
                v, e = main_val.split("±")
                main_val = float(v.strip())
                main_err = float(e.strip())
            else:
                main_val = float(str(main_val).split("±")[0].strip())
                main_err = 0.0
    
            sum_no_main, err_no_main = 0.0, 0.0
            for row in bkg_rows:
                if row == main_bgr:
                    continue
                val = df_report.loc[row, "Events ± stat"]
                if isinstance(val, str) and "±" in val:
                    v, e = val.split("±")
                    sum_no_main += float(v.strip())
                    err_no_main = np.sqrt(err_no_main**2 + float(e.strip())**2)
                else:
                    sum_no_main += float(str(val).split("±")[0].strip())
    
            N = data_val - sum_no_main
            Rtt = N / main_val if main_val > 0 else float("nan")
            sigma_Rtt = np.sqrt(
                (data_err / main_val) ** 2
                + ((N / (main_val**2)) * main_err) ** 2
                + (err_no_main / main_val) ** 2
            )
    
            rtt_label = f"Rtt ({main_bgr})"
            df_report.loc[rtt_label] = [f"{Rtt:.3f} ± {sigma_Rtt:.3f}", float("nan")]
        else:
            Rtt = float("nan")

        # ==============================
        # Agregar ratio (Data-driven)
        # ==============================
        if self.applied_data_driven and ratio_estimated is not None:
            ratio_row_name = "QCD Ratio (X/Y)"
            formatted_ratio = f"{ratio_estimated:.3f} ± {ratio_estimated_error:.3f}"
            insert_position = next(
                (i for i, idx in enumerate(df_report.index) if str(idx).startswith("Data")),
                len(df_report)
            )
            df_report = pd.concat([
                df_report.iloc[:insert_position],
                pd.DataFrame({"Events ± stat": [formatted_ratio], "Contribution (%)": [float("nan")]}, index=[ratio_row_name]),
                df_report.iloc[insert_position:]
            ])

        # ===================================
        #  Agregar incertidumbre sistematica
        # ====================================
        if self.systematic_error:
            if self.control_region in ["wjets", "signal", "tt", "qcd"]:
                distriution_used = "lepton_met_mass"
            elif self.control_region in ["ztomumu"]:
                distriution_used = "mll"
                
            if self.applied_data_driven:                    
                systematic_errors_table  = systematic_error_table_report(
                        pkls_qcd  = self.pkl_files_qcd,
                        norms_qcd= self.normalizations_qcd,
                        qcd_shape = self.qcd_shape,
                        qcd_ratio = self.qcd_ratio,
                        qcd_ratio_integrated = self.qcd_ratio_integrated,
                        pkls =  self.pkl_map,
                        norms = self.normalization,
                        variable =  distriution_used,
                        binning = [0, 10000000000]
                )
            else:
                systematic_errors_table  = systematic_error_table_report(
                        pkls =  self.pkl_map,
                        norms = self.normalization,
                        variable =  distriution_used,
                        binning = [0, 10000000000]
                )                

            reverse_map = {v: k for k, v in rename_columns.items()}  # Mapa invertido
            reverse_map["QCD (Data-driven)"] = "qcd"
            
            syst_up, syst_down = [], []
            for idx in df_report.index:
                key = reverse_map.get(idx, idx).lower()
                if key in systematic_errors_table:
                    syst_up.append(systematic_errors_table[key]["error_up"])
                    syst_down.append(systematic_errors_table[key]["error_down"])
                elif idx == "Total bgr" and "total_bkg" in systematic_errors_table:
                    syst_up.append(systematic_errors_table["total_bkg"]["error_up"])
                    syst_down.append(systematic_errors_table["total_bkg"]["error_down"])
                else:
                    syst_up.append(float("nan"))
                    syst_down.append(float("nan"))
        
            # Asegurar que no haya duplicados
            for col in ["Syst up", "Syst down"]:
                if col in df_report.columns:
                    df_report = df_report.drop(columns=[col])
        
            # Agregar nuevas columnas
            df_report["Syst up"] = syst_up
            df_report["Syst down"] = syst_down
        
            # Reordenar columnas: insertar justo antes de Contribution
            cols = [c for c in df_report.columns if c not in ["Syst up", "Syst down"]]  # limpiar duplicados
            if "Contribution (%)" in cols:
                contrib_idx = cols.index("Contribution (%)")
                new_order = cols[:contrib_idx] + ["Syst up", "Syst down"] + cols[contrib_idx:]
                df_report = df_report.reindex(columns=new_order)

        
        return df_report, Rtt

        
    # -----------------------------------
    #        1D plot
    # -----------------------------------
        
    def get_plot_report(self, distribution: str, divided_GeV: bool, log: bool, overflow: bool, underflow: bool, main_bgr: str , sf_bgr: float, y_axis, ratio_limits, binning_hist):
        """
        Generate a plot report for a given distribution, including histogram calculation
        and visualization settings.
        
        Args:
            distribution (str): Name of the distribution/variable to plot.
            divided_GeV (bool): If True, convert x-axis units to GeV.
            log (bool): If True, use logarithmic scale on the y-axis.
            overflow (bool): If True, include overflow events in the last bin.
            underflow (bool): If True, include underflow events in the first bin.
            main_bgr (str): Key for the main background sample in the dataset.
            sf_bgr (float): Scale factor to apply to the main background.
            y_axis (str): Label for the y-axis (e.g., "Events", "Arbitrary Units").
            ratio_limits (tuple[float, float]): Min/max limits for the ratio plot (if used).
            binning_hist (np.ndarray): Array defining the bin edges for the histogram.
        
        Returns:
            None: This function generates plots but does not return a value.
        """

        if self.combined_2016:
            keys = [
                key for key in self.pkl_map['TTToSemiLeptonic_2016']['TTToSemiLeptonic'].keys()
                if not key.startswith("weights") and not key.endswith("up") and not key.endswith("down")
            ]
        else:
            keys = [
                key for key in self.pkl_map['TTToSemiLeptonic']['TTToSemiLeptonic'].keys()
                if not key.startswith("weights") and not key.endswith("up") and not key.endswith("down")
            ]

        print(f" Key available in the pkl files: {keys}")
        
        self.distribution = distribution
        self.binning_hist = binning_hist

        self.overflow = overflow
        self.underflow = underflow
        
        # Use get_hist() to process the data
        processed_hists = get_hist(
            feature=self.distribution,
            pkls=self.pkl_map,
            bins=self.binning_hist,
            weights_variation = "weights",
            consider_overflow=self.overflow,
            consider_underflow=self.underflow,
        )


        self.grouped_histos = {}
        
        for category, samples in self.grouped_samples.items():
            weighted_hist = None
    
            for first_sample in samples:
                if first_sample in processed_hists:
                    weighted_hist = np.zeros_like(processed_hists[first_sample])
                    break
    
            if weighted_hist is None:
                print(f"[WARNING] No valid samples found for group '{category}'")
                continue
    
            for sample in samples:
                if sample not in processed_hists:
                    continue
    
                hist = processed_hists[sample]
    
                # Don't apply scaling for data
                if category == "data":
                    weighted_hist += hist
                else:
                    weighted_hist += hist * self.normalization[sample]
    
            self.grouped_histos[category] = weighted_hist


        for sample in processed_hists:
            if sample.startswith("Signal"):
                hist = processed_hists[sample]

                # Quitar extensión .pkl si la tiene
                sample_clean = os.path.splitext(sample)[0]
                norm = self.normalization.get(sample_clean, 1.0)
                
                self.grouped_histos[sample_clean] = hist * norm
                
        if self.applied_data_driven:

            qcd_estimation, qcd_estimation_error = get_qcd_estimation(
                cr_BCD_pkls=self.pkl_files_qcd,
                cr_BCD_normalization=self.normalizations_qcd,
                qcd_shape = self.qcd_shape,
                qcd_ratio = self.qcd_ratio,
                bins=self.binning_hist,
                distribution=self.distribution,
                consider_overflow=self.overflow,
                consider_underflow=self.underflow,
                qcd_ratio_integrated = self.qcd_ratio_integrated,
                combined_2016 = self.combined_2016
            )

            self.grouped_histos['qcd'] = qcd_estimation

        else:
            qcd_estimation_error = {}
            
            

        

        if self.combined_2016:
            # Filtrar solo señales con sufijo 2016 o 2016APV
            signal_keys_to_combine = [k for k in self.grouped_histos.keys() 
                                      if k.startswith("SignalTau_") and ("_2016" in k or "_2016APV" in k)]
        
            combined_signals = {}
            for key in signal_keys_to_combine:
                # Extraer masa
                mass = key.replace("SignalTau_", "").replace("_2016APV", "").replace("_2016", "")
                group_name = f"SignalTau_{mass}"
        
                if group_name not in combined_signals:
                    combined_signals[group_name] = np.zeros_like(self.grouped_histos[key])
        
                combined_signals[group_name] += self.grouped_histos[key]
        
            # Eliminar solo las originales _2016 y _2016APV
            for key in signal_keys_to_combine:
                self.grouped_histos.pop(key)
        
            # Añadir las combinadas
            self.grouped_histos.update(combined_signals)



    
            
        hist_plotter = HistogramPlotter(year=self.year, lepton_flavor = self.lepton_flavor, combined_2016 = self.combined_2016, is_signal = self.signal, output_dir = self.output_folder)

        if main_bgr in self.grouped_histos:
            self.grouped_histos[main_bgr] = (
                self.grouped_histos[main_bgr] * sf_bgr
            )

        if self.systematic_error and self.distribution in ["mll", "lepton_met_mass"]:
            syst_variation = self.get_systematics_per_bin_per_bgr(distribution = self.distribution, binning = self.binning_hist,  include_nominal= True)

            
            hist_plotter.plot(
                grouped_histos=self.grouped_histos,
                binning=self.binning_hist,
                feature=self.distribution,
                main_bgr_variable=main_bgr,
                SF_main_bgr = sf_bgr,
                log_scale=log,
                events_gev = divided_GeV,
                cms_loc = 0.0,
                y_axis_range=y_axis,
                ratio_axis_range = ratio_limits,
                signals = self.signal_superposition,
                denominator = self.df_with_errors,
                include_systematics = self.systematic_error,
                systematics = syst_variation,
                qcd_estimation_error = qcd_estimation_error
            )

        else:
            systematics = {}
            hist_plotter.plot(
                grouped_histos=self.grouped_histos,
                binning=self.binning_hist,
                feature=self.distribution,
                main_bgr_variable=main_bgr,
                SF_main_bgr = sf_bgr,
                log_scale=log,
                events_gev = divided_GeV,
                cms_loc = 0.0,
                y_axis_range=y_axis,
                ratio_axis_range = ratio_limits,
                signals = self.signal_superposition,
                denominator = self.df_with_errors,
                include_systematics = False,
                systematics = systematics,
                qcd_estimation_error = qcd_estimation_error
            )

        self.errors_per_bin = hist_plotter.errors_plot()

        return processed_hists, self.grouped_histos

    def event_table_by_bin(self) -> pd.DataFrame:
        """
        Construye una tabla donde cada fila es un bin (según 'binning') y
        cada columna representa un grupo (ej. 'tt', 'st', etc.), con:
            - Valor central ± error estadístico
            - Si self.systematic_error=True: añade también errores sistemáticos (Up, Down)
            - Si self.systematic_error=False: solo ± estadístico
    
        Además:
        - Columna 'Total MC': suma de todas las muestras excepto 'Data'
        - Columna 'Data / Total MC': razón entre Data y MC
        - Fila 'Total' con sumas por grupo y razón total
        """
    
        # Nombres legibles
        rename_columns = {
            "vv": "Diboson",
            "st": "Single Top",
            "wj": r"WJetToLNu",
            "tt": r"$t\bar{t}$",
            "dy": r"DYJetsToLNu",
            "higgs": "Higgs",
            "qcd": "QCD",
            "data": "Data",
        }
    
        n_bins = len(self.binning_hist) - 1
        bin_labels = [f"{self.binning_hist[i]}-{self.binning_hist[i+1]}_" for i in range(n_bins)]
    
        # --- Valores ---
        df_values = pd.DataFrame(self.grouped_histos, index=bin_labels)
        df_values = df_values.rename(columns=rename_columns)
    
        # --- Errores estadísticos y sistemáticos ---
        syst_variation = None
        if self.systematic_error:
            syst_variation = self.get_systematics_per_bin_per_bgr(
                distribution=self.distribution,
                binning=self.binning_hist,
                include_nominal=False
            )
    
        df_stat, df_syst_up, df_syst_down = {}, {}, {}
    
        for group_name, errors in self.errors_per_bin.items():
            col_name = rename_columns.get(group_name, group_name)  # Nombre legible
            stat_err = np.array(errors)
            df_stat[col_name] = stat_err
    
            # Añadir sistemáticos si corresponde
            if self.systematic_error and syst_variation and group_name in syst_variation:
                syst_df = syst_variation[group_name][1]
                up_list, down_list = [], []
                for bin_label in bin_labels:
                    try:
                        up = syst_df.loc["Total", f"{bin_label}|Nom-Up|"]
                        down = syst_df.loc["Total", f"{bin_label}|Nom-Down|"]
                    except KeyError:
                        up, down = 0.0, 0.0
                    up_list.append(up)
                    down_list.append(down)
                df_syst_up[col_name] = up_list
                df_syst_down[col_name] = down_list
    
        df_stat = pd.DataFrame(df_stat, index=bin_labels)
        if self.systematic_error:
            df_syst_up = pd.DataFrame(df_syst_up, index=bin_labels)
            df_syst_down = pd.DataFrame(df_syst_down, index=bin_labels)
    
        # --- Construir tabla final ---
        df = pd.DataFrame(index=bin_labels)
        for col in df_values.columns:
            if col == "Data":
                df[col] = df_values[col].round(2)  # Data solo valor
            else:
                stat = df_stat[col]
                if self.systematic_error and col in df_syst_up.columns:
                    syst_up = df_syst_up[col]
                    syst_down = df_syst_down[col]
                    df[col] = [
                        f"{val:.2f} ± {st:.2f} (Up: {up:.2f}, Down: {down:.2f})"
                        for val, st, up, down in zip(df_values[col], stat, syst_up, syst_down)
                    ]
                else:
                    df[col] = [f"{val:.2f} ± {st:.2f}" for val, st in zip(df_values[col], stat)]
    
        # --- Totales MC ---
        mc_columns = [c for c in df_values.columns if c != "Data" and not c.startswith("Signal")]
        total_mc = df_values[mc_columns].sum(axis=1)
        total_stat = np.sqrt((df_stat[mc_columns]**2).sum(axis=1))
    
        if self.systematic_error:
            common_cols = [c for c in mc_columns if c in df_syst_up.columns]
    
            total_up = np.sqrt((df_syst_up[common_cols]**2).sum(axis=1))
            total_down = np.sqrt((df_syst_down[common_cols]**2).sum(axis=1))
    
            df["Total MC"] = [
                f"{val:.2f} ± {st:.2f} (Up: {up:.2f}, Down: {down:.2f})"
                for val, st, up, down in zip(total_mc, total_stat, total_up, total_down)
            ]
        else:
            df["Total MC"] = [f"{val:.2f} ± {st:.2f}" for val, st in zip(total_mc, total_stat)]
    
        # --- Data / Total MC ---
        if "Data" in df_values.columns:
            df["Data / Total MC"] = np.where(total_mc > 0, df_values["Data"] / total_mc, np.nan).round(2)
        else:
            df["Data / Total MC"] = np.nan
    
        # --- Fila Total ---
        total_row_vals = df_values.sum(numeric_only=True)
        total_row_stat = np.sqrt((df_stat**2).sum(numeric_only=True))
    
        row = {}
        for col in df_values.columns:
            if col == "Data":
                row[col] = f"{total_row_vals[col]:.2f}"
            else:
                if self.systematic_error and col in df_syst_up.columns:
                    total_row_up = np.sqrt((df_syst_up[col]**2).sum())
                    total_row_down = np.sqrt((df_syst_down[col]**2).sum())
                    row[col] = (
                        f"{total_row_vals[col]:.2f} ± {total_row_stat[col]:.2f} "
                        f"(Up: {total_row_up:.2f}, Down: {total_row_down:.2f})"
                    )
                else:
                    row[col] = f"{total_row_vals[col]:.2f} ± {total_row_stat[col]:.2f}"
    
        if self.systematic_error:
            common_cols = [c for c in mc_columns if c in df_syst_up.columns]
    
            total_row_up = np.sqrt((df_syst_up[common_cols]**2).sum().sum())
            total_row_down = np.sqrt((df_syst_down[common_cols]**2).sum().sum())
    
            row["Total MC"] = (
                f"{total_row_vals[mc_columns].sum():.2f} ± {np.sqrt((total_row_stat[mc_columns]**2).sum()):.2f} "
                f"(Up: {total_row_up:.2f}, Down: {total_row_down:.2f})"
            )
        else:
            row["Total MC"] = (
                f"{total_row_vals[mc_columns].sum():.2f} ± {np.sqrt((total_row_stat[mc_columns]**2).sum()):.2f}"
            )
    
        if "Data" in total_row_vals and total_row_vals[mc_columns].sum() > 0:
            row["Data / Total MC"] = round(total_row_vals["Data"] / total_row_vals[mc_columns].sum(), 2)
        else:
            row["Data / Total MC"] = np.nan
    
        df.loc["Total"] = row
    
        return df





        
    # -----------------------------------
    #        2D plot
    # -----------------------------------
    def get_2D_plot_report(self, X_distribution, X_binning, Y_distribution, Y_binning, background, include_overflow, include_underflow, bin_values):

        self.X_distribution = X_distribution 
        self.X_binning = X_binning
        self.Y_distribution =  Y_distribution
        self.Y_binning = Y_binning
        
        hist =get_group_hist2d(
            group_name=background,
            grouped_samples=self.grouped_samples,
            pkls=self.pkl_map,
            norms=self.normalization,
            feature_x=self.X_distribution,
            feature_y=self.Y_distribution,
            bins_x=self.X_binning,
            bins_y=self.Y_binning,
            include_overflow=include_overflow,
            include_underflow=include_underflow
        )    

        plot_2d_hist(
            hist2d=hist,
            bins_x=X_binning,
            bins_y=Y_binning,
            xlabel=X_distribution,
            ylabel=Y_distribution,
            title=get_rename_map(self.grouped_samples)[background],
            year = self.year,
            show_bin_values = bin_values
        )

        df_table = get_binning_table(hist, X_binning, Y_binning, x_name=X_distribution, y_name=Y_distribution)
        
        return df_table

    # -----------------------------------
    #        2D weights plot
    # -----------------------------------
    def get_2D_weights(self, df_total_bgr, df_main_bgr, df_data, range, filter_parameters, smoothed):
        
        weights = weights_2D(df_total_bgr = df_total_bgr, 
                             df_main_bgr = df_main_bgr,
                             df_data = df_data, 
                             ratio_range = range, 
                             year = self.year, 
                             output_folder = self.output_folder, 
                             filter_params = filter_parameters, 
                             apply_smoothing = smoothed,
                             X_variable = self.X_distribution  ,
                             X_binning = self.X_binning, 
                             Y_variable = self.Y_distribution , 
                             Y_binning = self.Y_binning)

        return weights
    

    # -----------------------------------
    #        Root files
    # -----------------------------------
    def get_root_files(self, CR_name:str, with_plots: bool, distribution: str, binning):

        if self.applied_data_driven:
            event_table, percentages  = load_systematic_variations(
                pkls_qcd = self.pkl_files_qcd, norms_qcd = self.normalizations_qcd,
                qcd_shape = self.qcd_shape, qcd_ratio = self.qcd_ratio, qcd_ratio_integrated = self.qcd_ratio_integrated,
                pkls = self.pkl_map, norms = self.normalization,  
                variable = distribution, binning = binning, with_plots = with_plots, 
                year = self.year, lepton = self.lepton_flavor,
                region = CR_name, output_dir = self.root_files_folder
            )

        else:
            event_table, percentages  = load_systematic_variations(
                pkls = self.pkl_map, norms = self.normalization,  
                variable = distribution, binning = binning, with_plots = with_plots, 
                year = self.year, lepton = self.lepton_flavor,
                region = CR_name, output_dir = self.root_files_folder
            )
        


        return event_table, percentages

    # ------------------------------------
    #   Systematic variation per bin
    # ------------------------------------
    def get_systematics_per_bin_per_bgr(self, distribution, binning, include_nominal):

        """
        if self.control_region in ["wjets", "signal"]:
            backgrounds = ["tt", "dy", "st", "vv", "higgs", "wj", "qcd"]
        else:
            backgrounds = ["tt", "dy", "st", "vv", "higgs", "wj", "qcd"]
        """

        backgrounds = ["tt", "dy", "st", "vv", "higgs", "wj", "qcd"]
    
        results = {}
        for bgr in backgrounds:

            if self.applied_data_driven:
                df_bgr_per_bin = load_systematic_variation_per_bgr(
                    pkls_qcd = self.pkl_files_qcd, norms_qcd = self.normalizations_qcd,
                    qcd_shape = self.qcd_shape, qcd_ratio = self.qcd_ratio, qcd_ratio_integrated = self.qcd_ratio_integrated,
                    pkls = self.pkl_map, norms = self.normalization,  
                    variable = distribution, binning = binning, 
                    bgr = bgr,
                )

            else:
                df_bgr_per_bin = load_systematic_variation_per_bgr(
                    pkls = self.pkl_map, norms = self.normalization,  
                    variable = distribution, binning = binning,
                    bgr = bgr
                )

            # 👇 Saltar si no existe en los pkls
            if df_bgr_per_bin is None:
                print(f"[INFO] Se omite '{bgr}' porque no está en los pkls")
                continue
                
            df_bgr, df_syst_error = process_systematics_table(
                df_bgr_per_bin,
                include_nominal=include_nominal
            )
    
            results[bgr] = (df_bgr, df_syst_error)
    
        return results


    

    # ------------------------------------
    #   QCD estimation: comparison
    # ------------------------------------
    def qcd_estimation_hist(self, cr_x: str, distribution, binning, overflow, underflow):

        cr_x_map = {
            "cr_b":  os.path.join(self.cr_B_folder, "summary", "pkl"),
            "cr_c":  os.path.join(self.cr_C_folder, "summary", "pkl"),
            "cr_d":  os.path.join(self.cr_D_folder, "summary", "pkl")        
        }
        
        qcd_shape = get_qcd_estimation_shape(
                pkls = load_all_pickles(cr_x_map[cr_x]), 
                bins= binning,
                distribution=distribution,
                consider_overflow=overflow,
                consider_underflow=underflow,
                normalization_factors=self.normalization
        )

        return qcd_shape

    


    def qcd_comparison(
        self,
        distribution: str,
        binning: np.ndarray,
        overflow: bool = True,
        underflow: bool = True,
        normalize: bool = False,
        log_scale: bool = True,
        y_range: tuple = None,
    ):
        """
        Grafica estimaciones QCD en CR_B, CR_C, CR_D y devuelve la tabla correspondiente.
    
        Parámetros:
        - distribution: nombre de la variable a graficar (ej. "mt_tau_met")
        - binning: array con los bordes de los bins
        - overflow, underflow: bool, si se consideran over/underflows en la estimación
        - normalize: bool, si True normaliza las distribuciones a unidad
        - log_scale: bool, si True usa escala logarítmica en el eje Y
        - y_range: tuple (min, max) para el eje Y
    
        Retorna:
        - pd.DataFrame con los valores por bin y total
        """
        plt.style.use(hep.style.CMS)
    
        bins = np.array(binning)
    
        # Obtener histogramas
        qcd_map = {
            "CR_B": np.array(self.qcd_estimation_hist("cr_b", distribution, binning, overflow, underflow)),
            "CR_C": np.array(self.qcd_estimation_hist("cr_c", distribution, binning, overflow, underflow)),
            "CR_D": np.array(self.qcd_estimation_hist("cr_d", distribution, binning, overflow, underflow)),
        }
    
        # Normalizar si se solicita
        if normalize:
            for key in qcd_map:
                total = np.sum(qcd_map[key])
                qcd_map[key] = qcd_map[key] / total if total > 0 else qcd_map[key]
    
        # Colores
        colors = {
            "CR_B": "#1f77b4",
            "CR_C": "#2ca02c",
            "CR_D": "#d62728",
        }
    
        # Gráfico
        fig, ax = plt.subplots(figsize=(8, 6))
    
        for region in ["CR_B", "CR_C", "CR_D"]:
            values = qcd_map[region]
            ax.step(
                bins,
                np.append(values, values[-1]),
                where="post",
                color=colors[region],
                linewidth=2,
                label=fr"QCD {region.replace('_', '$_')}$"
            )
    
        ax.set_xlabel(r"$m_T(\tau, p_T^{miss})$ [GeV]", fontsize=14)
        ax.set_ylabel("Events / GeV" if not normalize else "A.U.", fontsize=14)
    
        if log_scale:
            ax.set_yscale("log")
    
        if y_range is not None:
            ax.set_ylim(*y_range)
        else:
            if log_scale:
                ax.set_ylim(1e-1 if normalize else 1, 2 if normalize else 2e4)
    
        ax.set_xlim(bins[0], bins[-1])
    
        hep.cms.text("Preliminary", loc=0, fontsize=16)
        hep.cms.lumitext(self.lumi_map[self.year], fontsize=12)
    
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=3,
            fontsize=12,
            frameon=False
        )
    
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/qcd_comparison_{self.year}.pdf", format="pdf", bbox_inches="tight")
        plt.show()
    
        # Tabla
        bin_labels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    
        df_qcd = pd.DataFrame({
            "Bin": bin_labels,
            "CR_B": qcd_map["CR_B"],
            "CR_C": qcd_map["CR_C"],
            "CR_D": qcd_map["CR_D"],
        })
    
        if normalize:
            total_row = {"Bin": "Total", "CR_B": 1.0, "CR_C": 1.0, "CR_D": 1.0}
        else:
            total_row = {
                "Bin": "Total",
                "CR_B": np.sum(self.qcd_estimation_hist("cr_b", distribution, binning, overflow, underflow)),
                "CR_C": np.sum(self.qcd_estimation_hist("cr_c", distribution, binning, overflow, underflow)),
                "CR_D": np.sum(self.qcd_estimation_hist("cr_d", distribution, binning, overflow, underflow)),
            }
    
        df_qcd = pd.concat([df_qcd, pd.DataFrame([total_row])], ignore_index=True)
    
        return df_qcd

        

    def get_maps(self):
        return self.pkl_map, self.json_map , self.normalization, self.sumw
        
        