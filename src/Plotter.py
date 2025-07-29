import os
import math
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# Local libraries
from src.utils_pkl import load_pkl_files
from src.utils_2D_weights import weights_2D
from src.utils_metadata import load_json_files
from src.utils_plot import get_hist, HistogramPlotter
from src.utils_errors import get_table_cutflow_unscaled, compute_eff_cutflow 
from src.utils_plot_2D import get_group_hist2d, plot_2d_hist, get_binning_table 
from src.utils_qcd import QCD_squema_plot, qcd_estimation, get_qcd_estimation, get_qcd_estimation_shape
from src.utils_systematic_variations import load_systematic_variations, load_systematic_variation_per_bgr
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

        # Distirbution
        distribution_with_obj_level_var: str = "lepton_met_mass",
        
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

        self.distribution_with_obj_level_var = distribution_with_obj_level_var
        
        self.stadistical_error_using = stadistical_error_using

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
            samples_folder_2016 = os.path.join(base_folder, "2016")
            samples_folder_2016APV = os.path.join(base_folder, "2016APV")
        
            pkl_folder_2016APV = os.path.join(samples_folder_2016APV, "summary", "pkl")
            json_folder_2016APV = os.path.join(samples_folder_2016APV, "summary", "metadata")
            
            pkl_folder_2016 = os.path.join(samples_folder_2016, "summary", "pkl")
            json_folder_2016 = os.path.join(samples_folder_2016, "summary", "metadata")


            print(f" Reading pkl files for 2016APV from: {pkl_folder_2016APV}, and 2016 from  {pkl_folder_2016}")
            pkl_map_2016APV = load_all_pickles(pkl_folder_2016APV)
            json_map_2016APV = load_all_jsons(json_folder_2016APV)
            normalization_2016APV, sumw_2016APV = get_weights(luminosity = lumi_2016APV, xsecs = xsecs, pkls = pkl_map_2016APV, jsons = json_map_2016APV, normalized_to = normalized_to)



            print(f"\n Reading json files for 2016APV from: {json_folder_2016APV}, and 2016 from {json_folder_2016}")

            
            pkl_map_2016 = load_all_pickles(pkl_folder_2016)
            json_map_2016 = load_all_jsons(json_folder_2016)
            normalization_2016, sumw_2016 = get_weights(luminosity = lumi_2016, xsecs = xsecs, pkls = pkl_map_2016, jsons = json_map_2016, normalized_to = normalized_to)


            # ---- Combined_map                
            self.pkl_map = {
                f"{k}_2016": v for k, v in pkl_map_2016.items()
            }
            self.pkl_map.update({
                f"{k}_2016APV": v for k, v in pkl_map_2016APV.items()
            })

            self.json_map = {
                f"{k}_2016": v for k, v in json_map_2016.items()
            }
            self.json_map.update({
                f"{k}_2016APV": v for k, v in json_map_2016APV.items()
            })
        
            self.normalization = {
                f"{k}_2016": v for k, v in normalization_2016.items()
            }
            self.normalization.update({
                f"{k}_2016APV": v for k, v in normalization_2016APV.items()
            })
            
            
            self.sumw = {
                f"{k}_2016": v for k, v in sumw_2016.items()
            }
            self.sumw.update({
                f"{k}_2016APV": v for k, v in sumw_2016APV.items()
            })
            

        else:

            pkl_folder = os.path.join(samples_folder, "summary", "pkl")
            json_folder = os.path.join(samples_folder, "summary", "metadata")

            print(f" Reading pkl files from: {pkl_folder}")
            print(f" Reading json files from: {json_folder}")
            
            self.pkl_map = load_all_pickles(pkl_folder)
            self.json_map = load_all_jsons(json_folder)
            self.normalization, self.sumw = get_weights(luminosity = lumi, xsecs = xsecs, pkls = self.pkl_map, jsons = self.json_map, normalized_to = normalized_to)



        # ------------------------------------------------
        #     Grouped samples
        # ------------------------------------------------
        self.grouped_samples = group_samples(self.json_map.keys())

    
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
    
            df_with_errors = df_grouped.copy()
            for col in df_grouped.columns:
                for cut in df_grouped.index:
                    val = df_grouped.at[cut, col]
                    err = None
                    if cut in error_grouped_df.index and col in error_grouped_df.columns:
                        err = error_grouped_df.at[cut, col]
    
                    if cut == "sumw":
                        df_with_errors.at[cut, col] = f"{val:.2f}" if pd.notna(val) else ""
                    elif col.startswith("Data"):
                        df_with_errors.at[cut, col] = f"{val:.2f}" if pd.notna(val) else ""
                    elif pd.notna(val) and pd.notna(err):
                        df_with_errors.at[cut, col] = f"{val:.2f} ± {err:.2f}"
                    else:
                        df_with_errors.at[cut, col] = ""
    
            if sumw_row is not None:
                # Agregar sumw para tabla combinada
                sumw_row_grouped = pd.Series(dtype="object")
                for group_name, group_col in rename_columns.items():
                    matching_samples = self.grouped_samples.get(group_name, [])
                    total_sumw = sum([sumw_row[sample] for sample in matching_samples if sample in sumw_row])
                    sumw_row_grouped[group_col] = total_sumw
    
                bkg_cols = [col for col in sumw_row_grouped.index if not col.startswith("Data")]
                sumw_row_grouped["Total"] = sum([sumw_row_grouped[col] for col in bkg_cols if pd.notna(sumw_row_grouped[col])])
    
                df_with_errors.loc["sumw"] = sumw_row_grouped

            # ---- QCD data driven ------
            if self.applied_data_driven:
                self.qcd = qcd_estimation(json_map =self.json_map, normalization = self.normalization, 
                                    variation = variation ,combined_samples=combined_samples, combined_2016=self.combined_2016,
                                    grouped_samples=self.grouped_samples,
                                    cr_B_folder = self.cr_B_folder, cr_C_folder = self.cr_C_folder, cr_D_folder = self.cr_D_folder,
                                    shape_region=self.qcd_shape, ratio_regions=self.qcd_ratio
                )
                return df_with_errors, scaled_error_df, self.qcd
            else:
                return df_with_errors, scaled_error_df  #, self.qcd
    
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
    
        if self.applied_data_driven:
            valor_str = self.qcd['QCD Estimated'].iloc[-1]
            qcd_estimated = float(valor_str.split('±')[0].strip())
    
        report_map = {}
    
        for sample, info in self.json_map.items():
            n_events = info.get("weighted_final_nevents", None)
            if n_events is None:
                print(f"⚠️  No se encontró 'weighted_final_nevents' para la muestra {sample}")
                continue
    
            norm = self.normalization.get(sample, 1.0)
            expected = float(n_events) * norm
            report_map[sample] = expected
    
        # Agrupar según self.grouped_samples
        grouped_report = defaultdict(float)
        for group_name, samples in self.grouped_samples.items():
            for sample in samples:
                if sample in report_map:
                    grouped_report[group_name] += report_map[sample]
    
        # Renombrar con etiquetas legibles
        rename_columns = get_rename_map(self.grouped_samples)
        renamed_grouped_report = {rename_columns.get(k, k): v for k, v in grouped_report.items()}
    
        # Crear DataFrame principal
        df_report = pd.DataFrame([renamed_grouped_report], index=["Events"]).transpose()
    
        # Agregar QCD (Data-driven) antes de cualquier fila que empiece con 'Data'
        if qcd_estimated is not None:
            qcd_row_name = "QCD (Data-driven)"
            insert_position = next(
                (i for i, idx in enumerate(df_report.index) if str(idx).startswith("Data")),
                len(df_report)
            )
            df_report = pd.concat([
                df_report.iloc[:insert_position],
                pd.DataFrame({"Events": [qcd_estimated]}, index=[qcd_row_name]),
                df_report.iloc[insert_position:]
            ])
    
        # Fila Total (suma de todos los fondos)
        bkg_mask = ~df_report.index.str.startswith("Data") & (df_report.index != "Data/Total bgr")
        total = df_report.loc[bkg_mask, "Events"].sum()
        df_report.loc["Total bgr"] = total
    
        # Fila Data/Total
        data_rows = df_report.index[df_report.index.str.startswith("Data")]
        if len(data_rows) > 0:
            data_total = df_report.loc[data_rows, "Events"].sum()
            ratio = data_total / total if total > 0 else float("nan")
            df_report.loc["Data/Total bgr"] = ratio
        else:
            df_report.loc["Data/Total bgr"] = float("nan")
    
        # Añadir columna de porcentaje de contribución
        contribution = []
        for idx in df_report.index:
            if idx in ["Total bgr", "Data/Total bgr"] or str(idx).startswith("Data"):
                contribution.append(float("nan"))
            else:
                contrib = 100 * df_report.loc[idx, "Events"] / total if total > 0 else float("nan")
                contribution.append(contrib)
    
        df_report["Contribution (%)"] = contribution
    
        # Ordenar solo los fondos por contribución
        bkg_rows = df_report.index[
            (~df_report.index.str.startswith("Data")) &
            (df_report.index != "Total bgr") &
            (df_report.index != "Data/Total bgr")
        ]
        non_bkg_rows = df_report.index.difference(bkg_rows)
    
        df_bkg_sorted = df_report.loc[bkg_rows].sort_values("Contribution (%)", ascending=False)
        df_rest = df_report.loc[non_bkg_rows]
    
        # Concatenar: fondos ordenados + resto sin tocar
        df_report = pd.concat([df_bkg_sorted, df_rest])
    
        # Redondear
        df_report = df_report.round(2)
        df_report.columns.name = "Samples"
    
        return df_report
    
        
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

        if self.signal:
            for sample in processed_hists:
                if sample.startswith("Signal"):
                    hist = processed_hists[sample]
 
                    # Quitar extensión .pkl si la tiene
                    sample_clean = os.path.splitext(sample)[0]
                    norm = self.normalization.get(sample_clean, 1.0)
                    
                    self.grouped_histos[sample_clean] = hist * norm

 
        if self.applied_data_driven:
            qcd_hist = get_qcd_estimation(
                pkls_folder_shape = os.path.join(self.cr_B_folder, "summary", "pkl"),
                pkls_folder_num = os.path.join(self.cr_C_folder, "summary", "pkl"),
                pkls_folder_den = os.path.join(self.cr_D_folder, "summary", "pkl"),
                bins=self.binning_hist,
                distribution=self.distribution,
                consider_overflow=self.overflow,
                consider_underflow=self.underflow,
                normalization_factors=self.normalization,
                ratio_per_bin = self.qcd_ratio_integrated
            )


        
            self.grouped_histos['qcd'] = qcd_hist
            
        hist_plotter = HistogramPlotter(year=self.year, lepton_flavor = self.lepton_flavor, combined_2016 = self.combined_2016, is_signal = self.signal)
     

        
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
        )
            
        return processed_hists, self.grouped_histos


    
    def event_table_by_bin(self) -> pd.DataFrame:
        """
        Construye una tabla donde cada fila es un bin (según 'binning') y
        cada columna representa un grupo (ej. 'tt', 'st', etc.), con el número
        de eventos por bin y grupo.
    
        Agrega:
        - Columna 'Total MC': suma de todas las muestras excepto 'data'
        - Columna 'Data / Total MC': razón entre data y MC
        - Fila 'Total' con sumas por grupo y razón total
    
        Returns:
            pd.DataFrame: Tabla con una fila por bin, columnas por grupo,
                          y columnas adicionales de totales y razones.
        """
    
        # Mapeo de nombres legibles para columnas
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
        bin_labels = [f"[{self.binning_hist[i]}, {self.binning_hist[i+1]})" for i in range(n_bins)]
    
        data = {}
    
        for group_name, bin_array in self.grouped_histos.items():
            if len(bin_array) != n_bins:
                raise ValueError(f"El grupo '{group_name}' tiene {len(bin_array)} valores, se esperaban {n_bins}.")
            data[group_name] = bin_array
    
        df = pd.DataFrame(data, index=bin_labels)
    
        # Renombrar columnas si están en el mapeo
        df = df.rename(columns=rename_columns)
    
        # Detectar nombre de columna de data (renombrada)
        data_col = rename_columns.get("data", "data")
    
        # Columnas MC (excluyendo 'Data')
        mc_columns = [col for col in df.columns if col != data_col]
    
        # Total MC
        df["Total MC"] = df[mc_columns].sum(axis=1)
    
        # Ratio Data / Total MC, evitando división por cero
        df["Data / Total MC"] = np.where(
            df["Total MC"] > 0, df[data_col] / df["Total MC"], np.nan
        )
    
        # Fila Total
        total_row = df.sum(numeric_only=True)
        if total_row["Total MC"] > 0:
            total_row["Data / Total MC"] = total_row[data_col] / total_row["Total MC"]
        else:
            total_row["Data / Total MC"] = np.nan
    
        df.loc["Total"] = total_row
    
        return df.round(2)
        

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
    def get_root_files(self, CR_name:str, with_plots: bool):
        event_table, percentages  = load_systematic_variations(self.pkl_map, self.normalization, self.distribution, self.binning_hist, with_plots, self.year, self.lepton_flavor, CR_name, self.root_files_folder)

        
        return event_table, percentages

    # ------------------------------------
    #   Systematic variation per bin
    # ------------------------------------
    def get_systematics_per_bin_per_bgr(self, bgr):
        
        df_bgr = load_systematic_variation_per_bgr(self.pkl_map, self.normalization, self.distribution, self.binning_hist, bgr)

        return df_bgr


    # ------------------------------------
    #   QCD estimation
    # ------------------------------------
    def qcd_estimation(self, cr_x: str):

        cr_x_map = {
            "cr_b":  os.path.join(self.cr_B_folder, "summary", "pkl"),
            "cr_c":  os.path.join(self.cr_C_folder, "summary", "pkl"),
            "cr_d":  os.path.join(self.cr_D_folder, "summary", "pkl")        
        }
        
        qcd_shape = get_qcd_estimation_shape(
                pkls = load_all_pickles(cr_x_map[cr_x]), 
                bins=self.binning_hist,
                distribution=self.distribution,
                consider_overflow=self.overflow,
                consider_underflow=self.underflow,
                normalization_factors=self.normalization
        )

        return qcd_shape

    
    def get_maps(self):
        return self.pkl_map, self.json_map , self.normalization, self.sumw


        
        