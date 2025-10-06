import os
import ROOT
import numpy as np
import pandas as pd
import mplhep as hep
from array import array
from typing import Optional
import matplotlib.pyplot as plt
from collections import defaultdict
from src.utils import group_samples
from collections import OrderedDict


def make_histograms(
    pkls_qcd: dict = None,
    norms_qcd: dict = None,
    qcd_shape: str = "",
    qcd_ratio: list[str] = None,
    qcd_ratio_integrated: bool = False,
    pkls: dict = None,
    norms: dict = None, 
    variable: str = "",
    binning: list[float] = None,
):
    histograms = variation_histograms(pkls, norms, variable, binning)
    
    if pkls_qcd is not None:
        qcd_histos = build_qcd_datadriven(
            pkls = pkls_qcd, 
            norms = norms_qcd, 
            qcd_shape = qcd_shape,
            qcd_ratio = qcd_ratio,
            qcd_ratio_integrated = qcd_ratio_integrated,                             
            variable = variable, 
            binning = binning, 
        )
    
        histograms["qcd"] = qcd_histos

    return histograms

def systematic_error_table_report(    
    pkls_qcd: dict = None,
    norms_qcd: dict = None,
    qcd_shape: str = "",
    qcd_ratio: list[str] = None,
    qcd_ratio_integrated: bool = False,
    pkls: dict = None,
    norms: dict = None, 
    variable: str = "",
    binning: list[float] = None
):
    # 1. Construir histograms
    histograms = make_histograms(
        pkls_qcd = pkls_qcd,
        norms_qcd = norms_qcd,
        qcd_shape = qcd_shape,
        qcd_ratio = qcd_ratio,
        qcd_ratio_integrated =  qcd_ratio_integrated, 
        pkls = pkls,
        norms = norms, 
        variable = variable,
        binning = binning,
    )

    
    # 2. Calcular errores sistemáticos (1 bin)
    syst_errors = {}

    for bkg, content in histograms.items():
        nominal = content.get("nominal")
        if nominal is None:
            continue

        # Si es array, asumimos un solo bin → cogemos el valor total
        nominal_val = float(np.sum(nominal))

        up_sq, down_sq = 0.0, 0.0

        for var, arr in content.get("variations_up", {}).items():
            var_val = float(np.sum(arr))
            diff = np.abs((var_val - nominal_val))   
            up_sq += diff**2

        for var, arr in content.get("variations_down", {}).items():
            var_val = float(np.sum(arr))
            diff = np.abs((var_val - nominal_val))
            down_sq += diff**2

        syst_errors[bkg] = {
            "nominal": nominal_val,
            "error_up": np.sqrt(up_sq),
            "error_down": np.sqrt(down_sq)
        }

    return syst_errors

    
    
def load_systematic_variations(
    pkls_qcd: dict = None,
    norms_qcd: dict = None,
    qcd_shape: str = "",
    qcd_ratio: list[str] = None,
    qcd_ratio_integrated: bool = False,
    pkls: dict = None,
    norms: dict = None,
    with_plots: bool = False,
    variable: str = "",
    binning: list[float] = None,
    region: str = "",
    lepton: str = "",
    year: str = "",
    output_dir: str = "",
):  

    histograms = make_histograms(
        pkls_qcd = pkls_qcd,
        norms_qcd = norms_qcd,
        qcd_shape = qcd_shape,
        qcd_ratio = qcd_ratio,
        qcd_ratio_integrated =  qcd_ratio_integrated, 
        pkls = pkls,
        norms = norms, 
        variable = variable,
        binning = binning,
    )


    # --- Si 'data' no existe, crear histograma vacío ---
    if "data" not in histograms or histograms["data"].get("nominal") is None:
        print("⚠️ 'data' no encontrada. Creando ROOT con histograma vacío...")
        empty_data_hist = {
            "nominal": np.zeros(len(binning)-1),
            "variations_up": {},
            "variations_down": {}
        }
        histograms["data"] = empty_data_hist

    # --- Opcional: dibujar histogramas ---
    if with_plots:
        plot_variation_histograms(histograms, binning, output_dir=output_dir, log=False, year = year)

    # --- Guardar todos los histogramas en ROOT ---
    save_histograms_to_root(histograms, binning,
                            output_dir=output_dir,
                            region=region,
                            lepton=lepton,
                            year=year)

    # --- Generar tablas ---
    event_table = get_variation_event_table(histograms)
    df_deviations = get_total_relative_deviation_all(histograms, binning)

    return event_table, df_deviations


def load_systematic_variation_per_bgr(
    pkls_qcd: dict = None,
    norms_qcd: dict = None,
    qcd_shape: str = "",
    qcd_ratio: list[str] = None,
    qcd_ratio_integrated: bool = False,
    pkls: dict = None,
    norms: dict = None,
    variable: str = "",
    binning: list[float] = None,
    bgr: str = ""
):
    
    histograms = make_histograms(
        pkls_qcd = pkls_qcd,
        norms_qcd = norms_qcd,
        qcd_shape = qcd_shape,
        qcd_ratio = qcd_ratio,
        qcd_ratio_integrated =  qcd_ratio_integrated, 
        pkls = pkls,
        norms = norms, 
        variable = variable,
        binning = binning,
    )
    
    
    if bgr not in histograms:
        print(f"[INFO] Background '{bgr}' no está en histograms_dict, se omite.")
        return None

    df_bgr = get_binwise_table_for_sample(histograms, binning, bgr)

    return df_bgr




def variation_histograms(pkls, norm, distribution, binning):
    """
    Construye histogramas agrupados por grupo para variaciones a nivel de objeto.
    Incluye underflow en el primer bin y overflow en el último bin.
    """

    # =====================================================
    # Revisamos si hay muestras de _2016 y _2016APV
    has_2016APV = any(k.endswith("_2016APV") for k in pkls.keys())
    has_2016 = any(k.endswith("_2016") for k in pkls.keys())
    combined_2016 = has_2016APV & has_2016
    # ======================================================
        
    bins = binning
    list_group_samples = group_samples(pkls)

    grouped_object_variations = {}
    variation_map = {}

    for sample, arrays in pkls.items():
    
        # Verifica que tenga lo mínimo necesario
        if any(key.endswith("_2016APV") for key in pkls.keys()):
            sample_tmp = sample
            sample = sample.rsplit("_", 1)[0]
        else:
            sample_tmp = sample

        if distribution not in arrays[sample] or "weights" not in arrays[sample]:
            continue

        variation_map[sample_tmp] = {}

        # Primero, guarda el nominal si las longitudes coinciden
        mass = arrays[sample][distribution]
        weights = arrays[sample]["weights"]
        if len(mass) == len(weights):
            variation_map[sample_tmp]["nominal"] = {
                distribution: mass,
                "weights": weights
            }
    
        # Luego recorre variaciones
        for key in arrays[sample].keys():
            if key.startswith("weights_"):
                var_suffix = key.replace("weights_", "")
                weights = arrays[sample][key]
                mass_key = f"{distribution}_{var_suffix}"
                mass = arrays[sample].get(mass_key, arrays[sample][distribution])
    
                if len(mass) == len(weights):
                    variation_map[sample_tmp][var_suffix] = {
                        distribution: mass,
                        "weights": weights
                    }

    # Función auxiliar para histogramas con under/overflow
    def hist_with_flow(values, weights, bins):
        hist, _ = np.histogram(values, bins=bins, weights=weights)
        # underflow
        underflow = np.sum(weights[values < bins[0]])
        # overflow
        overflow  = np.sum(weights[values > bins[-1]])
        hist[0]  += underflow
        hist[-1] += overflow
        return hist

    # Rellenar los histogramas por grupo
    for group, samples in list_group_samples.items():
        grouped_object_variations[group] = {
            "nominal": np.zeros(len(bins) - 1),
            "variations_up": {},
            "variations_down": {}
        }

        for sample in samples:
            if sample not in variation_map or sample not in norm:
                continue

            for variation, arrs in variation_map[sample].items():
                values = arrs[distribution]
                weights = arrs["weights"] * norm[sample]

                hist = hist_with_flow(values, weights, bins)

                if variation == "nominal":
                    grouped_object_variations[group]["nominal"] += hist
                elif variation.endswith("Up") or variation.endswith("up"):
                    key = variation.replace("Up", "")
                    grouped_object_variations[group]["variations_up"].setdefault(key, np.zeros(len(bins) - 1))
                    grouped_object_variations[group]["variations_up"][key] += hist
                elif variation.endswith("Down") or variation.endswith("down"):
                    key = variation.replace("Down", "")
                    grouped_object_variations[group]["variations_down"].setdefault(key, np.zeros(len(bins) - 1))
                    grouped_object_variations[group]["variations_down"][key] += hist

    # Crear total_bkg (suma de todos menos data y signal)
    grouped_object_variations["total_bkg"] = {
        "nominal": np.zeros(len(bins) - 1),
        "variations_up": {},
        "variations_down": {}
    }

    for group, content in grouped_object_variations.items():
        if group in ["data", "signal", "total_bkg"]:
            continue
        grouped_object_variations["total_bkg"]["nominal"] += content["nominal"]

        for key, hist in content["variations_up"].items():
            grouped_object_variations["total_bkg"]["variations_up"].setdefault(key, np.zeros(len(bins) - 1))
            grouped_object_variations["total_bkg"]["variations_up"][key] += hist

        for key, hist in content["variations_down"].items():
            grouped_object_variations["total_bkg"]["variations_down"].setdefault(key, np.zeros(len(bins) - 1))
            grouped_object_variations["total_bkg"]["variations_down"][key] += hist

    return grouped_object_variations




    
def plot_variation_histograms(histograms, binning, output_dir="plots", log=False, x_label="Variable", title_prefix="", year = ""):
    png_folder = os.path.join(output_dir, "png_files")
    os.makedirs(png_folder, exist_ok=True)
    bin_centers = 0.5 * (np.array(binning[:-1]) + np.array(binning[1:]))

    sample_map = {
        "tt": "tt",
        "st": "Single Top",
        "vv": "Diboson",
        "higgs": "Higgs",
        "dy": "DYJetsToLNu",
        "wj": "WJetToLNu",
        "qcd": "QCD",
        "total_bkg": "Total MC",
        "data": "Data"
    }

    for sample, contents in histograms.items():
        variations_up = contents.get("variations_up", {})
        variations_down = contents.get("variations_down", {})
        nominal = contents.get("nominal", None)

        if not variations_up and not variations_down and nominal is None:
            continue

        plt.figure(figsize=(20, 8))
        hep.style.use("CMS")

        # === Dibujar Nominal ===
        handles = []
        labels = []

        if nominal is not None:
            h_nominal, = plt.step(bin_centers, nominal, where="mid", color="black", linewidth=1.5)
            handles.append(h_nominal)
            labels.append("Nominal")

            # Rellenar un segundo espacio vacío para que Nominal ocupe toda la fila
            handles.append(plt.plot([], [])[0])
            labels.append("")

        # === Agrupar variaciones por base ===
        variation_pairs = {}
        for var in set(list(variations_up.keys()) + list(variations_down.keys())):
            base = var.replace("_up", "").replace("_down", "").replace("Up", "").replace("Down", "").rstrip("_")
            if base not in variation_pairs:
                variation_pairs[base] = {}
            if var in variations_up:
                variation_pairs[base]["Up"] = variations_up[var]
            if var in variations_down:
                variation_pairs[base]["Down"] = variations_down[var]

        # === Dibujar variaciones emparejadas ===
        for base, pair in sorted(variation_pairs.items()):
            # Si no hay una de las dos, poner un dummy
            if "Up" in pair:
                h_up, = plt.step(bin_centers, pair["Up"], where="mid", linestyle="--")
            else:
                h_up, = plt.plot([], [], linestyle="--")
            if "Down" in pair:
                h_down, = plt.step(bin_centers, pair["Down"], where="mid", linestyle="-.")
            else:
                h_down, = plt.plot([], [], linestyle="-.")

            handles.extend([h_up, h_down])
            labels.extend([f"{base} Up", f"{base} Down"])

        # === Estilo general ===
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel("Events", fontsize=14)
        label = sample_map.get(sample.lower(), sample)
        plt.title(f"{title_prefix}{label}", fontsize=14)
        if log:
            plt.yscale("log")
        plt.grid(True)
        hep.cms.text("Preliminary", loc=0)

        lumi_map = {
            "2017": "41.5 fb$^{-1}$ (2017, 13 TeV)",
            "2018": "59.8 fb$^{-1}$ (2018, 13 TeV)",
            "2016": "16.8 fb$^{-1}$ (2016, 13 TeV)",
            "2016APV": "19.5 fb$^{-1}$ (2016, 13 TeV)"
        }
        
        plt.text(0.98, 0.95, f"{lumi_map[year]}", transform=plt.gca().transAxes, ha="right", va="top", fontsize=12)

        # === Leyenda en dos columnas ===
        plt.legend(handles=handles, labels=labels, loc="center left", bbox_to_anchor=(1, 0.5),
                   fontsize=10, frameon=False, ncol=2)

        plt.tight_layout(rect=[0, 0, 0.82, 1])  # espacio lateral para leyenda
        plt.savefig(os.path.join(png_folder, f"{sample}_variations.png"))
        plt.show()
        plt.close()
        print(f"png files have been stored in {png_folder}")



def rename_map():
    return {
        "electron_reco_Above": "CMS_eff_e_reco_above20",
        "electron_reco_Below": "CMS_eff_e_reco_below20",
        "electron_id": "CMS_eff_e_id",
        "jet_JES": "CMS_scale_j",
        "jet_JER": "CMS_res_j",
        "pujetid": "CMS_eff_j_PUJET",
        "bc_jets": "CMS_btag_heavy",
        "light_jets": "CMS_btag_light",
        "met_UNCLUSTERED": "CMS_MET_unclustered",
        "pileup": "CMS_pileup",
        "jet -> tau_h fake rate": "CMS_eff_tau_idDeepTauVSjet",
        "e -> tau_h fake rate": "CMS_eff_tau_idDeepTauVSe",
        "mu -> tau_h fake rate": "CMS_eff_tau_idDeepTauVSmu",
        "muon_id": "CMS_eff_m_id",
        "muon_iso": "CMS_eff_m_iso",
        "muon_reco": "CMS_eff_m_reco",
        "TES": "CMS_t_energy",
        "ROCHESTER": "CMS_rochester",
        "L1Prefiring": "CMS_l1_ecal_prefiring",
        "QCD_vs_W": "CMS_eff_W_particleNet",
        "QCD_vs_Top": "CMS_eff_T_particleNet",
        "fatjet_JES": "CMS_scale_fj",
        "fatjet_JER": "CMS_res_fj",   
        "met_trigger_wj": "CMS_eff_MET_trigger",
        "met_trigger_tt": "CMS_eff_MET_trigger",
        "met_trigger_wj": "CMS_eff_MET_trigger",
        "ISR_MLM_2016APV": "CMS_isr",
        "ISR_MLM_2016": "CMS_isr",
        "ISR_MLM_2017": "CMS_Z_isr",
        "ISR_FxFx_2016APV": "CMS_Z_isr",
        "ISR_FxFx_2016": "CMS_Z_isr",
        "ISR_FxFx_2017": "CMS_Z_isr",
        "ISR_FxFx_2018": "CMS_Z_isr",
        "top_boost_weight_tau_2016APV": "CMS_ttbar_boost",
        "top_boost_weight_tau_2016": "CMS_ttbar_boost",
        "top_boost_weight_tau_2017": "CMS_ttbar_boost",
        "top_boost_weight_tau_2018": "CMS_ttbar_boost",
    }

def find_base_name(var_name, rename_dict):
    for key in rename_dict:
        if var_name.startswith(key):
            return rename_dict[key]
    return var_name  # fallback if no match

def create_root_histogram(values, bin_edges, name):
    hist = ROOT.TH1F(name, name, len(bin_edges) - 1, array('d', bin_edges))
    for i, val in enumerate(values):
        hist.SetBinContent(i + 1, val)
    return hist

def save_histograms_to_root(
    histograms_dict,
    bin_edges,
    output_dir="root_files",
    region="",
    lepton="",
    year="",
    sample_map=None
):
    os.makedirs(output_dir, exist_ok=True)

    sample_map = sample_map or {
        "tt": "tt",
        "st": "SingleTop",
        "vv": "Diboson",
        "higgs": "Higgs",
        "dy": "DYJetsToLNu",
        "wj": "WJetToLNu",
        "total_bkg": "Total_bgr",
        "data": "Data",  # Cambiado de "data" a "Data" con D mayúscula
        "qcd": "QCD"
    }

    rename_dict = rename_map()
    suffix = f"{region}_{lepton}_{year}"

    for sample, variations in histograms_dict.items():
        sample_name = sample_map.get(sample, sample)
        output_name = f"{sample_name}_{suffix}.root"
        output_path = os.path.join(output_dir, output_name)

        # Verificar si hay datos para esta muestra
        nominal_values = variations.get("nominal")
        variations_up = variations.get("variations_up", {})
        variations_down = variations.get("variations_down", {})
        
        # Si no hay datos nominales ni variaciones, crear archivo vacío
        if nominal_values is None and not variations_up and not variations_down:
            print(f"⚠️  No hay datos para {sample_name}, creando archivo ROOT vacío...")
            
            root_file = ROOT.TFile(output_path, "RECREATE")
            # Crear histograma nominal vacío
            empty_hist = create_root_histogram(np.zeros(len(bin_edges) - 1), bin_edges, f"CMS_{suffix}_nom")
            empty_hist.Write()
            root_file.Close()
            
            print(f"✅ Archivo ROOT vacío creado: {output_path}")
            continue

        root_file = ROOT.TFile(output_path, "RECREATE")

        # Histograma nominal
        if nominal_values is not None:
            hist = create_root_histogram(nominal_values, bin_edges, f"CMS_{suffix}_nom")
            hist.Write()
        else:
            # Crear histograma nominal vacío si no existe
            empty_hist = create_root_histogram(np.zeros(len(bin_edges) - 1), bin_edges, f"CMS_{suffix}_nom")
            empty_hist.Write()

        # Variaciones "up"
        for var_name, values in variations_up.items():
            base_name = find_base_name(var_name, rename_dict)
            hist = create_root_histogram(values, bin_edges, f"{base_name}_{suffix}_Up")
            hist.Write()

        # Variaciones "down"
        for var_name, values in variations_down.items():
            base_name = find_base_name(var_name, rename_dict)
            hist = create_root_histogram(values, bin_edges, f"{base_name}_{suffix}_Down")
            hist.Write()

        root_file.Close()
        print(f"✅ Successfully saved: {output_path}")



def get_variation_event_table(histograms_dict):
    """
    Crea una tabla con una fila por variación (sin Up/Down en el nombre de fila),
    y columnas del tipo: 'tt Down', 'tt', 'tt Up', 'Single Top Down', etc.
    """

    sample_map = {
        "tt": "tt",
        "st": "Single Top",
        "vv": "Diboson",
        "higgs": "Higgs",
        "dy": "Drell-Yan",
        "wj": "WJets",
        "qcd": "QCD",
        "total_bkg": "Total MC",
    }

    variations = {}
    nominals = {}

    for sample, contents in histograms_dict.items():
        if sample.lower() in ["data", "signal"]:
            continue

        sample_name = sample_map.get(sample, sample)

        # Nominal
        nominal_val = np.sum(contents.get("nominal", 0))
        nominals[sample_name] = nominal_val

        # Up variations
        for var_name, hist in contents.get("variations_up", {}).items():
            row_name = var_name.replace("_up", "").replace("_Up", "").replace("Up", "").rstrip("_")
            variations.setdefault(row_name, {})[f"{sample_name} Up"] = np.sum(hist)

        # Down variations
        for var_name, hist in contents.get("variations_down", {}).items():
            row_name = var_name.replace("_down", "").replace("_Down", "").replace("Down", "").rstrip("_")
            variations.setdefault(row_name, {})[f"{sample_name} Down"] = np.sum(hist)

    # Convertir a DataFrame
    df = pd.DataFrame.from_dict(variations, orient="index")

    # Insertar columnas nominales entre Down y Up
    new_cols = []
    for col in sorted(set(c.rsplit(" ", 1)[0] for c in df.columns)):
        col_down = f"{col} Down"
        col_nom = f"{col}"
        col_up = f"{col} Up"

        if col_down in df.columns:
            new_cols.append(col_down)
        new_cols.append(col_nom)
        if col_up in df.columns:
            new_cols.append(col_up)

        # Insertar el valor nominal en esa columna
        df[col_nom] = nominals.get(col, 0)

    # Reordenar columnas
    df = df.reindex(columns=new_cols)
    df = df.fillna(0)

    # Renombrar filas (índice) usando rename_map
    renamer = rename_map()
    df = df.rename(index=lambda x: next((v for k, v in renamer.items() if x.startswith(k)), x))


    return df

def get_binwise_table_for_sample(histograms_dict, binning, sample_key, sample_map=None):
    """
    Versión que:
    - Hace matching flexible con rename_map() (incluye variaciones con sufijos como _Tight)
    - Mantiene TODAS las sistemáticas
    - Estructura de columnas: Down | Nominal | Up por cada bin
    - Sin valores NaN (usa nominal donde falten variaciones)
    """

    rename_dict = rename_map()
    bin_ranges = [f"{binning[i]}-{binning[i+1]}" for i in range(len(binning)-1)]
    contents = histograms_dict[sample_key]
    nominal_vals = contents.get("nominal", np.zeros(len(binning)-1))
    
    # Mapeo inverso: {nombre_renombrado: patrón_original}
    inverse_rename = {v: k for k, v in rename_dict.items()}
    
    # Recolectar todas las variaciones únicas del archivo
    all_variations = set()
    all_variations.update(contents.get("variations_up", {}).keys())
    all_variations.update(contents.get("variations_down", {}).keys())
    
    # Construir mapeo completo: {nombre_original: nombre_renombrado}
    full_mapping = {}
    for orig_var in all_variations:
        # Buscar el patrón más largo que coincida (para manejar sufijos)
        matched_key = None
        for pattern in rename_dict.keys():
            if pattern in orig_var:  # Matching flexible
                if matched_key is None or len(pattern) > len(matched_key):
                    matched_key = pattern
        if matched_key:
            full_mapping[orig_var] = rename_dict[matched_key]
        else:
            full_mapping[orig_var] = orig_var  # Conservar original si no hay match
    
    # Agrupar variaciones por nombre renombrado
    sys_groups = defaultdict(dict)
    for orig_var, renamed_var in full_mapping.items():
        if orig_var in contents.get("variations_up", {}):
            sys_groups[renamed_var]['up'] = contents["variations_up"][orig_var]
        if orig_var in contents.get("variations_down", {}):
            sys_groups[renamed_var]['down'] = contents["variations_down"][orig_var]
    
    # Construir DataFrame
    data = []
    for sys_name in sorted(sys_groups.keys()):
        sys_data = sys_groups[sys_name]
        row = {"Systematic": sys_name}
        
        # Procesar cada bin
        for i, br in enumerate(bin_ranges):
            # Down (usa nominal si no existe)
            down_val = sys_data.get('down', [nominal_vals[i]] * len(binning))[i]
            row[(br, "Down")] = down_val
            
            # Nominal (siempre presente)
            row[(br, "Nominal")] = nominal_vals[i]
            
            # Up (usa nominal si no existe)
            up_val = sys_data.get('up', [nominal_vals[i]] * len(binning))[i]
            row[(br, "Up")] = up_val
        
        # Totales
        down_total = sum(sys_data.get('down', nominal_vals))
        up_total = sum(sys_data.get('up', nominal_vals))
        
        row[("Total", "Down")] = down_total
        row[("Total", "Nominal")] = nominal_vals.sum()
        row[("Total", "Up")] = up_total
        
        data.append(row)
    
    # Crear DataFrame con MultiIndex
    df = pd.DataFrame(data)
    df.set_index("Systematic", inplace=True)
    
    # Ordenar columnas
    sorted_columns = []
    for br in bin_ranges + ["Total"]:
        sorted_columns.extend([(br, "Down"), (br, "Nominal"), (br, "Up")])
    
    return df[sorted_columns]

def get_total_relative_deviation_all(histograms_dict, binning):
    """
    Versión final con:
    - Headers en dos filas (nombres de muestra y variaciones)
    - (%) añadido a Up/Down
    - 'Total MC' como últimas columnas
    """

    sample_map = {
        "tt": "tt",
        "st": "Single Top",
        "qcd": "QCD",
        "vv": "Diboson",
        "higgs": "Higgs",
        "dy": "Drell-Yan",
        "wj": "WJets",
        "total_bkg": "Total MC"
    }

    
    rename_dict = rename_map()
    results = defaultdict(dict)
    sample_map = sample_map or {}

    # Recolectar todas las variaciones
    all_variations = set()
    for contents in histograms_dict.values():
        all_variations.update(contents.get("variations_up", {}).keys())
        all_variations.update(contents.get("variations_down", {}).keys())

    # Mapeo de variaciones a sistemáticas CMS
    variation_to_systematic = {}
    for orig_var in all_variations:
        matched = None
        for pattern in rename_dict:
            if pattern in orig_var:
                if not matched or len(pattern) > len(matched):
                    matched = pattern
        if matched:
            variation_to_systematic[orig_var] = rename_dict[matched]

    # Procesar datos
    for sample_key, contents in histograms_dict.items():
        if sample_key.lower() in ["data", "signal"]:
            continue

        nominal = contents.get("nominal")
        if nominal is None:
            continue

        nominal_total = np.sum(nominal)
        if nominal_total == 0:
            continue

        label = sample_map.get(sample_key, sample_key)

        # Procesar variaciones
        for direction in ["up", "down"]:
            for orig_var, hist in contents.get(f"variations_{direction}", {}).items():
                if orig_var in variation_to_systematic:
                    sys_name = variation_to_systematic[orig_var]
                    deviation = 100 * (np.sum(hist) - nominal_total) / nominal_total
                    results[sys_name][(label, direction.capitalize())] = round(deviation, 2)

    # Crear DataFrame con MultiIndex
    if not results:
        return pd.DataFrame()

    # Obtener muestras únicas ordenadas, poniendo 'Total MC' al final
    samples = sorted(set(col[0] for row in results.values() for col in row.keys()))
    if 'Total MC' in samples:
        samples.remove('Total MC')
        samples.append('Total MC')  # Mover al final
    
    # Construir datos y columnas
    data = []
    index = []
    for sys_name, deviations in results.items():
        index.append(sys_name)
        row = []
        for sample in samples:
            for variation in ["Up (%)", "Down (%)"]:
                row.append(deviations.get((sample, variation.split(' ')[0]), np.nan))
        data.append(row)
    
    # Crear columnas con (%) - Usamos "Up (%)" y "Down (%)"
    columns = pd.MultiIndex.from_product(
        [samples, ["Up (%)", "Down (%)"]],
        names=['Sample', 'Variation']
    )
    
    df = pd.DataFrame(data, index=index, columns=columns)
    
    # Eliminar el nombre del índice para que no aparezca "Systematic"
    df.index.name = None
    
    return df






def process_systematics_table(df: pd.DataFrame, include_nominal: bool = True) -> pd.DataFrame:
    """
    Procesa tabla de sistemáticas con columnas (bin, variation).
    Devuelve un DataFrame con columnas planas ordenadas por bin:
        - '<bin>_nominal'   (si include_nominal=True)
        - '<bin>_|Nom-Up|'
        - '<bin>_|Nom-Down|'
    Al final agrega una fila 'QuadratureSum' con la suma en cuadratura de cada columna.
    """
    def _norm(s): return str(s).strip().lower()

    # Construir mapa bin -> { 'nominal': col, 'up': col, 'down': col }
    bin_map = OrderedDict()
    for col in df.columns:
        if isinstance(col, tuple) and len(col) >= 2:
            b, v = col[0], col[1]
        else:
            s = str(col)
            parts = [p.strip() for p in s.split(',')] if ',' in s else s.split()
            if len(parts) >= 2:
                b, v = parts[0], parts[-1]
            else:
                continue

        vn = _norm(v)
        if vn.startswith('nom'):
            kind = 'nominal'
        elif 'up' in vn or '+' in vn:
            kind = 'up'
        elif 'down' in vn or '-' in vn:
            kind = 'down'
        else:
            continue

        if b not in bin_map:
            bin_map[b] = {}
        bin_map[b][kind] = col

    # Ordenar bins numéricamente dejando 'Total' al final
    def _bin_key(b):
        if str(b).lower() == 'total':
            return float('inf')
        try:
            return int(str(b).split('-')[0])
        except:
            return 10**9

    sorted_bins = sorted(bin_map.keys(), key=_bin_key)

    # Construir resultado
    rows_idx = df.index
    data = OrderedDict()
    col_order = []

    for b in sorted_bins:
        mapping = bin_map[b]
        nom_col = mapping.get('nominal')
        up_col = mapping.get('up')
        down_col = mapping.get('down')

        # nominal
        nominal = None
        if nom_col is not None:
            nominal = pd.to_numeric(df[nom_col], errors='coerce')
        elif include_nominal:
            nominal = pd.Series([pd.NA] * len(df), index=rows_idx)

        if include_nominal:
            data[f"{b}_nominal"] = nominal
            col_order.append(f"{b}_nominal")

        # absUp
        if up_col is not None:
            up = pd.to_numeric(df[up_col], errors='coerce')
            data[f"{b}_|Nom-Up|"] = (up - nominal).abs() if nominal is not None else pd.NA
            col_order.append(f"{b}_|Nom-Up|")

        # absDown
        if down_col is not None:
            down = pd.to_numeric(df[down_col], errors='coerce')
            data[f"{b}_|Nom-Down|"] = (down - nominal).abs() if nominal is not None else pd.NA
            col_order.append(f"{b}_|Nom-Down|")

    df_out = pd.DataFrame(data, index=rows_idx)[col_order]

    # --- Agregar fila con suma en cuadratura ---
    quadrature_sum = np.sqrt((df_out.astype(float) ** 2).sum(axis=0))
    df_out.loc["Total"] = quadrature_sum

    # --- Subtabla: solo la fila Total ---
    df_sub = df_out.loc[["Total"]]

    return df_out, df_sub


# -------------------------------------------------
# Contruyendo el archivo root de QCD: Data-driven
# -------------------------------------------------
def build_qcd_datadriven(
    pkls: dict,
    norms: dict,
    qcd_shape: str,
    qcd_ratio: list[str],
    qcd_ratio_integrated: bool,
    variable: str,
    binning: list[float],
):
    """
    Construye histogramas de QCD (data-driven) con variaciones sistemáticas:
      QCD = Data - (sum otros bkg)
      QCD^var = Data - sum(bkg^var)

    El transfer factor (TF) se calcula automáticamente usando QCD_num/QCD_den:
      - Si qcd_ratio_integrated=True → TF escalar (integrado)
      - Si qcd_ratio_integrated=False → TF bin a bin

    Guarda en ROOT file.
    """

    # --- 1. Construir histogramas nominales y con variaciones ---
    histograms  = variation_histograms(pkls[qcd_shape], norms[qcd_shape], variable, binning)
    numerator   = variation_histograms(pkls[qcd_ratio[0]], norms[qcd_ratio[0]], variable, binning)
    denominator = variation_histograms(pkls[qcd_ratio[1]], norms[qcd_ratio[1]], variable, binning)

    
    # --- 1b. Función auxiliar para estimar QCD = Data - MC ---
    def estimate_qcd(hists: dict, variation: Optional[str] = None, direction: str = "up") -> np.ndarray:
        """
        Devuelve QCD = Data - MC para nominal o para una variación.
        - variation=None → usa nominal
        - variation="X", direction="up"/"down" → usa la variación sistemática
        """
        # Data siempre nominal
        data_h = hists.get("data", {}).get("nominal")
        if data_h is None:
            raise ValueError("No se encontró 'data' en el pkl")

        mc_groups = ["tt", "st", "wj", "vv", "dy", "higgs"]
        total_mc = np.zeros_like(data_h, dtype=float)

        for grp in mc_groups:
            if grp not in hists:
                continue

            if variation is None:
                h = hists[grp].get("nominal")
            else:
                var_dict = hists[grp].get(f"variations_{direction}", {})
                h = var_dict.get(variation, hists[grp].get("nominal"))

            if h is not None:
                total_mc += h

        return data_h - total_mc

    # --- 1c. Calcular QCD estimado en numerator y denominator ---
    qcd_num = estimate_qcd(numerator)
    qcd_den = estimate_qcd(denominator)

    # --- 1d. Calcular transfer factor ---
    if qcd_ratio_integrated:
        tf_num = np.sum(qcd_num)
        tf_den = np.sum(qcd_den)
        transfer_factor = tf_num / tf_den if tf_den != 0 else 0.0
    else:
        transfer_factor = np.divide(
            qcd_num,
            qcd_den,
            out=np.zeros_like(qcd_num, dtype=float),
            where=qcd_den != 0
        )

    #print("Transfer factor:", transfer_factor)

    # --- 2. QCD en la región del shape (nominal) ---
    qcd_shape_nominal = estimate_qcd(histograms)
    qcd_nominal = qcd_shape_nominal * transfer_factor  # aplica escalar o array

    # --- 3. Construir variaciones sistemáticas para QCD ---
    qcd_variations_up = {}
    qcd_variations_down = {}

    # Recolectar todas las variaciones sistemáticas únicas
    all_variations = set()
    mc_groups = ["tt", "st", "wj", "vv", "dy", "higgs"]
    for grp in mc_groups:
        if grp in histograms:
            all_variations.update(histograms[grp].get("variations_up", {}).keys())
            all_variations.update(histograms[grp].get("variations_down", {}).keys())

    for variation in all_variations:
        qcd_up   = estimate_qcd(histograms, variation=variation, direction="up") * transfer_factor
        qcd_down = estimate_qcd(histograms, variation=variation, direction="down") * transfer_factor

        qcd_variations_up[variation] = qcd_up
        qcd_variations_down[variation] = qcd_down

    # --- 4. Crear estructura final de QCD ---
    qcd_histograms = {
        "nominal": qcd_nominal,
        "variations_up": qcd_variations_up,
        "variations_down": qcd_variations_down
    }


    return qcd_histograms





