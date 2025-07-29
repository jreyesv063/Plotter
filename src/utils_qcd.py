import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as patches
from typing import Dict, Any, Optional, List

from src.utils import load_all_pickles, load_all_jsons


def QCD_squema_plot(cr: str, shape: str):

    cr_map = {
        "wjets": "W+jets CR",
        "signal": "Signal",
    }

    shape_map = {
        "cr_b": "CR_B",
        "cr_c": "CR_C",
        "cr_d": "CR_D",
    }

    # --- TF según la región shape ---
    tf_map = {
        "cr_b": ("C", "D"),
        "cr_c": ("B", "D"),
        "cr_d": ("B", "C"),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 2.4)

    def draw_block(x, y, label):
        ax.add_patch(patches.Rectangle((x + 0.02, y - 0.02), 0.9, 0.9, facecolor='#0d3c52', edgecolor='none'))
        ax.add_patch(patches.Rectangle((x, y), 0.9, 0.9, facecolor='#19526b', edgecolor='black'))
        ax.text(x + 0.45, y + 0.45, label, ha='center', va='center',
                fontsize=16, fontweight='bold', color='white')

    draw_block(0.05, 1.35, cr_map[cr])   # W+jets CR o Signal
    draw_block(1.25, 1.35, "CR B")
    draw_block(0.05, 0.05, "CR C")
    draw_block(1.25, 0.05, "CR D")

    # Coordenadas de centros
    block_coords = {
        "wjets": (0.05 + 0.45, 1.35 + 0.45),
        "signal": (0.05 + 0.45, 1.35 + 0.45),
        "cr_b": (1.25 + 0.45, 1.35 + 0.45),
        "cr_c": (0.05 + 0.45, 0.05 + 0.45),
        "cr_d": (1.25 + 0.45, 0.05 + 0.45),
    }

    origin = block_coords[shape]
    target = block_coords[cr]

    ax.annotate(
        "",
        xy=target,
        xytext=origin,
        arrowprops=dict(
            arrowstyle='-|>',
            mutation_scale=30,
            lw=4,
            facecolor='yellow',
            edgecolor='yellow',
            connectionstyle="arc3,rad=1.0"
        )
    )

    # Obtener letras para TF
    tf_a, tf_b = tf_map[shape]

    # Texto de fórmula
    ax.text(1.1, 2.3, 
            fr"$\mathrm{{QCD}}(W + \mathrm{{jets}}) = \mathrm{{{shape_map[shape]}}} \times \mathrm{{TF}}_{{{tf_a},{tf_b}}}$",
            fontsize=14, ha='center')

    # Ejes
    ax.annotate('', xy=(2.1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.annotate('', xy=(0, 2.3), xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=3, color='red'))

    ax.text(1.1, -0.2, r"$\Delta \phi (jets, p_T^{\mathrm{miss}})$", ha='center', fontsize=14, color='red')
    ax.text(-0.2, 1.2, "DeepTau ID", rotation=90, va='center', fontsize=14, color='red')

    ax.plot([0.9, 0.9], [0, -0.05], color='red', lw=2)
    ax.plot([1.9, 1.9], [0, -0.05], color='red', lw=2)
    ax.plot([0, -0.05], [1.3, 1.3], color='red', lw=2)
    ax.plot([0, -0.05], [2.3, 2.3], color='red', lw=2)

    ax.text(-0.6, 1.75, " DeepTau >= Tight", fontsize=12, ha='left')
    ax.text(-0.7, 0.45, "Tight > DeepTau > Loose", fontsize=12, ha='left')

    ax.text(0.4, -0.12, r" > 0.7", fontsize=12)
    ax.text(1.6, -0.12, r" < 0.7", fontsize=12)

    if cr == "wjets":
        ax.text(1.00, 1.1, r"$\mathrm{N(b) = 0}$", fontsize=14)
    elif cr == "signal":
        ax.text(1.00, 1.1, r"$\mathrm{N(b) = 1}$", fontsize=14)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def get_qcd_cutflow(json_map, normalization, variation, grouped_samples=None, combined_samples=False, combined_2016=False):
    def get_rename_map(groups):
        return {k: k for k in groups}  # identidad si no hay renombres

    def compute_eff_cutflow(cutflow_table, normalization):
        scaled_errors = {}
        for sample, cuts in cutflow_table.items():
            norm = normalization.get(sample, 1.0)
            errors = {}
            for cut, value in cuts.items():
                try:
                    val = float(value) * norm
                    errors[cut] = np.sqrt(val)
                except:
                    errors[cut] = 0.0
            scaled_errors[sample] = errors
        return {}, scaled_errors

    def get_table_cutflow_unscaled(json_map):
        out = {}
        for dataset, content in json_map.items():
            if "cutflow" in content:
                out[dataset] = content["cutflow"]
        return out

    _, scaled_error_df = compute_eff_cutflow(
        cutflow_table=get_table_cutflow_unscaled(json_map),
        normalization=normalization
    )

    if combined_2016:
        sample_ref = "ST_tW_top_5f_inclusiveDecays_2016"
    else:
        sample_ref = "ST_tW_top_5f_inclusiveDecays"
        

    allowed_variations = [key for key in json_map[sample_ref].keys() if key.startswith("cutflow")]
    if variation not in allowed_variations:
        raise ValueError(f"❌ La variation '{variation}' no está en los datasets.")

    # Armar base de cortes
    for ds in json_map:
        if variation in json_map[ds]:
            base_cuts = list(json_map[ds][variation].keys())
            break
    else:
        raise ValueError(f"No se encontró la variation {variation} en ningún dataset.")

    cutflow_scaled = {}

    for dataset in json_map:
        norm = float(normalization.get(dataset, 1.0))
        cutflow_nominal = json_map[dataset].get("cutflow", {})

        if dataset in ['SingleElectron', 'SingleMuon', 'Tau', 'MET']:
            scaled = {}
            for cut in base_cuts:
                value = cutflow_nominal.get(cut)
                if value is None:
                    suffix = "_" + variation.split("cutflow")[-1].lstrip("_")
                    if cut.endswith(suffix):
                        cut_base = cut.removesuffix(suffix)
                        value = cutflow_nominal.get(cut_base)
                try:
                    scaled[cut] = float(value) * norm if value is not None else None
                except:
                    scaled[cut] = None
            cutflow_scaled[dataset] = scaled
            continue

        cutflow_source = json_map[dataset].get(variation, {})
        scaled = {}
        for cut in base_cuts:
            value = cutflow_source.get(cut)
            if value is None:
                base_cut = cut.rsplit("_", 1)[0] if "_" in cut else cut
                value = cutflow_nominal.get(base_cut)
            try:
                scaled[cut] = float(value) * norm if value is not None else None
            except:
                scaled[cut] = None
        cutflow_scaled[dataset] = scaled

    df = pd.DataFrame.from_dict(cutflow_scaled, orient="index").transpose()

    # Agrupar muestras si se pide (solo para MC, no para datos)
    if combined_samples and grouped_samples:
        grouped_cutflows = defaultdict(lambda: defaultdict(float))
        for group_name, samples in grouped_samples.items():
            for sample in samples:
                # Solo procesar muestras que no son datos
                if sample not in ['SingleElectron', 'SingleMuon', 'Tau', 'MET']:
                    for cut in df.index:
                        val = df.get(sample, {}).get(cut)
                        if val is not None:
                            grouped_cutflows[group_name][cut] += val
        df_grouped = pd.DataFrame.from_dict(grouped_cutflows, orient="index").transpose()
        df_grouped = df_grouped.rename(columns=get_rename_map(grouped_samples))
        
        # Calcular el Total MC (suma de todos los fondos conocidos)
        mc_columns = [col for col in df_grouped.columns if col not in ['SingleElectron', 'SingleMuon', 'Tau', 'MET']]
        df_grouped["Total MC"] = df_grouped[mc_columns].sum(axis=1)
        
        # Mantener los datos originales
        for data_type in ['SingleElectron', 'SingleMuon', 'Tau', 'MET']:
            if data_type in df.columns:
                df_grouped[data_type] = df[data_type]
        
        df = df_grouped

    # Calcular Data (suma de todos los datasets de datos)
    data_samples = [ds for ds in json_map if ds in ['SingleElectron', 'SingleMuon', 'Tau', 'MET']]
    df["Data"] = 0.0
    for dataset in data_samples:
        for cut in df.index:
            val = cutflow_scaled[dataset].get(cut)
            if val is not None:
                df.at[cut, "Data"] += val

    # Calcular Total MC si no se ha calculado ya
    if "Total MC" not in df.columns:
        mc_samples = [col for col in df.columns if col not in ['SingleElectron', 'SingleMuon', 'Tau', 'MET', 'Data']]
        df["Total MC"] = df[mc_samples].sum(axis=1)

    # Calcular QCD (D-D) y su error
    qcd_values = []
    for cut in df.index:
        data_val = df.at[cut, "Data"]
        total_mc_val = df.at[cut, "Total MC"]
        err_data = np.sqrt(data_val) if data_val > 0 else 0.0
        
        # Calcular error total MC (suma en cuadratura de errores individuales)
        err_total_mc = 0.0
        for sample, errors in scaled_error_df.items():
            if sample not in ['SingleElectron', 'SingleMuon', 'Tau', 'MET']:
                err_total_mc += errors.get(cut, 0.0)**2
        err_total_mc = np.sqrt(err_total_mc)

        if pd.notna(data_val) and pd.notna(total_mc_val):
            diff = data_val - total_mc_val
            err = np.sqrt(err_data**2 + err_total_mc**2)
            qcd_values.append(f"{diff:.2f} ± {err:.2f}")
        else:
            qcd_values.append("")

    # Crear DataFrame final con las columnas requeridas
    result_df = pd.DataFrame({
        "Data": df["Data"].round(2),
        "Total MC": df["Total MC"].round(2),
        "QCD (D-D)": qcd_values
    }, index=df.index)

    return result_df
def qcd_estimation(json_map, normalization, variation, grouped_samples=None, 
                  combined_samples=False, combined_2016=False, 
                  cr_B_folder="", cr_C_folder="", cr_D_folder="",
                  shape_region="cr_b", ratio_regions=["cr_c", "cr_d"]):
    """
    Calcula la estimación de QCD combinando tres regiones de control.
    
    Args:
        json_map: Mapa JSON para la región principal
        normalization: Factores de normalización
        variation: Variación del cutflow a usar
        grouped_samples: Grupos de muestras para combinar
        combined_samples: Si combinar muestras
        combined_2016: Si usar combinación para 2016
        cr_B_folder: Carpeta con JSONs para región B
        cr_C_folder: Carpeta con JSONs para región C
        cr_D_folder: Carpeta con JSONs para región D
        shape_region: Región para shape (cr_b, cr_c o cr_d)
        ratio_regions: Lista [X, Y] para el ratio X/Y
        
    Returns:
        DataFrame con los resultados combinados
    """
    QCD_squema_plot(cr = "wjets", shape = shape_region)

    # Cargar los JSONs de cada región de control
    cr_B_jsons = load_all_jsons(os.path.join(cr_B_folder, "summary", "metadata"))
    cr_C_jsons = load_all_jsons(os.path.join(cr_C_folder, "summary", "metadata"))
    cr_D_jsons = load_all_jsons(os.path.join(cr_D_folder, "summary", "metadata"))

    # Obtener los cutflows de QCD para cada región
    cr_b_qcd = get_qcd_cutflow(cr_B_jsons, normalization, variation, 
                              grouped_samples, combined_samples, combined_2016)
    cr_c_qcd = get_qcd_cutflow(cr_C_jsons, normalization, variation,
                              grouped_samples, combined_samples, combined_2016)
    cr_d_qcd = get_qcd_cutflow(cr_D_jsons, normalization, variation,
                              grouped_samples, combined_samples, combined_2016)

    # Verificar que tenemos las regiones necesarias
    available_regions = {
        "cr_b": cr_b_qcd if not cr_b_qcd.empty else None,
        "cr_c": cr_c_qcd if not cr_c_qcd.empty else None,
        "cr_d": cr_d_qcd if not cr_d_qcd.empty else None
    }

    # Validar parámetros
    if shape_region not in available_regions or available_regions[shape_region] is None:
        raise ValueError(f"La región de shape '{shape_region}' no está disponible")

    for r in ratio_regions:
        if r not in available_regions or available_regions[r] is None:
            raise ValueError(f"La región de ratio '{r}' no está disponible")

    # Función para extraer valor y error de formato "X ± Y"
    def extract_value_error(qcd_str):
        if pd.isna(qcd_str) or not isinstance(qcd_str, str) or "±" not in qcd_str:
            return 0.0, 0.0
        parts = qcd_str.split("±")
        return float(parts[0].strip()), float(parts[1].strip())
    
    # Obtener datos de cada región
    shape_df = available_regions[shape_region]
    X_df = available_regions[ratio_regions[0]]
    Y_df = available_regions[ratio_regions[1]]
    
    # Calcular QCD estimado = shape * (X/Y)
    results = []

    for cut in shape_df.index:
        # Obtener valores para shape
        shape_val, shape_err = extract_value_error(shape_df.loc[cut, "QCD (D-D)"])
        
        # Encontrar el corte correspondiente en las otras regiones
        # Buscamos coincidencias flexibles para manejar diferencias True/False
        def find_matching_cut(target_df, base_cut):
            # Primero intenta con el nombre exacto
            if base_cut in target_df.index:
                return base_cut
            
            # Si no encuentra, busca variantes con True/False
            base_name = base_cut.rsplit('_', 1)[0]
            for candidate in target_df.index:
                if candidate.startswith(base_name + '_'):
                    return candidate
            return None
        
        x_cut = find_matching_cut(X_df, cut)
        y_cut = find_matching_cut(Y_df, cut)
        
        if x_cut is None or y_cut is None:
            print(f"Advertencia: No se encontró corte correspondiente para {cut} en alguna región")
            continue
            
        # Obtener valores para X e Y usando los cortes correspondientes
        try:
            X_val, X_err = extract_value_error(X_df.loc[x_cut, "QCD (D-D)"])
            Y_val, Y_err = extract_value_error(Y_df.loc[y_cut, "QCD (D-D)"])
        except KeyError as e:
            print(f"Error al acceder a los cortes: {e}")
            continue
        
        # Calcular ratio X/Y con propagación de errores
        if Y_val != 0:
            ratio = X_val / Y_val
            ratio_err = ratio * np.sqrt((X_err/X_val)**2 + (Y_err/Y_val)**2)
        else:
            ratio = 0.0
            ratio_err = 0.0
        
        # Calcular QCD estimado
        qcd_estimated = shape_val * ratio
        qcd_estimated_err = np.sqrt((shape_err * ratio)**2 + (shape_val * ratio_err)**2)
        
        results.append({
            "Cut": cut,
            f"QCD {shape_region}": shape_df.loc[cut, "QCD (D-D)"],
            f"QCD {ratio_regions[0]}": X_df.loc[x_cut, "QCD (D-D)"],
            f"QCD {ratio_regions[1]}": Y_df.loc[y_cut, "QCD (D-D)"],
            "Ratio (X/Y)": f"{ratio:.4f} ± {ratio_err:.4f}",
            "QCD Estimated": f"{qcd_estimated:.2f} ± {qcd_estimated_err:.2f}"
        })

    return pd.DataFrame(results).set_index("Cut")


def get_qcd_estimation_shape(
    pkls,
    bins: np.ndarray,
    distribution: str,
    consider_overflow: bool = True,
    consider_underflow: bool = True,
    normalization_factors: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Autonomously estimates QCD background by identifying data samples and subtracting all MC backgrounds.
    
    Args:
        pkls: Dictionary of samples {sample_name: {subkey: arrays}}
        bins: Bin edges for the histogram
        distribution: Variable name to histogram
        consider_overflow: Include overflow in last bin
        consider_underflow: Include underflow in first bin
        normalization_factors: Optional scale factors for MC samples {sample_name: factor}
        
    Returns:
        Estimated QCD histogram
        
    Raises:
        ValueError: If no data samples are found
    """
    # Auto-detect data samples (looking for typical CMS data naming patterns)
    def is_data_sample(sample_name: str) -> bool:
        data_patterns = {
            'SingleMuon', 'SingleElectron', 'DoubleMuon', 'DoubleEG', 
            'Tau', 'MET', 'JetHT', 'EGamma', 'HTMHT', 'ZeroBias'
        }
        return any(sample_name.startswith(pattern) for pattern in data_patterns)

    # Initialize accumulators
    total_data = np.zeros(len(bins)-1)
    total_mc = np.zeros(len(bins)-1)



    for sample_name, sample_data in pkls.items():
        # Handle nested structure (2016APV cases)
        subkey = sample_name.rsplit('_', 1)[0] if '_2016APV' in sample_name else sample_name
        if subkey not in sample_data:
            continue
            
        arrays = sample_data[subkey]
        if distribution not in arrays:
            continue
            
        is_data = is_data_sample(sample_name)
        weights = None if is_data else arrays.get("weights")

        # Compute base histogram
        variable = arrays[distribution]
        hist, _ = np.histogram(variable, bins=bins, weights=weights)

        # Apply overflow/underflow corrections
        if consider_underflow or consider_overflow:
            mask_under = variable < bins[0]
            mask_over = variable > bins[-1]
            
            correction = np.sum(weights[mask_under] if weights is not None else mask_under.sum()) if consider_underflow else 0
            hist[0] += correction
            
            correction = np.sum(weights[mask_over] if weights is not None else mask_over.sum()) if consider_overflow else 0
            hist[-1] += correction

        # Apply normalization if provided and MC
        if not is_data and normalization_factors:
            hist *= normalization_factors.get(sample_name, 1.0)

        # Accumulate
        if is_data:
            total_data += hist
        else:
            total_mc += hist

    # Verify we found data
    if np.sum(total_data) == 0:
        raise ValueError("No data samples found - cannot estimate QCD")

    # Calculate QCD (with physical constraints)
    qcd = total_data - total_mc
    qcd = np.clip(qcd, 0, None)  # Remove negative bins
    
    # Optional: Normalize to data-MC difference in integral
    data_int = np.sum(total_data)
    mc_int = np.sum(total_mc)
    if data_int > mc_int and np.sum(qcd) > 0:
        qcd *= (data_int - mc_int) / np.sum(qcd)

    return qcd

def transfer_factor_qcd(
    num_folder: str,
    den_folder: str,
    bins: np.ndarray,
    distribution: str,
    consider_overflow: bool = True,
    consider_underflow: bool = True,
    normalization_factors: Optional[Dict[str, float]] = None,
    integrated: bool = False
) -> np.ndarray:
    # Read pkl files:
    pkls_num = load_all_pickles(num_folder)
    pkls_den = load_all_pickles(den_folder)

    qcd_shape_num = get_qcd_estimation_shape(pkls_num, bins, distribution, consider_overflow, consider_underflow, normalization_factors)
    qcd_shape_den = get_qcd_estimation_shape(pkls_den, bins, distribution, consider_overflow, consider_underflow, normalization_factors)

    if integrated:
        num_integral = np.sum(qcd_shape_num)
        den_integral = np.sum(qcd_shape_den)
        if den_integral == 0:
            raise ZeroDivisionError("QCD denominator integral is zero in transfer factor calculation.")
        return num_integral / den_integral

    # Return bin-by-bin transfer factor
    with np.errstate(divide='ignore', invalid='ignore'):
        TF = np.divide(qcd_shape_num, qcd_shape_den, out=np.zeros_like(qcd_shape_num), where=qcd_shape_den!=0)
    return TF

    #TF = qcd_shape_num / qcd_shape_den

    #return TF
    

def get_qcd_estimation(    
    pkls_folder_shape: str,
    pkls_folder_num: str,
    pkls_folder_den: str,
    bins: np.ndarray,
    distribution: str,
    consider_overflow: bool = True,
    consider_underflow: bool = True,
    normalization_factors: Optional[Dict[str, float]] = None,
    ratio_per_bin: bool = False
) -> np.ndarray:
    
    # Read pkl files:
    pkls = load_all_pickles(pkls_folder_shape)

    qcd_shape = get_qcd_estimation_shape(pkls, bins, distribution, consider_overflow, consider_underflow, normalization_factors)
    transfer_factor =  transfer_factor_qcd(pkls_folder_num, pkls_folder_den, bins, distribution, consider_overflow, consider_underflow, normalization_factors, ratio_per_bin)

    print(transfer_factor)
    return qcd_shape * transfer_factor