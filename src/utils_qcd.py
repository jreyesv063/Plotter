import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as patches
from typing import Dict, Any, Optional, List

from src.utils import load_all_pickles, load_all_jsons, get_rename_map, get_weights
from src.utils_errors import get_table_cutflow_unscaled, compute_eff_cutflow, compute_statistical_error


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


def get_qcd_cutflow(json_map, normalization, variation,
                    grouped_samples=None, combined_samples=False, combined_2016=False):
    """
    Construye la tabla de cutflow con Data, Total MC y QCD (Data - MC) con errores propagados.
    """
    pd.set_option('display.float_format', '{:.2f}'.format)

    # --------------------------------------------------------------------------
    # Filtrar señales de QCD
    # --------------------------------------------------------------------------
    json_map = {k: v for k, v in json_map.items() if not k.startswith("Signal")}
    normalization = {k: v for k, v in normalization.items() if not k.startswith("Signal")}

    if grouped_samples is not None:
        grouped_samples = {
            g: [s for s in samples if not s.startswith("Signal")]
            for g, samples in grouped_samples.items()
        }
        grouped_samples = {g: s for g, s in grouped_samples.items() if s}

    # --------------------------------------------------------------------------
    result_map, scaled_error_df = compute_eff_cutflow(
        cutflow_table=get_table_cutflow_unscaled(json_map, "cutflow_raw"),
        normalization=normalization
    )

    is_cutflow = variation.startswith("cutflow")
    cutflow_scaled = {}

    # Determinar los cuts base
    try:
        base_cuts = next(
            list(json_map[ds][variation].keys()) for ds in json_map if variation in json_map[ds]
        )
    except StopIteration:
        raise ValueError(f"❌ Ningún dataset contiene la variation '{variation}'")

    # --------------------------------------------------------------------------
    # Escalado normal de todos los datasets
    # --------------------------------------------------------------------------
    for dataset in json_map:
        norm = float(normalization.get(dataset, 1.0))
        cutflow_nominal = json_map[dataset].get("cutflow", {})

        # --- DATA ---
        if dataset.split("_")[0] in ['SingleElectron', 'SingleMuon', 'Tau', 'MET']:
            # Dataset con sufijo de año
            scaled = {}
            for cut in base_cuts:
                value = cutflow_nominal.get(cut)
                scaled[cut] = float(value) * norm if value is not None else 0.0
            cutflow_scaled[dataset] = scaled
            continue

        # --- MC ---
        cutflow_source = json_map[dataset].get(variation, {})
        scaled = {}
        for cut in base_cuts:
            value = cutflow_source.get(cut)
            if value is None:
                base_cut = cut.rsplit("_", 1)[0] if "_" in cut else cut
                value = cutflow_nominal.get(base_cut)
            scaled[cut] = float(value) * norm if value is not None else None
        cutflow_scaled[dataset] = scaled

    # --------------------------------------------------------------------------
    # Combinar Data si combined_2016=True
    # --------------------------------------------------------------------------
    if combined_2016:
        data_keys = [k for k in cutflow_scaled.keys() if k.startswith(("MET", "SingleMuon", "SingleElectron", "Tau"))]
        if data_keys:
            # Sumar ambos años por cut
            combined_data = {}
            for cut in base_cuts:
                combined_data[cut] = sum(cutflow_scaled.get(k, {}).get(cut, 0.0) for k in data_keys)
            # Guardar bajo una clave única "Data"
            cutflow_scaled["Data_combined"] = combined_data
            data_dataset = "Data_combined"
    else:
        # Single year: solo tomar el primer dataset de Data que exista
        for k in cutflow_scaled.keys():
            if k.split("_")[0] in ['MET', 'SingleMuon', 'SingleElectron', 'Tau']:
                data_dataset = k
                break

    # --------------------------------------------------------------------------
    # Convertir a DataFrame
    # --------------------------------------------------------------------------
    df = pd.DataFrame.from_dict(cutflow_scaled, orient="index").transpose()
    sumw_row = df.loc["sumw"].copy() if "sumw" in df.index else None

    # --------------------------------------------------------------------------
    # Combinar samples si combined_samples=True
    # --------------------------------------------------------------------------
    if combined_samples and grouped_samples:
        grouped_cutflows = defaultdict(lambda: defaultdict(float))
        grouped_errors = defaultdict(lambda: defaultdict(float))

        for group_name, samples in grouped_samples.items():
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
        rename_columns = get_rename_map(grouped_samples)
        df_grouped = df_grouped.rename(columns=rename_columns)
        error_grouped_df = error_grouped_df.rename(columns=rename_columns)

        bkg_cols = [col for col in df_grouped.columns if not col.startswith("Data")]
        df_grouped["Total"] = df_grouped[bkg_cols].sum(axis=1)
        error_grouped_df["Total"] = np.sqrt(np.square(error_grouped_df[bkg_cols]).sum(axis=1))

        # Data combinado o simple
        df_grouped[f"Data ({data_dataset})"] = df_grouped.index.map(
            lambda cut: cutflow_scaled[data_dataset].get(cut, 0.0)
        )

        # Construcción tabla final
        data_with_err = []
        total_with_err = []
        qcd_with_err = []
        for cut in df_grouped.index:
            data_val = df_grouped.at[cut, f"Data ({data_dataset})"] if f"Data ({data_dataset})" in df_grouped.columns else 0.0
            total_val = df_grouped.at[cut, "Total"]
            err_data = np.sqrt(data_val) if data_val > 0 else 0.0
            err_total = error_grouped_df.at[cut, "Total"] if "Total" in error_grouped_df.columns else 0.0
            data_with_err.append(f"{data_val:.2f} ± {err_data:.2f}")
            total_with_err.append(f"{total_val:.2f} ± {err_total:.2f}")
            diff = data_val - total_val
            err_qcd = np.sqrt(err_data**2 + err_total**2)
            qcd_with_err.append(f"{diff:.2f} ± {err_qcd:.2f}")

        result_df = pd.DataFrame({
            "Data": data_with_err,
            "Total": total_with_err,
            "QCD (D-D)": qcd_with_err
        }, index=df_grouped.index)

        if sumw_row is not None:
            result_df.loc["sumw"] = ""
        return result_df

    # --------------------------------------------------------------------------
    # Caso sin combinación de samples
    # --------------------------------------------------------------------------
    df_with_errors = df.copy()
    for col in df.columns:
        for cut in df.index:
            val = df.at[cut, col]
            err = scaled_error_df.get(col, {}).get(cut)
            if cut == "sumw":
                df_with_errors.at[cut, col] = f"{val:.2f}" if pd.notna(val) else ""
            elif col.startswith("Data"):
                err_data = np.sqrt(val) if val > 0 else 0.0
                df_with_errors.at[cut, col] = f"{val:.2f} ± {err_data:.2f}"
            elif pd.notna(val) and pd.notna(err):
                df_with_errors.at[cut, col] = f"{val:.2f} ± {err:.2f}"
            else:
                df_with_errors.at[cut, col] = ""
    if sumw_row is not None:
        df_with_errors.loc["sumw"] = sumw_row

    return df_with_errors


def qcd_estimation(variation, 
                   grouped_samples=None, 
                  combined_samples=False, combined_2016=False, 
                  cr_BCD_normalization=None, cr_BCD_jsons=None,
                  shape_region="cr_b", ratio_regions=["cr_c", "cr_d"]):
    """
    Calcula la estimación de QCD combinando tres regiones de control.
    
    Args:
        variation: Variación del cutflow a usar
        grouped_samples: Grupos de muestras para combinar
        combined_samples: Si combinar muestras
        combined_2016: Si usar combinación para 2016
        cr_BCD_normalization: Diccionario para las 3 regiones de control, cr_b; cr_c; cr_d. COntiene Luminosidad * xsec/sumw
        cr_BCD_jsons:  Diccionario para las 3 regiones de control, cr_b; cr_c; cr_d
        shape_region: Región para shape (cr_b, cr_c o cr_d)
        ratio_regions: Lista [X, Y] para el ratio X/Y
        
    Returns:
        DataFrame con los resultados combinados
    """
    # Obtener los cutflows de QCD para cada región  
    cr_b_qcd = get_qcd_cutflow(cr_BCD_jsons[shape_region], cr_BCD_normalization[shape_region], variation, 
                              grouped_samples, combined_samples, combined_2016)
    
    cr_c_qcd = get_qcd_cutflow(cr_BCD_jsons[ratio_regions[0]], cr_BCD_normalization[ratio_regions[0]], variation,
                              grouped_samples, combined_samples, combined_2016)
    
    cr_d_qcd = get_qcd_cutflow(cr_BCD_jsons[ratio_regions[1]], cr_BCD_normalization[ratio_regions[1]], variation,
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
    normalization_factors: Optional[Dict[str, float]] = None,
    combined_2016: bool = True
) -> (np.ndarray, np.ndarray):
    """
    Estimates QCD background from data minus MC.
    Returns (qcd, qcd_err), where:
      - data error = sqrt(N_data)
      - MC error   = compute_statistical_error(bin, total) * total
    """

    def is_data_sample(sample_name: str) -> bool:
        data_patterns = {
            'SingleMuon', 'SingleElectron', 'DoubleMuon', 'DoubleEG',
            'Tau', 'MET', 'JetHT', 'EGamma', 'HTMHT', 'ZeroBias'
        }
        return any(sample_name.startswith(pattern) for pattern in data_patterns)

    def compute_qcd_for_subset(sub_pkls):
        nbins = len(bins) - 1
        total_data = np.zeros(nbins)
        total_mc = np.zeros(nbins)

        total_data_int = 0.0
        total_mc_int = 0.0

        # Construir histogramas
        for sample_name, sample_data in sub_pkls.items():
            base_name = sample_name.rsplit('_', 1)[0] if sample_name.endswith(("2016", "2016APV")) else sample_name
            arrays = sample_data.get(base_name, None)
            if arrays is None or distribution not in arrays:
                continue

            is_data = is_data_sample(base_name)
            weights = None if is_data else arrays.get("weights")
            variable = arrays[distribution]

            hist, _ = np.histogram(variable, bins=bins, weights=weights)

            # Under/overflow
            if consider_underflow:
                mask_under = variable < bins[0]
                hist[0] += np.sum(weights[mask_under] if weights is not None else mask_under.sum())
            if consider_overflow:
                mask_over = variable > bins[-1]
                hist[-1] += np.sum(weights[mask_over] if weights is not None else mask_over.sum())

            # Normalización MC
            if not is_data and normalization_factors:
                hist *= normalization_factors.get(sample_name, 1.0)

            if is_data:
                total_data += hist
                total_data_int += np.sum(hist)
            else:
                total_mc += hist
                total_mc_int += np.sum(hist)

        if total_data_int == 0:
            print("⚠️ Atención: No se encontraron muestras de data en este subset de pkls")
            return np.zeros(nbins), np.zeros(nbins)

        # Errores bin a bin
        data_err = np.sqrt(total_data)  # Poisson
        mc_err = np.zeros(nbins)

        if total_mc_int > 0:
            for i in range(nbins):
                n_bin = total_mc[i]
                eff_err = compute_statistical_error(n_bin, total_mc_int)
                mc_err[i] = eff_err * total_mc_int

        # QCD y propagación de incertidumbre
        qcd_raw = total_data - total_mc
        qcd = np.clip(qcd_raw, 0, None)
        qcd_err = np.sqrt(data_err**2 + mc_err**2)

        # Normalización global (como en tu código original)
        data_int = np.sum(total_data)
        mc_int = np.sum(total_mc)
        qcd_sum = np.sum(qcd)
        if data_int > mc_int and qcd_sum > 0:
            S = (data_int - mc_int) / qcd_sum
            qcd *= S
            qcd_err *= S

        return qcd, qcd_err

    if combined_2016:
        pkls_2016 = {k: v for k, v in pkls.items() if k.endswith("_2016")}
        pkls_2016APV = {k: v for k, v in pkls.items() if k.endswith("_2016APV")}

        qcd_2016, err_2016 = compute_qcd_for_subset(pkls_2016)
        qcd_2016APV, err_2016APV = compute_qcd_for_subset(pkls_2016APV)

        qcd_sum = qcd_2016 + qcd_2016APV
        err_sum = np.sqrt(err_2016**2 + err_2016APV**2)
        return qcd_sum, err_sum
    else:
        return compute_qcd_for_subset(pkls)




"""
def transfer_factor_qcd(
    pkls, normalization,
    qcd_ratio,
    binning, distribution,
    consider_overflow, consider_underflow,
    integrated,
    combined_2016
):
  
    qcd_shape_num, qcd_error_num = get_qcd_estimation_shape(pkls[qcd_ratio[0]], binning, distribution, consider_overflow, consider_underflow, normalization[qcd_ratio[0]], combined_2016 = combined_2016)

    qcd_shape_den, qcd_error_den = get_qcd_estimation_shape(pkls[qcd_ratio[1]], binning, distribution, consider_overflow, consider_underflow, normalization[qcd_ratio[1]], combined_2016 = combined_2016)

    if integrated:
        num_integral = np.sum(qcd_shape_num)
        den_integral = np.sum(qcd_shape_den)
        if den_integral == 0:
            raise ZeroDivisionError("QCD denominator integral is zero in transfer factor calculation.")

        TF = num_integral / den_integral
        
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            TF = np.divide(qcd_shape_num, qcd_shape_den, out=np.zeros_like(qcd_shape_num), where=qcd_shape_den!=0)
            
    return TF
"""
def transfer_factor_qcd(
    pkls, normalization,
    qcd_ratio,
    binning, distribution,
    consider_overflow, consider_underflow,
    integrated,
    combined_2016
):
    # Numerador
    qcd_shape_num, qcd_error_num = get_qcd_estimation_shape(
        pkls[qcd_ratio[0]], binning, distribution,
        consider_overflow, consider_underflow,
        normalization[qcd_ratio[0]],
        combined_2016 = combined_2016
    )

    # Denominador
    qcd_shape_den, qcd_error_den = get_qcd_estimation_shape(
        pkls[qcd_ratio[1]], binning, distribution,
        consider_overflow, consider_underflow,
        normalization[qcd_ratio[1]],
        combined_2016 = combined_2016
    )

    if integrated:
        num_integral = np.sum(qcd_shape_num)
        den_integral = np.sum(qcd_shape_den)
        num_err = np.sqrt(np.sum(qcd_error_num**2))  # combinar en cuadratura
        den_err = np.sqrt(np.sum(qcd_error_den**2))

        if den_integral == 0:
            raise ZeroDivisionError("QCD denominator integral is zero in transfer factor calculation.")

        TF = num_integral / den_integral

        # Error propagation
        rel_err2 = 0.0
        if num_integral > 0:
            rel_err2 += (num_err / num_integral) ** 2
        if den_integral > 0:
            rel_err2 += (den_err / den_integral) ** 2

        TF_err = TF * np.sqrt(rel_err2)

        return TF, TF_err

    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            TF = np.divide(qcd_shape_num, qcd_shape_den,
                           out=np.zeros_like(qcd_shape_num),
                           where=qcd_shape_den!=0)

        TF_err = np.zeros_like(TF)

        for i in range(len(binning)-1):
            N, D = qcd_shape_num[i], qcd_shape_den[i]
            dN, dD = qcd_error_num[i], qcd_error_den[i]

            if D > 0 and N > 0:
                rel_err2 = (dN/N)**2 + (dD/D)**2
                TF_err[i] = TF[i] * np.sqrt(rel_err2)
            else:
                TF_err[i] = 0.0

        return TF, TF_err


def get_qcd_estimation(    
    cr_BCD_pkls,
    cr_BCD_normalization,
    qcd_shape,
    qcd_ratio,
    bins: np.ndarray,
    distribution: str,
    consider_overflow: bool = True,
    consider_underflow: bool = True,
    qcd_ratio_integrated: bool = False,
    combined_2016: bool = False
) -> np.ndarray:

    # QCD shape + error
    qcd_shape, qcd_error_shape = get_qcd_estimation_shape(cr_BCD_pkls[qcd_shape], bins, distribution, consider_overflow, consider_underflow, cr_BCD_normalization[qcd_shape], combined_2016 = combined_2016)

    # Transfer factor + error
    qcd_TF, qcd_TF_error = transfer_factor_qcd(pkls = cr_BCD_pkls, normalization =cr_BCD_normalization,
                                 qcd_ratio = qcd_ratio,
                                 binning = bins, distribution = distribution, 
                                 consider_overflow = consider_overflow, consider_underflow = consider_underflow, 
                                 integrated = qcd_ratio_integrated,
                                 combined_2016 = combined_2016)

    # QCD estimation + error
    qcd_estimation = qcd_shape * qcd_TF

    if np.ndim(qcd_shape) == 1:  # caso binned
        qcd_estimation_error = np.sqrt((qcd_TF * qcd_error_shape)**2 +
                                   (qcd_shape * qcd_TF_error)**2)
    else:  # caso integrado
        qcd_estimation_error = np.sqrt((qcd_TF * qcd_error_shape)**2 +
                                       (qcd_shape * qcd_TF_error)**2)
    

    print(" ===================================")
    print(f"QCD transfer factor:")
    print(f" Valor central: {qcd_TF }")
    print(f" Error: {qcd_TF_error}")
    print(" ===================================")
    #print(" QCD estimation using data-driven")
    #print(f" Valor central {qcd_estimation}")
    #print(f" Error: {qcd_estimation_error}")   
    
    

    return qcd_estimation, qcd_estimation_error