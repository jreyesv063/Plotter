import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import mplhep as hep

# ---------------- Efficiency Calculation ---------------- #
def efficiency_cal(num, den):
    num_value = num
    den_value = den

    # Suponiendo errores Poisson
    num_var = num
    den_var = den

    eff = np.divide(num_value, den_value, out=np.zeros_like(num_value, dtype=float), where=den_value != 0)
    eff_var = (np.divide(num_var, den_value**2, out=np.zeros_like(num_var, dtype=float), where=den_value != 0) +
               np.divide(num_value**2 * den_var, den_value**4, out=np.zeros_like(num_var, dtype=float), where=den_value != 0))
    eff_inc = np.sqrt(eff_var)

    return np.delete(eff, -1), np.delete(eff_inc, -1)

def adjust_eff(eff, eff_inc):
    def adjust(array):
        if array[0] == 0:
            array = array[1:]
        if array[-1] != array[-2]:
            array = np.append(array, array[-1])
        return array

    return adjust(eff), adjust(eff_inc)

def process_efficiency(num, den):
    eff, eff_inc = efficiency_cal(num, den)
    return adjust_eff(eff, eff_inc)

# ---------------- Curve Fit and Error Band ---------------- #
def fit_efficiency_curve(binning, eff_data, a=None, b=None, c=None, d=None):
    x_binning = [(binning[i] + binning[i+1]) / 2 for i in range(len(binning) - 1)]

    def fit_function(x, a, b, c, d):
        return a + b * (1 + erf((np.sqrt(x) - c) / d))

    if a is None:
        a = min(eff_data)
    if b is None:
        b = (max(eff_data) - min(eff_data)) / 2
    if c is None:
        c = np.sqrt(np.median(x_binning))
    if d is None:
        d = 50.0

    guess = [a, b, c, d]

    parameters, covariance_matrix = curve_fit(
        fit_function,
        x_binning,
        eff_data,
        p0=guess,
        maxfev=10000
    )

    random_x = np.linspace(x_binning[0], x_binning[-1], 50000)
    fit_curve = fit_function(random_x, *parameters)
    fit_interp = interp1d(random_x, fit_curve, kind='cubic', fill_value='extrapolate')
    eff_fit = fit_interp(x_binning)

    return eff_fit, x_binning, parameters, covariance_matrix

# ---------------- Helper: Build Efficiency Table ---------------- #
def build_efficiency_table_from_fit(group_num, group_den, binning, main_back, fit_params, cov_matrix):
    import numpy as np
    import pandas as pd
    from scipy.special import erf

    def get_eff(num, den):
        return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den != 0)

    def safe_divide(n1, n2):
        return np.divide(n1, n2, out=np.full_like(n1, np.nan, dtype=float), where=n2 != 0)

    def calculate_binomial_uncertainty(num, den):
        """Calcula incertidumbre binomial con cuidado para divisiones por cero"""
        eff = get_eff(num, den)
        mask = (den > 0) & (eff > 0) & (eff < 1)
        inc = np.zeros_like(eff)
        inc[mask] = np.sqrt(eff[mask] * (1 - eff[mask]) / den[mask])
        return inc

    # --- Samples ---
    all_samples = set(group_num.keys()) & set(group_den.keys())
    mc_samples = [s for s in all_samples if s.lower() != "data"]

    if "data" not in group_num or "data" not in group_den:
        raise ValueError("No se encontró la muestra 'data' en los grupos.")
    if main_back not in group_num or main_back not in group_den:
        raise ValueError(f"El fondo principal '{main_back}' no está presente en los grupos.")

    # --- Data ---
    data_num = group_num["data"]
    data_den = group_den["data"]
    eff_data = get_eff(data_num, data_den)
    eff_inc_data = calculate_binomial_uncertainty(data_num, data_den)

    # --- Main background ---
    main_num = group_num[main_back]
    main_den = group_den[main_back]
    eff_main = get_eff(main_num, main_den)
    eff_inc_main = calculate_binomial_uncertainty(main_num, main_den)

    # --- Total MC ---
    mc_num = sum(group_num[s] for s in mc_samples)
    mc_den = sum(group_den[s] for s in mc_samples)
    eff_mc = get_eff(mc_num, mc_den)
    eff_inc_mc = calculate_binomial_uncertainty(mc_num, mc_den)

    # --- ENFOQUE ROBUSTO para cálculo de pesos ---
    other_bkgs = [s for s in mc_samples if s != main_back]
    other_num = sum(group_num[s] for s in other_bkgs)
    other_den = sum(group_den[s] for s in other_bkgs)
    
    # Calcular eff_diff con cuidado con denominadores cero
    diff_num = data_num - other_num
    diff_den = data_den - other_den
    eff_diff = get_eff(diff_num, diff_den)
    
    # Calcular incertidumbre para eff_diff (propagación más cuidadosa)
    var_data = (data_num * (data_den - data_num)) / np.where(data_den > 0, data_den**3, 1)
    var_other = (other_num * (other_den - other_num)) / np.where(other_den > 0, other_den**3, 1)
    eff_diff_inc = np.sqrt(var_data + var_other)
    
    # Peso central
    weights = safe_divide(eff_diff, eff_main)

    # --- Propagación de error MEJORADA ---
    # Para weight = eff_diff / eff_main
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = (eff_diff_inc**2) / (eff_main**2)
        term2 = (eff_diff**2 * eff_inc_main**2) / (eff_main**4)
        sigma_w = np.sqrt(term1 + term2)
    
    # Manejar casos especiales
    sigma_w = np.nan_to_num(sigma_w, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.nan_to_num(weights, nan=0.0, posinf=5.0, neginf=0.0)

    # Variaciones sin límites artificiales estrictos
    weight_up = weights + sigma_w
    weight_down = weights - sigma_w
    
    # Solo límites físicos razonables
    weight_up = np.clip(weight_up, 0.0, 5.0)    # Máximo 5.0 para ver variaciones
    weight_down = np.clip(weight_down, 0.0, 5.0)

    # --- Binning ---
    x_binning = [(binning[i] + binning[i+1]) / 2 for i in range(len(binning) - 1)]
    bin_ranges = [f"[{binning[i]}, {binning[i+1]})" for i in range(len(binning) - 1)]

    # --- Ajuste de curva (si es necesario) ---
    def eval_fit(x):
        a, b, c, d = fit_params
        return a + b * (1 + erf((np.sqrt(x) - c) / d))

    fit_vals = np.array([eval_fit(x) for x in x_binning]) if fit_params is not None else np.zeros_like(x_binning)

    # --- DataFrame con 3 DECIMALES ---
    df = pd.DataFrame({
        "bin_range": bin_ranges,
        "bin_center": x_binning,
        "eff_data": np.round(eff_data, 4),        # 4 decimales para ver mejor
        "eff_main": np.round(eff_main, 4),
        "eff_total_mc": np.round(eff_mc, 4),
        "eff_diff": np.round(eff_diff, 4),
        "weight": np.round(weights, 4),
        "weight_up": np.round(weight_up, 4),
        "weight_down": np.round(weight_down, 4),
        "sigma_w": np.round(sigma_w, 6),          # 6 decimales para ver la incertidumbre
        "eff_inc_data": np.round(eff_inc_data, 6),
        "eff_inc_main": np.round(eff_inc_main, 6),
        "eff_diff_inc": np.round(eff_diff_inc, 6),
    })

    return df


# ---------------- Run Efficiency Curve ---------------- #
def run_efficiency_curve_plot(
    group_num: dict,
    group_den: dict,
    binning: list,
    main_back: str,
    year: str,
    distribution: str,
    trigger_name: str = "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight",
    folder: str = "plots"
):

    all_samples = set(group_num.keys()) & set(group_den.keys())
    all_mc_samples = [s for s in all_samples if s.lower() != "data"]

    def sum_samples(group: dict, samples: list) -> np.ndarray:
        return sum((group[s] for s in samples if s in group))

    data_num = group_num.get("data")
    data_den = group_den.get("data")
    mc_num = sum_samples(group_num, all_mc_samples)
    mc_den = sum_samples(group_den, all_mc_samples)

    if main_back not in group_num or main_back not in group_den:
        raise ValueError(f"El sample principal '{main_back}' no se encuentra en los grupos")

    main_num = group_num[main_back]
    main_den = group_den[main_back]

    eff_data, eff_inc_data = process_efficiency(data_num, data_den)

    eff_data_fit, x_binning, parameters, covariance = fit_efficiency_curve(
        binning, eff_data,
        a=min(eff_data), b=max(eff_data) - min(eff_data), c=5, d=2
    )

    df_eff_table = build_efficiency_table_from_fit(
        group_num=group_num,
        group_den=group_den,
        binning=binning,
        main_back=main_back,
        fit_params=parameters,
        cov_matrix=covariance
    )


    # Guardar tabla
    #df_eff_table.to_csv(f"eff_table_{main_back}_{year}_{distribution}.csv", index=False)

    # Mostrar el gráfico
    get_plot_efficiency_curve(
        binning=binning,
        data_num=data_num,
        data_den=data_den,
        mc_num=mc_num,
        mc_den=mc_den,
        main_num=main_num,
        main_den=main_den,
        year=year,
        distribution=distribution,
        main_back=main_back,
        trigger_name=trigger_name,
        output_folder = folder
    )

    return df_eff_table


def error_band_fit(x_binning, fit_params, cov_matrix):
    a, b, c, d = fit_params

    def df_da(x): return np.ones_like(x)
    def df_db(x): return 1 + erf((np.sqrt(x) - c) / d)
    def df_dc(x):
        arg = (np.sqrt(x) - c) / d
        return -2 * (b / (d * np.sqrt(np.pi))) * np.exp(-arg**2)
    def df_dd(x):
        arg = (np.sqrt(x) - c) / d
        return -2 * ((b * (np.sqrt(x) - c)) / (d**2 * np.sqrt(np.pi))) * np.exp(-arg**2)

    def derivatives(x):
        return np.vstack([df_da(x), df_db(x), df_dc(x), df_dd(x)]).T

    def variance(x):
        dfdp = derivatives(x)
        return np.sum(dfdp @ cov_matrix * dfdp, axis=1)

    return variance(x_binning)
    
# ---------------- Plotting Function ---------------- #
# ---------------- Plotting ---------------- #
def get_plot_efficiency_curve(
    binning, data_num, data_den, mc_num, mc_den, main_num, main_den,
    year, distribution, main_back, trigger_name,
    output_folder
):
    import numpy as np
    import matplotlib.pyplot as plt
    import hist
    import mplhep as hep
    
    def get_eff(num, den):
        return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den != 0)
    
    def safe_divide(n1, n2):
        return np.divide(n1, n2, out=np.full_like(n1, np.nan, dtype=float), where=n2 != 0)
    
    def process_efficiency(num, den):
        eff = get_eff(num, den)
        # Calcular incertidumbre binomial
        inc = np.zeros_like(eff)
        mask = (den > 0) & (eff > 0) & (eff < 1)
        inc[mask] = np.sqrt(eff[mask] * (1 - eff[mask]) / den[mask])
        return eff, inc

    # --- Calcular eficiencias ---
    eff_data, eff_inc_data = process_efficiency(data_num, data_den)
    eff_mc, eff_inc_mc = process_efficiency(mc_num, mc_den)
    eff_main, eff_inc_main = process_efficiency(main_num, main_den)

    # --- Fit de la curva ---
    eff_data_fit, x_binning, parameters, covariance = fit_efficiency_curve(
        binning, eff_data,
        a=min(eff_data), b=max(eff_data) - min(eff_data), c=5, d=2
    )

    variances = error_band_fit(x_binning, parameters, covariance)
    std_devs = 2 * np.sqrt(variances)

    # x-error
    xerr = np.diff(binning) / 2
    if len(xerr) != len(x_binning):
        xerr = np.append(xerr, xerr[-1])
    xerr = xerr[:len(x_binning)]

    # --- CÁLCULO ROBUSTO DE PESOS ---
    # Reconstruir "Data - Otros fondos" a nivel de conteos
    # Asumiendo que mc_num incluye TODOS los fondos (main + otros)
    other_num = mc_num - main_num  # Otros fondos = Total MC - Main
    other_den = mc_den - main_den
    
    # Calcular eficiencia de (Data - Otros fondos)
    diff_num = data_num - other_num
    diff_den = data_den - other_den
    eff_diff = get_eff(diff_num, diff_den)
    
    # Calcular peso final
    weights = safe_divide(eff_diff, eff_main)
    
    # --- Propagación de error MEJORADA para pesos ---
    # Incertidumbre para eff_diff
    var_data = (data_num * (data_den - data_num)) / np.where(data_den > 0, data_den**3, 1)
    var_other = (other_num * (other_den - other_num)) / np.where(other_den > 0, other_den**3, 1)
    eff_diff_inc = np.sqrt(var_data + var_other)
    
    # Propagación para weight = eff_diff / eff_main
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = (eff_diff_inc**2) / (eff_main**2)
        term2 = (eff_diff**2 * eff_inc_main**2) / (eff_main**4)
        sigma_w = np.sqrt(term1 + term2)
    
    # Manejar casos especiales
    sigma_w = np.nan_to_num(sigma_w, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.nan_to_num(weights, nan=0.0, posinf=5.0, neginf=0.0)

    # --- Ratio fit ---
    ratio_data_fit = eff_data / eff_data_fit
    ratio_inc_data_fit = std_devs
    upper_band = 1 + ratio_inc_data_fit
    lower_band = 1 - ratio_inc_data_fit

    band_up = eff_data_fit + std_devs
    band_down = eff_data_fit - std_devs

    # Extender para bandas
    delta_x = x_binning[-1] - x_binning[-2] if len(x_binning) > 1 else 1.0
    X_fill = np.append(x_binning, x_binning[-1] + delta_x)
    fit_extended = np.append(eff_data_fit, eff_data_fit[-1])
    main_y_up = np.append(band_up, band_up[-1])
    main_y_down = np.append(band_down, band_down[-1])
    ratio_y_down = np.append(lower_band, lower_band[-1])
    ratio_y_up = np.append(upper_band, upper_band[-1])

    # Ratio data/MC
    ratio_data_bgr = eff_data / eff_mc
    ratio_inc_data_bgr = ratio_data_bgr * np.sqrt(
        (eff_inc_data / eff_data) ** 2 + (eff_inc_mc / eff_mc) ** 2
    )

    back_map = {
        "tt": "tt",
        "wj": "W+jets",
        "vv": "VV",
        "dy": "DYJetsToLL",
        "st": "SingleTop"
    }

    fig, (ax, ax_ratio, ax_ratio2, ax_weight) = plt.subplots(
        nrows=4, ncols=1, figsize=(8, 12),
        gridspec_kw={'height_ratios': [3, 1, 1, 1]}
    )
    plt.subplots_adjust(hspace=0.1)

    eff_data_max_error = np.minimum(eff_data + eff_inc_data, 1.0) - eff_data
    eff_mc_max_error = np.minimum(eff_mc + eff_inc_mc, 1.0) - eff_mc
    eff_main_max_error = np.minimum(eff_main + eff_inc_main, 1.0) - eff_main

    # --- Plot 1: Efficiency ---
    ax.errorbar(x_binning, eff_data, xerr=xerr, 
                yerr=(eff_inc_data, eff_data_max_error), fmt='o', 
                color="black", lw=0.5, label="Data")
    ax.errorbar(x_binning, eff_mc, fmt='o', 
                yerr=(eff_inc_mc, eff_mc_max_error), color="blue", 
                lw=0.5, label="Total bgr")
    ax.errorbar(x_binning, eff_main, fmt='o', 
                yerr=(eff_inc_main, eff_main_max_error), color="green", 
                lw=0.5, label=f"Main bgr: {back_map.get(main_back, main_back)}")
    ax.plot(X_fill, fit_extended, label='Fit', color='red')
    ax.fill_between(X_fill, main_y_down, main_y_up, color='cyan', alpha=1, label='Stat. fit unc.')

    # --- Plot 2: Ratio Data/Fit ---
    ax_ratio.errorbar(x_binning, ratio_data_fit, xerr=xerr,
                      yerr=(eff_inc_data / eff_data, eff_data_max_error / eff_data),
                      fmt='o', color="black", lw=0.5)
    ax_ratio.fill_between(X_fill, ratio_y_down, ratio_y_up, color='cyan', alpha=1)
    ax_ratio.axhline(y=1, color='red', linestyle='--')

    # --- Plot 3: Ratio Data/MC ---
    ax_ratio2.errorbar(x_binning, ratio_data_bgr, xerr=xerr,
                       yerr=(ratio_inc_data_bgr, ratio_inc_data_bgr),
                       fmt='o', color="blue", lw=0.5)
    ax_ratio2.axhline(y=1, color='red', linestyle='--')

    # --- Plot 4: PESOS CORRECTOS con barras de error ---
    ax_weight.errorbar(x_binning, weights, xerr=xerr, yerr=sigma_w,
                       fmt='o', color="purple", lw=0.5, label="Weights")
    ax_weight.axhline(y=1, color='red', linestyle='--')
    ax_weight.legend()

    # Formatting
    ax.tick_params(axis='x', labelbottom=False)
    ax_ratio.tick_params(axis='x', labelbottom=False)
    ax_ratio2.tick_params(axis='x', labelbottom=False)

    ax.set_ylabel(r'$\varepsilon$', fontsize=14)
    ax_ratio.set_ylabel(r'$\varepsilon_{(Data)}$ / Fit', fontsize=12)
    ax_ratio2.set_ylabel(r'$\varepsilon_{\mathrm{Data}} / \varepsilon_{\mathrm{Total\;Bgr}}$', fontsize=12)
    ax_weight.set_ylabel("Weight", fontsize=12)

    label_map = {
        "recoil_pt": r'$p_{T}^{miss}(recoil) [GeV]$',
        "pt_nomu_plus": r'$p_{T}^{miss}(\mu) [GeV]$',
        "": r'$p_{T}^{miss} [GeV]$'
    }
    ax_weight.set_xlabel(label_map.get(distribution, distribution), fontsize=12)

    lumi_text = {
        "2017": "41.5 fb$^{-1}$ (2017, 13 TeV)",
        "2018": "59.8 fb$^{-1}$ (2018, 13 TeV)",
        "2016APV": "19.5 fb$^{-1}$ (2016APV, 13 TeV)",
        "2016": "16.8 fb$^{-1}$ (2016, 13 TeV)"
    }
    hep.cms.lumitext(lumi_text.get(year, year), fontsize=12, ax=ax)
    hep.cms.text("Preliminary", loc=0, fontsize=14, ax=ax)
    ax.legend(loc='best', bbox_to_anchor=(1.0, 0.35), ncol=2, fontsize=11)
    ax.text(0.35, 0.5, trigger_name, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', color='hotpink')

    ax.set_ylim(0.0, 1.1)
    ax_ratio.set_ylim(0.5, 1.2)
    ax_ratio2.set_ylim(0.5, 1.2)
    ax_weight.set_ylim(0.0, 3.0)  # Ajustado para pesos de trigger

    filename = f"{output_folder}/eff_{main_back}/{year}/eff_{main_back}_{year}_{distribution}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()