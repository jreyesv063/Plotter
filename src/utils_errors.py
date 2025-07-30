import ROOT
import math
import pandas as pd
from scipy.stats import beta
from scipy.stats import poisson



def calc_bayes_eff_error(numerator: float, denominator: float) -> float:
    """
    Compute Bayesian efficiency uncertainty using ROOT's TGraphAsymmErrors::BayesDivide.
    This exactly replicates the ROOT behavior.
    """
    if denominator == 0:
        return 0.0

    # Create numerator and denominator histograms
    h_num = ROOT.TH1F("h_num", "", 1, 0, 1)
    h_den = ROOT.TH1F("h_den", "", 1, 0, 1)

    h_num.SetBinContent(1, numerator)
    h_den.SetBinContent(1, denominator)
    h_num.Sumw2()
    h_den.Sumw2()

    # Create graph and compute Bayesian efficiency
    g = ROOT.TGraphAsymmErrors()
    g.BayesDivide(h_num, h_den, "b")

    # Extract asymmetric errors
    err_low = g.GetErrorYlow(0)
    err_high = g.GetErrorYhigh(0)

    efficiency = numerator / denominator

    # Follow same logic as your C++ ROOT function
    if err_high > err_low:
        err = err_high
        if err > efficiency:
            err = err_low
    else:
        err = err_low

    # Clean up (optional in notebooks, but good practice)
    del h_num
    del h_den
    del g

    return err


def calc_bin_eff_error(numerator: float, denominator: float) -> float:
    if denominator > 0:
        efficiency = numerator / denominator
        efferror = math.sqrt(efficiency * (1.0 - efficiency) / denominator)
        return efferror
    else:
        return 0.0



def get_table_cutflow_unscaled(json_map, table = "cutflow"):
    pd.set_option('display.float_format', '{:.2f}'.format)

    cutflow_unscaled = {}

    # Obtener cortes base desde el primer dataset que tenga "cutflow"
    try:
        base_cuts = next(
            list(json_map[ds][table].keys()) for ds in json_map if table in json_map[ds]
        )
    except StopIteration:
        raise ValueError(f"❌ Ningún dataset contiene la clave {table}")

    for dataset in json_map:
        cutflow_nominal = json_map[dataset].get(table, {})
        unscaled = {}
        for cut in base_cuts:
            value = cutflow_nominal.get(cut)

            if value is not None:
                try:
                    unscaled[cut] = float(value)
                except (ValueError, TypeError):
                    unscaled[cut] = None
            else:
                #print(f"⚠️  Campo '{cut}' no encontrado en dataset '{dataset}'")
                unscaled[cut] = None

        cutflow_unscaled[dataset] = unscaled

    df = pd.DataFrame.from_dict(cutflow_unscaled, orient="index").transpose()
    return df.round(2)

def compute_eff_cutflow(cutflow_table, normalization):
    result_map = {}
    ratio_data = {}
    scaled_errors = {}

    sumw_row = cutflow_table.loc['sumw']

    for cut in cutflow_table.index:
        result_map[cut] = {}
        ratio_data[cut] = {}
        scaled_errors[cut] = {}

        for sample in cutflow_table.columns:
            numerator = cutflow_table.at[cut, sample]
            denominator = sumw_row[sample]
            norm_factor = normalization.get(sample, 1.0)

            try:
                if cut == 'sumw':
                    ratio = 1.0
                    error = None
                else:
                    ratio = float(numerator) / float(denominator) if denominator else None
                    error = compute_statistical_error(numerator, denominator) if denominator else None
                    #print(f" Sample: {sample} ;  Cut: {cut};  Numerator {numerator}; Denominator {denominator};  Error {error}")
            except (ZeroDivisionError, TypeError, ValueError):
                ratio = None
                error = None


            ratio_data[cut][sample] = ratio

            if error is not None and denominator is not None:
                scaled_error = error * norm_factor * denominator
            else:
                scaled_error = None if cut == 'sumw' else None

            scaled_errors[cut][sample] = scaled_error

            result_map[cut][sample] = {
                'numerator': float(numerator),
                'denominator': float(denominator),
                'ratio': ratio,
                'error_eff': error,
                'normalization': norm_factor * denominator,
                'scaled_errors': scaled_error
            }


    # Crear DataFrames y respetar el orden original
    ratio_df = pd.DataFrame.from_dict(ratio_data, orient="index", columns=cutflow_table.columns)
    scaled_error_df = pd.DataFrame.from_dict(scaled_errors, orient="index", columns=cutflow_table.columns)

    ratio_df = ratio_df.reindex(index=cutflow_table.index)
    scaled_error_df = scaled_error_df.reindex(index=cutflow_table.index)

    return result_map, scaled_error_df

def compute_statistical_error(numerator: float, denominator: float) -> float:
    """
    Compute the statistical uncertainty using standard binomial error,
    but switch to Bayesian error if efficiency is too close to 0 or 1.
    """
    if denominator <= 0:
        return 0.0

    efficiency = numerator / denominator
    eff_err = calc_bin_eff_error(numerator, denominator)

    # Usar Bayes si eficiencia es cercana a 0 o 1
    if efficiency < 0.00001 or efficiency > 0.99999:
        eff_err = calc_bayes_eff_error(numerator, denominator)

    return eff_err
