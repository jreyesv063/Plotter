import ROOT
import numpy as np
import awkward as ak



def calc_bayes_eff_error_scalar(numerator, denominator):
    """Versión escalar (ROOT solo funciona así)"""
    if denominator == 0:
        return 0.0
    
    h_num = ROOT.TH1F("h_num", "", 1, 0, 1)
    h_den = ROOT.TH1F("h_den", "", 1, 0, 1)

    h_num.SetBinContent(1, np.abs(numerator))
    h_den.SetBinContent(1, np.abs(denominator))

    g = ROOT.TGraphAsymmErrors()
    g.BayesDivide(h_num, h_den, "b")

    err_low = g.GetErrorYlow(0)
    err_high = g.GetErrorYhigh(0)

    efficiency = numerator / denominator

    if err_high > err_low:
        err = err_high if err_high <= efficiency else err_low
    else:
        err = err_low

    del h_num, h_den, g
    return err


def calc_bayes_eff_error(numerator, denominator):

    errors = np.zeros_like(numerator)

    for i, n in enumerate(numerator):
        errors[i] = calc_bayes_eff_error_scalar(float(n), float(denominator))

    return errors



def compute_statistical_error(numerator, denominator):
    
    efficiency = np.abs(numerator / denominator)

    # binomial error por defecto
    errors = np.sqrt((efficiency * (1 - efficiency)) / denominator)

    # regiones donde usar Bayes
    bayes_mask = (
        (efficiency < 1e-7) 
    )
    
    bayes_errors = calc_bayes_eff_error(numerator, denominator)

    errors = ak.where(
        bayes_mask,
        bayes_errors,
        errors
    )

    
    return errors


def compute_systematic_error(histos, list_syst_var, distribution, cut):
    sumw_nominal = histos["nominal"]['sumw_all_weights']
    nominal = histos["nominal"][distribution][cut]['sumw']
    eff_nominal = nominal/sumw_nominal
        
    delta_up = {}
    for up_var in list_syst_var['Up']:              
        last_level = histos[up_var]['hist'][distribution]
        sumw_up = histos[up_var]['sumw_all_weights']
        
        if np.isnan(sumw_up):
            sumw_up = sumw_nominal
            
        name_cut = next(k for k in last_level if k.startswith(cut))        
        eff_up = last_level[name_cut]['sumw']/sumw_up   

        delta_up[up_var] = eff_up - eff_nominal

   
    total_syst_up = np.sqrt(
        np.sum(np.array(list(delta_up.values()))**2, axis=0)
    ) 


    delta_down = {}
    for down_var in list_syst_var['Down']:      
        last_level = histos[down_var]['hist'][distribution]
        sumw_down = histos[down_var]['sumw_all_weights']

        if np.isnan(sumw_down):
            sumw_down = sumw_nominal
            
        name_cut = next(k for k in last_level if k.startswith(cut))        
        
        eff_down = last_level[name_cut]['sumw']/sumw_down

        delta_down[down_var] = eff_down - eff_nominal

    total_syst_down = np.sqrt(
        np.sum(np.array(list(delta_down.values()))**2, axis=0)
    ) 

    return total_syst_up, total_syst_down


