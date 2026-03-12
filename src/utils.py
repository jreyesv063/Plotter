def group_samples(sample_keys):
    """
    Groups sample keys into categories based on naming patterns.
    
    Args:
        sample_keys: List or dict_keys of sample names
        
    Returns:
        Dictionary with grouped samples {category: [sample_names]}
    """
    groups = {
        'tt': [],
        'st': [],
        'wj': [],
        'vv': [],
        'dy': [],
        'higgs': [],
        'qcd': [],
        'data': [],
    }

    higgs_samples = {'VBFHToWWTo2L2Nu', 'VBFHToWWToLNuQQ', 'GluGluHToWWToLNuQQ'}
    vv_samples = {'WW', 'WZ', 'ZZ'}
    
    for sample in sample_keys:
        if sample.startswith("Signal"):
            mass_value = sample.split("_")[1]
            group_name = f"SignalTau_{mass_value}"
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(sample)
            
            continue

        # Remove year suffix if present
        base_name = sample.replace("_2016", "").replace("_2016APV", "")

        if base_name.startswith('TTTo'):
            groups['tt'].append(sample)
        elif base_name.startswith('ST'):
            groups['st'].append(sample)
        elif base_name.startswith('WJetsToLNu'):
            groups['wj'].append(sample)
        elif base_name.startswith('DYJetsToLL'):
            groups['dy'].append(sample)
        elif base_name.startswith('QCD'):
            groups['qcd'].append(sample)
        elif base_name in higgs_samples:
            groups['higgs'].append(sample)
        elif base_name in vv_samples:
            groups['vv'].append(sample)
        elif base_name.startswith(("SingleElectron", "SingleMuon", "Tau", "MET")):
            groups['data'].append(sample)
    
    # Remove empty categories
    return {k: v for k, v in groups.items() if v}




def get_rename_map(groups):
    rename_columns = {
        'tt': 'tt',
        'st': 'SingleTop',
        'wj': 'W+jets',
        'vv': 'VV',
        'dy': 'DrellYan+jets',
        'higgs': 'Higgs',
        'qcd': 'QCD',
        'total': 'Total bgr'
    }
    if 'data' in groups:
        data_sources = ', '.join(groups['data'])
        rename_columns['data'] = f"Data ({data_sources})"
    return rename_columns



# =============================================
# New functions for calculating limits
# =============================================

def get_histogram_safe(root_file, hist_name):
    """Recupera un histograma de forma segura."""
    if not root_file or root_file.IsZombie():
        return None
    h = root_file.Get(hist_name)
    if not h or not h.InheritsFrom("TH1"):
        return None
    hist = h.Clone()
    hist.SetDirectory(0)
    return hist


def safe_integral(hist):
    """Calcula integral de histograma de forma segura."""
    if hist:
        try:
            return hist.Integral()
        except:
            return 0.0
    return 0.0

def format_line(syst_name, syst_type, values, max_syst_width, max_val_width, precision=0):
    """Formatea una línea de sistemática."""
    if precision > 0:
        n_decimals = max(0, int(round(-math.log10(precision))))
    else:
        n_decimals = 3

    line = f"{syst_name.ljust(max_syst_width)}  {syst_type.ljust(6)}"

    for val in values:
        if val != "-" and val != "":
            val_float = float(val)

            # Si es entero, imprimir sin decimales
            if val_float.is_integer():
                val_str = str(int(val_float))
            else:
                val_str = f"{val_float:.{n_decimals}f}"
        else:
            val_str = val

        line += f" {val_str.rjust(max_val_width)}"

    return line