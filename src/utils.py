import os
import json
import pickle
from typing import Dict, Any

def ensure_directory(path, must_exist=False, description=""):
    if must_exist and not os.path.exists(path):
        raise FileNotFoundError(
            f"{description} not found at: {path}\n"
            "Please provide a valid path."
        )
    try:
        os.makedirs(path, exist_ok=True)
        print(f"{description} set to: {path}")
    except OSError as e:
        raise OSError(
            f"Failed to create {description} at {path}\n"
            f"Error: {str(e)}"
        ) from e


def load_all_pickles(folder_path: str) -> Dict[str, Any]:
    pkl_map = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    key = os.path.splitext(filename)[0]
                    # Eliminar _merged al final del nombre, si está
                    if key.endswith("_merged"):
                        key = key[:-7]
                    pkl_map[key] = data
            except Exception as e:
                print(f"[Error] Failed to load pickle {filename}: {e}")
    return pkl_map

def load_all_jsons(folder_path: str) -> Dict[str, Any]:
    json_map = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    key = os.path.splitext(filename)[0]
                    # Eliminar _merged al final del nombre, si está
                    if key.endswith("_merged"):
                        key = key[:-7]
                    json_map[key] = data
            except Exception as e:
                print(f"[Error] Failed to load JSON {filename}: {e}")
    return json_map

def get_weights(luminosity: float, xsecs: dict, pkls: dict, jsons: dict, normalized_to: str) -> dict:
    """
    Calculate weights for each sample based on luminosity and cross-sections,
    with validation of input data consistency.
    
    Args:
        luminosity: Integrated luminosity in pb^-1
        xsecs: Dictionary of cross-sections (sample_name: xsec in pb)
        pkls: Dictionary containing processed PKL data (sample_name: data)
        jsons: Dictionary containing JSON metadata (sample_name: metadata)
        
    Returns:
        Dictionary of calculated weights (sample_name: weight)
        
    Raises:
        ValueError: If there's a mismatch between PKLs and JSONs samples
                   If required samples are missing in cross-sections
    """
    
    # First verify that pkls and jsons contain the same samples
    pkl_samples = set(pkls.keys())
    json_samples = set(jsons.keys())
    
    # Check for samples present in one but not the other
    only_in_pkls = pkl_samples - json_samples
    only_in_jsons = json_samples - pkl_samples
    
    if only_in_pkls or only_in_jsons:
        error_msg = "Sample mismatch between PKLs and JSONs:\n"
        if only_in_pkls:
            error_msg += f"- Samples only in PKLs: {sorted(only_in_pkls)}\n"
        if only_in_jsons:
            error_msg += f"- Samples only in JSONs: {sorted(only_in_jsons)}"
        raise ValueError(error_msg)
    
    # Now verify all required samples have cross-sections
    required_samples = pkl_samples - {"SingleElectron", "SingleMuon", "Tau", "MET"}
    missing_xsecs = required_samples - set(xsecs.keys())
    
    if missing_xsecs:
        raise ValueError(
            f"Missing cross-sections for samples: {sorted(missing_xsecs)}\n"
            f"Available samples in xsecs: {sorted(xsecs.keys())}"
        )
    
    # Calculate weights
    weights = {}
    sumw_map = {}
    for sample in pkls.keys():  # We can use either pkls or jsons here since we verified they match
        
        if sample in {"SingleElectron", "SingleMuon", "Tau", "MET"}:
            # Data samples get weight 1
            weights[sample] = 1.0
        else:
            # MC samples: weight = luminosity * cross-section / sum_of_weights
            sumw = jsons[sample][normalized_to]
            sumw_map[sample] = sumw
            weights[sample] = luminosity * xsecs[sample] / sumw
            
    
    print(f"\n MC estimation normalized to {normalized_to}")
    
    return weights, sumw_map



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
        # Skip Signal samples entirely
        if sample.startswith("Signal"):
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