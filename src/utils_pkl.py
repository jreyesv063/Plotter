import gc
import os
import re
import psutil
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Any, DefaultDict, Tuple


def get_grouped_sample_paths(samples_folder: str = "2018") -> Dict[str, List[str]]:
    """
    Groups PKL file paths by the base prefix in the filename,
    removing the .pkl extension and optional _<number> suffix.

    Args:
        samples_folder: Directory containing the PKL files.

    Returns:
        A dictionary where keys are sample prefixes and values are lists of file paths.
    """
    
    pkl_files = [f for f in os.listdir(samples_folder) if f.endswith(".pkl")]
    
    if not pkl_files:
        raise ValueError(f"No .pkl files found in: {samples_folder}")

    grouped_paths = defaultdict(list)

    # Regex: matches 'sample_123.pkl' or 'sample.pkl' -> group 'sample'
    pattern = re.compile(r"^(.*?)(?:_\d+)?\.pkl$")

    for filename in pkl_files:
        match = pattern.match(filename)
        if match:
            prefix = match.group(1)
            full_path = os.path.join(samples_folder, filename)
            grouped_paths[prefix].append(full_path)
        else:
            raise ValueError(f"Filename doesn't match expected pattern: {filename}")
    
    return dict(grouped_paths)

    

def remove_corrupted_pkl_files(grouped_paths: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Attempts to open each PKL file. If corrupted, removes it from the grouped path dictionary.
    """
    import gc
    cleaned_paths = defaultdict(list)
    all_files = [(prefix, path) for prefix, files in grouped_paths.items() for path in files]

    for prefix, file_path in tqdm(all_files, desc="Checking PKL files", unit="file"):
        try:
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
            cleaned_paths[prefix].append(file_path)
            del obj
            gc.collect()
        except Exception as e:
            print(f"[Warning] Corrupted or unreadable file removed: {file_path}")
            print(f"          Reason: {str(e)}")

    return dict(cleaned_paths)



def extract_variation_groups(keys: List[str]) -> Dict[str, List[str]]:
    grouped = defaultdict(list)
    for key in keys:
        if key.endswith("Up"):
            base = key[:-2]
            grouped[base].append(key)
        elif key.endswith("Down"):
            base = key[:-4]
            grouped[base].append(key)
    return dict(grouped)

    


def extract_and_save_per_prefix(
    clean_grouped: Dict[str, List[str]], 
    output_dir: str,
    max_memory_gb: float = 14.0,
) -> None:
    """
    Extrae y guarda archivos .pkl por cada sample (prefix), reduciendo el uso de memoria.
    Calcula pesos combinados y descarta pesos base y variaciones tras su uso.
    """
    for prefix, file_list in clean_grouped.items():
        grouped_vars = None  # ← Reset por muestra
        var_dict = defaultdict(list)

        for file_path in file_list:
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)

                top_key = next(iter(data))
                arrays = data[top_key].get("arrays", {})
                keys = list(arrays.keys())

                # Inicializar grouped_vars si no se ha hecho y se encuentran variaciones
                if grouped_vars is None:
                    possible_groups = extract_variation_groups(keys)
                    if possible_groups:
                        grouped_vars = possible_groups

                for k, v in arrays.items():
                    is_weight = k.startswith("weights")
                    is_variation = any(k in v for v in grouped_vars.values()) if grouped_vars else False
                    keep_variables = not is_variation and not is_weight

                    if is_weight or is_variation or keep_variables:
                        try:
                            array_np = np.array(v.value)
                            if array_np.ndim > 0:
                                var_dict[k].append(array_np.astype(np.float32))
                        except Exception as e:
                            print(f"[Warning] Could not convert {k} in {file_path}: {e}")
                        del v

                del data, arrays
                gc.collect()

            except Exception as e:
                print(f"[Error] Failed to load {file_path}: {e}")

        # Concatenar
        concatenated = {}
        for var_name, array_list in var_dict.items():
            try:
                if array_list:
                    concatenated[var_name] = np.concatenate(array_list, dtype=np.float32)
                del array_list
            except Exception as e:
                print(f"[Error] Failed to concatenate {var_name} for {prefix}: {e}")

        del var_dict
        gc.collect()

        # Calcular pesos con variaciones si existen
        if "weights" in concatenated and grouped_vars:
            for base, variations in grouped_vars.items():
                for var in variations:
                    if var.endswith("Up") or var.endswith("Down"):
                        if var in concatenated:
                            try:
                                concatenated[f"weights_{var}"] = concatenated["weights"] * concatenated[var]
                            except Exception as e:
                                print(f"[Error] Failed to compute weights for {var} in {prefix}: {e}")

        # Eliminar variaciones y nominales si solo se usaron para construir pesos
        if grouped_vars:
            for base, variations in grouped_vars.items():
                for var in variations:
                    concatenated.pop(var, None)  # e.g. var_Up, var_Down
                concatenated.pop(base, None)    # base nominal

        gc.collect()

        # Guardar resultado
        output_path = os.path.join(output_dir, f"{prefix}_merged.pkl")
        try:
            with open(output_path, "wb") as f:
                pickle.dump({prefix: concatenated}, f)
            print(f"✅ Saved: {prefix} → {output_path}")
        except Exception as e:
            print(f"[Error] Could not save {prefix}: {e}")

        del concatenated
        gc.collect()
        


def load_pkl_files(samples_folder: str = "2018") -> Dict[str, Dict[str, Any]]:
    """
    Merges multiple PKL files from CMS data analysis into consolidated files.
    
    Args:
        samples_folder: Path to directory containing individual PKL files
                       (default: "2018")
        
    Returns:
        Dictionary containing merged data grouped by sample prefixes
        
    Raises:
        FileNotFoundError: If samples directory doesn't exist
        ValueError: If no PKL files found in directory
    """
    
    # Set up output directory structure
    output_merged_folder = os.path.join(samples_folder, "summary", "pkl")
    os.makedirs(output_merged_folder, exist_ok=True)

    grouped = get_grouped_sample_paths(samples_folder)
    clean_grouped = remove_corrupted_pkl_files(grouped)
    extract_and_save_per_prefix(clean_grouped, output_merged_folder)
    

    gc.collect()

    
    
    



