import os
import pickle
import numpy as np
from collections import defaultdict

# =========================================================
# Recursive histogram summing (generic)
# =========================================================

def is_hist(obj):
    return (
        isinstance(obj, dict)
        and "edges" in obj
        and "sumw" in obj
        and "sumw2" in obj
    )


def add_hist(acc, h):
    if acc is None:
        return {
            "edges": h["edges"],
            "sumw":  h["sumw"].copy(),
            "sumw2": h["sumw2"].copy(),
        }

    acc["sumw"]  += h["sumw"]
    acc["sumw2"] += h["sumw2"]
    return acc

def is_hist2d(obj):
    return (
        isinstance(obj, dict)
        and "hist" in obj
        and "ST_edges" in obj
        and "nj_edges" in obj
    )


def add_hist2d(acc, h):
    if acc is None:
        return {
            "hist": h["hist"].copy(),
            "ST_edges": h["ST_edges"],
            "nj_edges": h["nj_edges"],
        }

    acc["hist"] += h["hist"]
    return acc


def recursive_sum(src, dst):
    for k, v in src.items():

        if is_hist(v):
            dst[k] = add_hist(dst.get(k), v)

        elif is_hist2d(v):
            dst[k] = add_hist2d(dst.get(k), v)

        elif isinstance(v, (int, float)):
            dst[k] = dst.get(k, 0) + v

        elif isinstance(v, dict):
            if k not in dst:
                dst[k] = {}
            recursive_sum(v, dst[k])



def collapse_entries(data):
    """
    Collapses first-level entries (MET_1, MET_2, etc — whatever they are)
    into a single combined histogram structure.
    """
    total = {}
    for entry in data.values():
        recursive_sum(entry, total)
    return total


# =========================================================
# Deep merge of multiple pkl structures
# =========================================================

def merge_hist(existing, new):
    if isinstance(existing, dict) and isinstance(new, dict):
        merged = dict(existing)
        for k in new:
            merged[k] = merge_hist(merged[k], new[k]) if k in merged else new[k]
        return merged

    if isinstance(existing, np.ndarray) and isinstance(new, np.ndarray):
        return existing + new

    if isinstance(existing, (int, float)) and isinstance(new, (int, float)):
        return existing + new

    return new


# =========================================================
# Main loader: merge + collapse + save
# =========================================================

def load_pkl_files(in_dir: str, out_dir: str):

    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Directory not found: {in_dir}")

    os.makedirs(out_dir, exist_ok=True)

    groups = defaultdict(list)

    prefix_aliases = {
        "WJetsToLNu_ext": "WJetsToLNu_inclusive",
        "DYJetsToLL_M-50_ext": "DYJetsToLL_M-50_inclusive",
        "DYJetsToLL_M-50": "DYJetsToLL_M-50_inclusive",
    }

    # -----------------------------------------------------
    # Collect files
    # -----------------------------------------------------

    for file in os.listdir(in_dir):
        if not file.endswith(".pkl"):
            continue

        file_path = os.path.join(in_dir, file)

        # 🚨 Signal: no merge, just collapse
        if file.startswith("Signal"):
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)

                collapsed = collapse_entries(data)

                out_file = os.path.join(out_dir, file)
                with open(out_file, "wb") as f_out:
                    pickle.dump(collapsed, f_out)

                print(f"✅ Signal copied → {out_file}")
            except Exception as e:
                print(f"❌ Signal error {file}: {e}")
            continue

        # ---------- prefix extraction ----------

        name = file.replace(".pkl", "")
        parts = name.split("_")

        if parts[0] in ["QCD", "WJetsToLNu"] and len(parts) > 2:
            prefix = "_".join(parts[:2])

        elif parts[0] == "DYJetsToLL":
            prefix = "_".join(parts[:3]) if parts[1] == "nlo" else "_".join(parts[:2])

        elif parts[0] == "ST" and len(parts) > 2:
            prefix = "_".join(parts[:4]) if file.startswith("ST_s") else "_".join(parts[:5])

        else:
            prefix = parts[0]

        for alias, target in prefix_aliases.items():
            if prefix.startswith(alias):
                prefix = target
                break

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            groups[prefix].append(data)
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    # -----------------------------------------------------
    # Merge groups, then collapse entries
    # -----------------------------------------------------

    for prefix, hist_list in groups.items():
        if not hist_list:
            continue

        merged = hist_list[0]
        for h in hist_list[1:]:
            merged = merge_hist(merged, h)

        collapsed = collapse_entries(merged)

        out_file = os.path.join(out_dir, f"{prefix}.pkl")
        try:
            with open(out_file, "wb") as f:
                pickle.dump(collapsed, f)
            print(f"✅ Saved → {out_file}")
        except Exception as e:
            print(f"❌ Failed saving {out_file}: {e}")
