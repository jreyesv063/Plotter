import os
import json
from collections import defaultdict
from typing import Dict, List, Union


def merge_jsons(existing: Union[Dict, List, float, str],
                new: Union[Dict, List, float, str],
                parent_key: str = None) -> Union[Dict, List, float, str]:
    """
    Recursively merges two JSON-like structures (dicts, lists, or numeric values).
    Special case: the whole dictionary under "selections" is overwritten, not merged.
    """
    # 🚨 Special case: overwrite entire selections dictionary
    if parent_key == "selections":
        return new

    # Handle dictionary merging
    if isinstance(existing, dict) and isinstance(new, dict):
        merged = defaultdict(dict, existing)
        for key, value in new.items():
            merged[key] = merge_jsons(merged.get(key, {}), value, parent_key=key)
        return dict(merged)

    # Handle list concatenation
    elif isinstance(existing, list) and isinstance(new, list):
        return existing + new

    # Handle numeric addition
    else:
        try:
            return float(existing) + float(new) if existing or new else existing
        except (ValueError, TypeError):
            return new



def load_json_files(input_base: str, output_base: str) -> Dict[str, Dict]:

    metadata_dir = os.path.join(input_base, "metadata")
    out_dir = os.path.join(output_base, "metadata")

    if not os.path.exists(metadata_dir):
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    os.makedirs(out_dir, exist_ok=True)

    metadata_groups = defaultdict(list)
    merged_groups = {}

    prefix_aliases = {
        "WJetsToLNu_ext": "WJetsToLNu_inclusive",
        "DYJetsToLL_M-50_ext": "DYJetsToLL_M-50_inclusive",
        "DYJetsToLL_M-50": "DYJetsToLL_M-50_inclusive",
    }

    # --------------------------------------------------
    # Load files
    # --------------------------------------------------

    for file in os.listdir(metadata_dir):
        if not file.endswith("_metadata.json"):
            continue

        file_path = os.path.join(metadata_dir, file)

        # 🚨 Signal: copy as-is
        if file.startswith("Signal"):
            prefix = file.replace("_metadata.json", "")
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                out_file = os.path.join(out_dir, f"{prefix}.json")
                with open(out_file, "w") as f_out:
                    json.dump(data, f_out, indent=4)

                print(f"✅ Signal copied → {out_file}")
                merged_groups[prefix] = data
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
            continue

        # ---------- prefix extraction ----------

        parts = file.split("_")

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
            with open(file_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                metadata_groups[prefix].append(data)
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    # --------------------------------------------------
    # Merge and save
    # --------------------------------------------------

    for prefix, entries in metadata_groups.items():
        merged_data = {}

        for entry in entries:
            for key, value in entry.items():
                if key in merged_data:
                    merged_data[key] = merge_jsons(
                        merged_data[key],
                        value,
                        parent_key=key
                    )
                else:
                    merged_data[key] = value

        out_file = os.path.join(out_dir, f"{prefix}.json")
        try:
            with open(out_file, "w") as f_out:
                json.dump(merged_data, f_out, indent=4)

            print(f"✅ Saved → {out_file}")
            merged_groups[prefix] = merged_data
        except Exception as e:
            print(f"❌ Failed saving {out_file}: {e}")


