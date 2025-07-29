import os
import json
import shutil
from collections import defaultdict
from typing import Dict, List, Union, Any

def merge_jsons(existing: Union[Dict, List, float, str], 
                new: Union[Dict, List, float, str]) -> Union[Dict, List, float, str]:
    """
    Recursively merges two JSON-like structures (dicts, lists, or numeric values).
    
    Args:
        existing: The base JSON structure to merge into
        new: The new JSON structure to merge with
        
    Returns:
        The merged JSON structure
        
    Raises:
        ValueError: If incompatible types are encountered during merge
    """
    # Handle dictionary merging
    if isinstance(existing, dict) and isinstance(new, dict):
        merged = defaultdict(dict, existing)
        for key, value in new.items():
            merged[key] = merge_jsons(merged.get(key, {}), value)
        return dict(merged)
    
    # Handle list concatenation
    elif isinstance(existing, list) and isinstance(new, list):
        return existing + new
        
    # Handle numeric addition
    else:
        try:
            # Attempt numeric merge if possible
            return float(existing) + float(new) if existing or new else existing
        except (ValueError, TypeError):
            # Return original value if merge isn't possible
            return existing


def load_json_files(samples_folder: str = "..") -> Dict[str, Dict]:
    """
    Loads and merges metadata JSON files from a samples directory.
    
    Args:
        samples_folder: Path to the directory containing metadata files
        
    Returns:
        Dictionary containing merged metadata grouped by sample prefixes
        
    Raises:
        FileNotFoundError: If metadata directory doesn't exist
    """
    # Initialize container for grouped metadata
    metadata_groups = defaultdict(list)

    # Create output directory for merged files
    output_merged_folder = os.path.join(samples_folder, "summary", "metadata")
    os.makedirs(output_merged_folder, exist_ok=True)

    # Verify metadata directory exists
    metadata_dir = os.path.join(samples_folder, "metadata")
    if not os.path.exists(metadata_dir):
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    merged_groups = {}

    # Process each metadata file
    for file in os.listdir(metadata_dir):
        if not file.endswith("_metadata.json"):
            continue

        file_path = os.path.join(metadata_dir, file)
        
        # If filename starts with 'Signal', use full name (without suffix) as prefix
        if file.startswith("Signal"):
            prefix = file.replace("_metadata.json", "")  # Unique per signal file
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                output_file = os.path.join(output_merged_folder, f"{prefix}.json")
                with open(output_file, "w") as f_out:
                    json.dump(data, f_out, indent=4)
                print(f"✅ Copied without merging (Signal): {output_file}")
                merged_groups[prefix] = data
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error in {file}: {str(e)}")
            except Exception as e:
                print(f"❌ Unexpected error processing {file}: {str(e)}")
            continue  # Skip merging step

        # Extract file prefix based on sample type
        parts = file.split("_")
        if parts[0] in ['QCD', 'WJetsToLNu'] and len(parts) > 2:
            prefix = "_".join(parts[:2])
        elif parts[0] == 'DYJetsToLL' and len(parts) > 2:
            prefix = "_".join(parts[:3])
        elif parts[0] == 'ST' and len(parts) > 2:
            prefix = "_".join(parts[:4]) if file.startswith("ST_s") else "_".join(parts[:5])
        else:
            prefix = parts[0]

        # Load and validate JSON file
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                metadata_groups[prefix].append(data)
            else:
                print(f"⚠️ Warning: {file} contains invalid data (expected dict, got {type(data)})")
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in {file}: {str(e)}")
        except Exception as e:
            print(f"❌ Unexpected error processing {file}: {str(e)}")

    # Merge and save grouped metadata (non-Signal)
    for prefix, entries in metadata_groups.items():
        merged_data = {}
        for entry in entries:
            if not isinstance(entry, dict):
                print(f"❌ Skipping invalid entry in {prefix} (expected dict, got {type(entry)})")
                continue
            for key, value in entry.items():
                merged_data[key] = (
                    merge_jsons(merged_data[key], value)
                    if key in merged_data
                    else value
                )
        output_file = os.path.join(output_merged_folder, f"{prefix}_merged.json")
        try:
            with open(output_file, "w") as f_out:
                json.dump(merged_data, f_out, indent=4)
            print(f"✅ Saved: {prefix} →  {output_file}")
            merged_groups[prefix] = merged_data
        except Exception as e:
            print(f"❌ Failed to save {output_file}: {str(e)}")
