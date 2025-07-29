import os
import math
import json
import numpy as np
import pandas as pd
from collections import defaultdict


from src.utils import ensure_directory, load_all_pickles, load_all_jsons, get_weights, group_samples, get_rename_map
from src.utils_pkl import load_pkl_files
from src.utils_metadata import load_json_files
from src.utils_plot import get_hist, HistogramPlotter
from src.utils_errors import get_table_cutflow_unscaled, compute_eff_cutflow #calc_bayes_eff_error, calc_bin_eff_error
from src.utils_plot_2D import get_group_hist2d, plot_2d_hist, get_binning_table 
from src.utils_2D_weights import weights_2D

class Plotter:
    def __init__(
        self,
        
        # General configs
        year: str = "2017",                      # Data year to analyze (e.g., "2016", "2016APV", "2017", "2018")
        samples_folder: str = "..",              # Folder with pkl and json files
        output_folder: str = "..",               # Output directory path for saving plots
        lepton_flavor: str = "tau",              # Lepton type to analyze ("tau", "muon", "electron")

        # 2016APV and 2016 combined
        combined_2016: bool = False,             # Combine 2016 data (pre- and post-VFP) for analysis

        # Merge samples
        merge_samples: bool = True,

        # Distirbution
        distribution_with_obj_level_var: str = "lepton_met_mass",
        
        # Efficiency and stadistical error
        systematic_error: bool = False,
        normalized_to: str = "sumw",
        stadistical_error_using: str = "cutflow",

        # root files: limit studies
        created_root_files: bool = False,
        root_files_folder: str = "",

        # Signal samples
        is_SR: bool = False,
        signal_superposition: bool = False
        
    
    ) -> None:

        # Variables used in the plotter methods
        self.year = year
        self.lepton_flavor = lepton_flavor
        self.combined_2016 = combined_2016

        self.distribution_with_obj_level_var = distribution_with_obj_level_var
        
        self.stadistical_error_using = stadistical_error_using

        self.signal = is_SR 
        self.signal_superposition = signal_superposition



        self.output_folder = output_folder
        
        # ------------------------------------------------
        #                  Step 1  
        #      Verify if the sample folders exist
        #   Create output directory and root_file directory
        # ------------------------------------------------

        ensure_directory(samples_folder, must_exist=True, description="Samples directory")
        ensure_directory(output_folder, description="Output directory")

        if created_root_files:
            ensure_directory(root_files_folder, description="ROOT files directory")
        

        # ------------------------------------------------
        #                  Step 2  
        #      Load xsections and luminosities
        # ------------------------------------------------

        try:
            with open("jsons/DAS_xsec.json") as f:
                xsecs = json.load(f)
        
            with open("jsons/luminosity.json") as f:
                luminosity = json.load(f)
        
            if self.combined_2016:
                lumi_2016 = luminosity["2016"]
                lumi_2016APV = luminosity["2016APV"]
            else:
                lumi = luminosity[self.year]
        
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Required JSON file not found: {e}\n"
                "Make sure 'jsons/DAS_xsec.json' and 'jsons/luminosity.json' exist."
            ) from e
        except KeyError as e:
            raise KeyError(
                f"Missing expected key in luminosity.json: {e}\n"
                f"Available keys: {list(luminosity.keys())}"
            ) from e

        self.lumi_map = {
            "2017": "41.5 fb$^{-1}$ (2017, 13 TeV)",
            "2018": "59.8 fb$^{-1}$ (2018, 13 TeV)",
            "2016": "16.8 fb$^{-1}$ (2016, 13 TeV)",
            "2016APV": "19.5 fb$^{-1}$ (2016APV, 13 TeV)"
        }
        
        # ------------------------------------------------
        #     Merge json (metadata) and pkl files
        # ------------------------------------------------
        if merge_samples:

            if self.combined_2016:

                print("\n Loading pkl files for 2016")
                print(f"\n Luminosity {self.lumi_map['2016']}") 

                base_folder = os.path.dirname(samples_folder.rstrip("/"))
                
                samples_folder_2016 = os.path.join(base_folder, "2016")
                samples_folder_2016APV = os.path.join(base_folder, "2016APV")

                # pkl merge
                load_pkl_files(samples_folder = samples_folder_2016)
                print("\n")
                load_json_files(samples_folder= samples_folder_2016)
                

                print("\n")                
                print("Loading pkl files for 2016")
                print(f"\n Luminosity {self.lumi_map['2016APV']}") 
                
                load_pkl_files(samples_folder = samples_folder_2016APV)
                print("\n\n")
                load_json_files(samples_folder= samples_folder_2016APV)

                
            else:

                print(f"\n Loading pkl files for {self.year}")
                print(f"\n Luminosity {self.lumi_map[self.year]}") 
                
                load_pkl_files(samples_folder= samples_folder)
                print("\n\n")
                load_json_files(samples_folder= samples_folder)

        # ------------------------------------------------
        #     Load pkl, json and normalization
        # ------------------------------------------------        
        if self.combined_2016:
            
            pkl_folder_2016APV = os.path.join(samples_folder_2016APV, "summary", "pkl")
            json_folder_2016APV = os.path.join(samples_folder_2016APV, "summary", "metadata")



            print(f" Reading pkl files for 2016APV from: {pkl_folder_2016APV}, and 2016 from  {pkl_folder_2016}")
            pkl_map_2016APV = load_all_pickles(pkl_folder_2016APV)
            json_map_2016APV = load_all_jsons(json_folder_2016APV)
            normalization_2016APV, sumw_2016APV = get_weights(luminosity = lumi, xsecs = xsecs, pkls = pkl_map_2016APV, jsons = json_map_2016APV, normalized_to = normalized_to)



            print(f"\n Reading json files for 2016APV from: {json_folder_2016APV}, and 2016 from {json_folder_2016}")
            pkl_folder_2016 = os.path.join(samples_folder_2016, "summary", "pkl")
            json_folder_2016 = os.path.join(samples_folder_2016, "summary", "metadata")
            
            pkl_map_2016 = load_all_pickles(pkl_folder_2016)
            json_map_2016 = load_all_jsons(json_folder_2016)
            normalization_2016, sumw_2016 = get_weights(luminosity = lumi, xsecs = xsecs, pkls = pkl_map_2016, jsons = json_map_2016, normalized_to = normalized_to)


            # ---- Combined_map 
            self.pkl_map = {
                f"{k}_2016": v for k, v in pkl_map_2016.items()
            }
            self.pkl_map.update({
                f"{k}_2016APV": v for k, v in pkl_map_2016APV.items()
            })

            self.json_map = {
                f"{k}_2016": v for k, v in json_map_2016.items()
            }
            self.jsons_map.update({
                f"{k}_2016APV": v for k, v in json_map_2016APV.items()
            })
        
            self.normalization = {
                f"{k}_2016": v for k, v in normalization_2016.items()
            }
            self.normalization.update({
                f"{k}_2016APV": v for k, v in normalization_2016APV.items()
            })
            
            
            
            print("combined")

        else:
            pkl_folder = os.path.join(samples_folder, "summary", "pkl")
            json_folder = os.path.join(samples_folder, "summary", "metadata")

            print(f" Reading pkl files from: {pkl_folder}")
            print(f" Reading json files from: {json_folder}")
            
            self.pkl_map = load_all_pickles(pkl_folder)
            self.json_map = load_all_jsons(json_folder)
            self.normalization, self.sumw = get_weights(luminosity = lumi, xsecs = xsecs, pkls = self.pkl_map, jsons = self.json_map, normalized_to = normalized_to)


        # ------------------------------------------------
        #     Grouped samples
        # ------------------------------------------------
        self.grouped_samples = group_samples(self.json_map.keys())


        