#Author: Andrés Flórez. Idea based on code from Denis Rathjens. 
# Help from DeepSeek for debugging and compressed syntax

import os
import re
import ROOT
import math
import shutil
import traceback
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Local functions
from src.utils import get_histogram_safe, safe_integral, format_line


class Limits:
    def __init__(
        self,      
        # General
        channel: str,
        year: str,
        model: str,
        output_folder: str,        
        
        # Datacards creation
        bgr_tt: str,
        bgr_wj: str,
        signal: str,

        # Limits
        limits_case: str,
    ):

        # --------------------------
        # General
        # --------------------------
        self.channel = channel
        self.year = year
        self.model = model
        
        # --------------------------
        # Datacards creation
        # --------------------------
        self.bgr_tt = bgr_tt
        self.bgr_wj = bgr_wj
        self.signal = signal
        self.output_folder = output_folder

        # Crear carpeta si no existe
        os.makedirs(self.output_folder, exist_ok=True)
        
        # --------------------------
        # Limits
        # --------------------------
        self.limits_case = limits_case

        # --------------------------
        # Available masses en la carpeta signal
        # --------------------------
        self.available_masses = self.get_available_masses(self.signal)



    def get_available_masses(self, root_files):
        """
        Automatically detects the available masses in the signal folder
        based strictly on filenames like: M_1500_wj_tau_2017.root
        """
        masses = []
        
        try:
            # List all files in the signal folder
            files = os.listdir(root_files)
            
            # Strict pattern for names like M_1500_wj_tau_2017.root
            pattern = r'^M_(\d+)_'
            
            for file in files:
                if file.endswith('.root'):
                    match = re.search(pattern, file)
                    if match:
                        mass = int(match.group(1))
                        masses.append(mass)
            
            masses.sort()
            
            if not masses:
                print(f"⚠️ Warning: No masses found in {root_files}")
            else:
                print(f"✅ Detected masses: {masses}")
                
        except FileNotFoundError:
            print(f"❌ Error: Could not access the folder {root_files}")
        except Exception as e:
            print(f"❌ Error detecting masses: {e}")
        
        return masses


    
    def datacards_creation(self, bgr_case, precision):
        """
        Reference for histogram names: 
        
        https://cms-analysis.docs.cern.ch/guidelines/systematics/systematics/systematics_master.yml
        https://cms-analysis.docs.cern.ch/guidelines/uncertainty_digest/BTV/#btv-subjet-tagging-combine-names
        https://cms-analysis-corrections.docs.cern.ch/#2017-ul
        https://cms-analysis.docs.cern.ch/code/
        """        

        print(" =========================================================== ")
        print(f"\t Creating datacards for {bgr_case} ")
        print(" =========================================================== ")
        
        bgr_map = {
            "tt": self.bgr_tt,
            "wj": self.bgr_wj,
            "signal": self.signal            
        }

        if bgr_case not in bgr_map:
            raise ValueError(
                f"bgr_case '{bgr_case}' is not valid. It mist be one of: {list(bgr_map.keys())}"
            )        

        year_map = {
            "2016APV": "2016preVFP",
            "2016": "2016postVFP",
            "2017": "2017",
            "2018": "2018"
        }
        
        
        eospath = bgr_map[bgr_case]
        
        bgrNames = ["DYJetsToLNu", "Higgs", "SingleTop", "WJetToLNu", "tt", "Diboson", "QCD"]
        dataName = ["data_obs"]

        cardName = f"{self.model}_{bgr_case}_{self.channel}_{self.year}"


        systMaster = [
            # Taus
            [f"CMS_fake_t_DeepTau2017v2p1_VSe_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"CMS_fake_t_DeepTau2017v2p1_VSjet_{bgr_case}_{self.channel}_{self.year}", "shape"],    
            [f"CMS_fake_t_DeepTau2017v2p1_VSmu_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"CMS_scale_t_DeepTau2017v2p1_{bgr_case}_{self.channel}_{self.year}", "shape"],  
            # Muons
            [f"CMS_eff_m_id_syst_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"CMS_eff_m_iso_syst_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"CMS_eff_m_iso_syst_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"CMS_scale_m_{bgr_case}_{self.channel}_{self.year}", "shape"],
            # Electrons
            [f"CMS_eff_e_id_13TeV_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"CMS_eff_e_reco_above20_13TeV_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"CMS_eff_e_reco_below20_13TeV_{bgr_case}_{self.channel}_{self.year}", "shape"],
            # Jets
            [f"CMS_scale_j_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"CMS_res_j_{bgr_case}_{self.channel}_{self.year}", "shape"],
            # Bjets
            [f"CMS_btag_fixedWP_bc_simple_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"CMS_btag_fixedWP_light_simple_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"CMS_eff_j_PUJetID_eff_{year_map[self.year]}_{bgr_case}_{self.channel}_{self.year}", "shape"],
            # Fatjets
            [f"CMS_scale_fj_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"CMS_res_fj_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"CMS_eff_j_ParticleNet_W_Nominal_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"CMS_eff_j_ParticleNet_Top_Nominal_{bgr_case}_{self.channel}_{self.year}", "shape"],
            # Pileup
            [f"CMS_pileup_{bgr_case}_{self.channel}_{self.year}", "shape"],
            # L1Prefiring
            [f"CMS_l1_ecal_prefiring_{bgr_case}_{self.channel}_{self.year}", "shape"],
            # MET
            [f"CMS_scale_met_unclustered_energy_{year_map[self.year]}_{bgr_case}_{self.channel}_{self.year}", "shape"],
            # MET trigger 
            [f"CMS_eff_MET_trigger_{bgr_case}_{self.channel}_{self.year}", "shape"],  
            # Particle shower
            [f"ps_isr_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"ps_fsr_{bgr_case}_{self.channel}_{self.year}", "shape"],
            # PDFs
            [f"pdf_alphas_{bgr_case}_{self.channel}_{self.year}", "shape"],
            [f"pdf_lha_{bgr_case}_{self.channel}_{self.year}", "shape"],
            #[f"pdf_lha_alphas_{bgr_case}_{self.channel}_{self.year}", "shape"], # Algo esta pasando con este a la hora de trabajar con señales
            # Top pt reweighting
            [f"top_pt_reweighting_{bgr_case}_{self.channel}_{self.year}", "shape"],        
        ]        

        """
        # Pending implementation 
        lumi = [
            ["lumi_13TeV_correlated", "shape"],  
            ["lumi_2016", "shape"],             
            ["lumi_2017", "shape"],
            ["lumi_2018", "shape"],
        ]

        lumi_uncertainties = {
            "2016APV": {
                "LumiCorrVal": 0.006,
                "LumiStatVal": 0.01
            },
        
            "2016": {
                "LumiCorrVal": 0.006,
                "LumiStatVal": 0.01
            },
        
            "2017": {
                "LumiCorrVal": 0.009,
                "LumiStatVal": 0.02,
                "EleHLTzvtx": 0.991,
                "EleHLTzvtxUnc": 0.001,
                "Lumi1718Val": 0.006
            },
        
            "2018": {
                "LumiCorrVal": 0.02,
                "LumiStatVal": 0.015,
                "Lumi1718Val": 0.002
            }
        }

        # https://xsecdb-xsdb-official.app.cern.ch/xsdb/?columns=67108863&currentPage=0&pageSize=50
        xsection = [
            [f"cross_section_{sample_name}", "shape"]
        ]
        """

        # ---------------------------
        # LOOP OVER MASSES
        # ---------------------------

        for mass in self.available_masses:
            try:
                
                print(f"\n>>> Creating card for M={mass}")
        
                bin_name = f"{cardName}_{mass}"
                output_path = f"{self.output_folder}/{cardName}_M_{mass}.txt"

                # ---------------------------
                # Loading histogram information
                # ---------------------------
                
                 # Open data file
                data_file = ROOT.TFile.Open(f"{eospath}/data_obs_{bgr_case}_{self.channel}_{self.year}.root")
                data_hist = get_histogram_safe(data_file, f"data_obs_{bgr_case}_{self.channel}_{self.year}_nom")
        
                # Open background files
                background_files = {}
                background_hists = {}
                for PROCESS in bgrNames:
                    file_path = f"{eospath}/{PROCESS}_{bgr_case}_{self.channel}_{self.year}.root"
                    background_files[PROCESS] = ROOT.TFile.Open(file_path, "READ")
                    background_hists[PROCESS] = {
                        'nom': get_histogram_safe(background_files[PROCESS], f"{PROCESS}_{bgr_case}_{self.channel}_{self.year}_nom")
                    }
        
                # Open signal file
                signal_name = f"M_{mass}"
                signal_file_path = f"{eospath}/{signal_name}_{bgr_case}_{self.channel}_{self.year}.root"
                signal_file = ROOT.TFile.Open(signal_file_path, "READ")
                signal_hist_nom = get_histogram_safe(signal_file, f"{signal_name}_{bgr_case}_{self.channel}_{self.year}_nom")
        
                # ---------------------------
                # WRITE THE CARD
                # ---------------------------
        
                print(f"Creating Combine card file: {output_path}")
        
                with open(output_path, "w") as f:
                    # Cabecera
                    f.write("imax 1\n")
                    f.write(f"jmax {len(bgrNames)}\n")
                    f.write("kmax *\n")
                    f.write("----------\n")
                    f.write(f"bin {bin_name}\n")
                    f.write("----------\n")
                    f.write("observation -1\n")
                    f.write("----------\n")
        
                    # Section shapes
                    f.write(f"shapes data_obs {bin_name} {eospath}/data_obs_{bgr_case}_{self.channel}_{self.year}.root data_obs_{bgr_case}_{self.channel}_{self.year}_nom\n")
                    f.write(f"shapes * {bin_name} {eospath}/$PROCESS_{bgr_case}_{self.channel}_{self.year}.root $PROCESS_{bgr_case}_{self.channel}_{self.year}_nom $SYSTEMATIC\n")
                    f.write("----------\n")
        
                    # Process names and rates
                    allNames = [f"M_{mass}"] + bgrNames
                    allNumbers = [0] + list(range(1, len(bgrNames) + 1))
        
                    rate_values = []
                    rate_values.append(str(Decimal(safe_integral(signal_hist_nom))))
                    for bkg_name in bgrNames:
                        rate_values.append(str(Decimal(safe_integral(background_hists[bkg_name]['nom']))))
        
                    max_name_width = max(max(len(name) for name in allNames) + 2, 10)
        
                    f.write("bin" + "".join([f" {bin_name.rjust(max_name_width)}" for _ in allNames]) + "\n")
                    f.write("process" + "".join([f" {name.rjust(max_name_width)}" for name in allNames]) + "\n")
                    f.write("process" + "".join([f" {str(num).rjust(max_name_width)}" for num in allNumbers]) + "\n")
                    f.write("rate" + "".join([f" {float(rate):>{max_name_width}.3f}" for rate in rate_values]) + "\n")
                    f.write("----------\n")
        
                    # ---------------------------
                    # Systematic
                    # ---------------------------
                    max_syst_width = max(len(s[0]) for s in systMaster) + 2
                    max_val_width = 10
        
                    # Build systematic structure: - means it is not relevant, 1 is relevant
                    data = {proc: [("-", "-")] * len(systMaster) for proc in allNames}
        
                    for j, (syst_name, syst_type) in enumerate(systMaster):
                        for i, proc_name in enumerate(allNames):
                            if i == 0:  # Señal
                                hist_up = get_histogram_safe(signal_file, f"{syst_name}Up")
                                hist_down = get_histogram_safe(signal_file, f"{syst_name}Down")
                                rate = Decimal(safe_integral(signal_hist_nom))
                            else:  # Background
                                hist_up = get_histogram_safe(background_files[proc_name], f"{syst_name}Up")
                                hist_down = get_histogram_safe(background_files[proc_name], f"{syst_name}Down")
                                rate = Decimal(safe_integral(background_hists[proc_name]['nom']))
        
                            if hist_up and hist_down and rate > 0:
                                up_val = Decimal(safe_integral(hist_up)) / rate
                                down_val = Decimal(safe_integral(hist_down)) / rate
                                val_up = "-" if abs(up_val - 1) <= Decimal(precision) else 1
                                val_down = "-" if abs(down_val - 1) <= Decimal(precision) else 1
        
                            else:
                                val_up, val_down = "-", "-"
        
                            data[proc_name][j] = (val_up, val_down)
        
                    # Escribir sistemáticas en una sola línea por nombre
                    for j, (syst_name, syst_type) in enumerate(systMaster):
                        values_combined = []
                        for i, proc_name in enumerate(allNames):
                            up, down = data[proc_name][j]
                            if up != "-" and down != "-":
                                val = max(float(up), 1/float(down))
                                values_combined.append(val)
                            elif up != "-":
                                values_combined.append(float(up))
                            elif down != "-":
                                values_combined.append(float(down))
                            else:
                                values_combined.append("-")
                        line = format_line(syst_name, syst_type, values_combined, max_syst_width, max_val_width)
                        f.write(line + "\n")
        
                    # autoMCStats
                    f.write(f"\n{bin_name} autoMCStats 10\n")
        
                # ---------------------------
                # Cerrar archivos
                # ---------------------------
                data_file.Close()
                for bkg_file in background_files.values():
                    bkg_file.Close()
                signal_file.Close()
        
            except Exception as e:
                print(f"Error processing mass {mass}: {e}")
                traceback.print_exc()
                continue



    def limits_estimation_plot(self):
        """
        Creates a bash script to run combineCards.py for each mass.
        Input datacards are located in self.output_folder.
        Combined datacards will be written into self.output_folder/combined.
        Also copies brazilian_plot.py from the known src path into combined/
        AND runs python3 brazilian_plot.py automatically at the end.
        """
        
        # Directory where combined datacards will be stored
        combined_dir = os.path.join(self.output_folder, "combined")
        os.makedirs(combined_dir, exist_ok=True)
    
        # Path of the bash script to generate
        bash_path = os.path.join(combined_dir, "create_combined_datacards.sh")
    
        # Convert list of masses into a space-separated string
        mass_list = " ".join(str(m) for m in self.available_masses)
    
        # Absolute path to the directory containing input datacards
        input_dir = os.path.abspath(self.output_folder)
    
        # ---------------------------------------------------------------------
        #  PATH TO THE SRC DIRECTORY IN EOS
        # ---------------------------------------------------------------------
        root_folder = os.getcwd()
        src_dir = os.path.join(root_folder, "src")
        brazilian_src = os.path.join(src_dir, "brazilian_plot.py")
    
        found = os.path.isfile(brazilian_src)
    
        # ---------------------------------------------------------------------
        # Generate the .sh script
        # ---------------------------------------------------------------------
        bash_content = f"""#!/bin/bash
    
    # Create combined datacards
    for mass in {mass_list}; do
      combineCards.py \\
        signal="{input_dir}/wprime_signal_tau_2017_M_${{mass}}.txt" \\
        tt="{input_dir}/wprime_tt_tau_2017_M_${{mass}}.txt" \\
        wj="{input_dir}/wprime_wj_tau_2017_M_${{mass}}.txt" \\
        > "{combined_dir}/combined_M${{mass}}.txt"
    
      echo "Created combined_M${{mass}}.txt in {combined_dir}"
    done
    
    # Copy brazilian_plot.py from the fixed src path
    if [ -f "{brazilian_src}" ]; then
        cp "{brazilian_src}" "{combined_dir}/"
        echo "Copied brazilian_plot.py to {combined_dir}"
    else
        echo "WARNING: brazilian_plot.py not found at {brazilian_src}"
    fi
    
    echo "Running brazilian_plot.py..."
    cd "{combined_dir}"
    python3 brazilian_plot.py
    """
    
        # Write script to file
        with open(bash_path, "w") as f:
            f.write(bash_content)
    
        os.chmod(bash_path, 0o755)
    
        # Optional immediate copy from Python (not required but helpful)
        if found:
            try:
                shutil.copy(brazilian_src, combined_dir)
                print(f"Copied brazilian_plot.py to {combined_dir} (from Python).")
            except Exception as e:
                print("ERROR copying brazilian_plot.py:", e)
        else:
            print("WARNING: brazilian_plot.py was NOT found at the expected location.")
    
        print(f"\nBash script created at: {bash_path}")
        print("*** Run create_combined_datacards.sh in your lxplus account***\n")



    