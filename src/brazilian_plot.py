import ROOT
import os, re, sys
from array import array

class BrazilianPlot:
    def __init__(self, year: str, binS: str, channel: str):

        # ================================
        #       Year
        # ================================
        yearOptions = ["2016preVFP", "2016postVFP", "2017", "2018", "all"]
        lumi_map = {
            "2016preVFP": "19.5",
            "2016postVFP": "16.8",
            "2017": "41.5",
            "2018": "59.8",
            "all": "138",
        }
        if year not in yearOptions:
            raise ValueError(f"Invalid year. It must be one of: {yearOptions}")
        else:
            self.year = year
            self.lumi = lumi_map[year]

        # ================================
        #       Xsection
        # ================================ 
        self.map_xsec = {
            300: 91.7,
            400: 30.2,
            600: 5.97,
            750: 2.32,
            1000: 0.646,
            1500: 0.0891,
            2000: 0.0182,
            3000: 0.00134
        }
        
        # ================================
        #       General
        # ================================        
        self.binS = binS
        self.channel = channel

        

    def detect_masses(self, directory="."):
        """
        Scan the given directory and return a list of available masses
        based on files named 'combined_M<value>.txt'.
        """
        masses = []
        pattern = re.compile(r"combined_M(\d+)\.txt")
        
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                masses.append(int(match.group(1)))
    
        return sorted(masses)

    
    def get_datacard_path(self, mass):
        """
        Returns the path to the combined datacard for a given mass.
        Raises FileNotFoundError if the file does not exist.
        """
        path = f"combined_M{mass}.txt"
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Datacard for mass {mass} not found: {path}")
        return path
        
        
    def main(self, runBlinded, grid):
        """
        Generate the Brazilian plot for W' -> tau nu
        """
    
        print("===========================================")
        print("\t BRAZILIAN PLOT for W'")
        print("===========================================")
        
        print(f"Luminosity: {self.lumi} fb-1. \t year: {self.year}")
    
        # Detect available masses
        available_masses = self.detect_masses()
        print(f"Available masses: {available_masses}")
    
        # --------------------------------------
        # Extract limits from combine
        # --------------------------------------
        limitNumbers = []  # List of lists: each element = [ -2σ, -1σ, median, +1σ, +2σ ]
        valid_masses = []
    
        for mass in available_masses:
            print(f"\n--- W' mass = {mass} GeV ---")
            datacard = self.get_datacard_path(mass)
            cmd = f"combine -M AsymptoticLimits --run blind  -m {mass} {datacard}"
            print(f"\t Running: {cmd}")
            result = os.system(cmd)
    
            if result != 0:
                print(f" Combine failed: {result}")
                continue
    
            rootfile = f"higgsCombineTest.AsymptoticLimits.mH{mass}.root"
            if not os.path.exists(rootfile):
                print(f" ERROR: {rootfile} was not generated.")
                continue
    
            infile = ROOT.TFile(rootfile, "READ")
            tree = infile.Get("limit")
    
            limits = []
            for event in tree:
                xs_limit = event.limit * self.map_xsec[mass]
                limits.append(xs_limit)
            infile.Close()
    
            if len(limits) != 5:
                print(f" WARNING: Unexpected number of limits ({len(limits)}) for mass {mass}. Skipping.")
                continue
    
            # Print limits for this mass
            print(" ********************************************* ")
            print(f"\t Limits (pb) for m(W') = {mass} GeV:")
            print(f"\t\t -2_sigma: {limits[0]:.6f}")
            print(f"\t\t -1_sigma: {limits[1]:.6f}")
            print(f"\t\t Median: {limits[2]:.6f}")
            print(f"\t\t +1_sigma: {limits[3]:.6f}")
            print(f"\t\t +2_sigma: {limits[4]:.6f}")
            print(" ********************************************* ")
    
            limitNumbers.append(limits)
            valid_masses.append(mass)
    
        if len(valid_masses) == 0:
            print("No valid masses found. Exiting.")
            return
    
        # --------------------------------------
        # Prepare arrays for TGraph
        # --------------------------------------
        central = array('d', [limits[2] for limits in limitNumbers])
        theory  = array('d', [self.map_xsec[mass] for mass in valid_masses])
    
        # ±1σ and ±2σ bands (front + back)
        oneSigmaBandX = array('d', [])
        oneSigmaBandY = array('d', [])
        twoSigmaBandX = array('d', [])
        twoSigmaBandY = array('d', [])
    
        # Front: lower edge (-2σ, -1σ)
        for i, limits in enumerate(limitNumbers):
            mass = valid_masses[i]
            twoSigmaBandX.append(mass)
            twoSigmaBandY.append(limits[0])
            oneSigmaBandX.append(mass)
            oneSigmaBandY.append(limits[1])
    
        # Back: upper edge (+1σ, +2σ) reversed
        for i in reversed(range(len(limitNumbers))):
            mass = valid_masses[i]
            limits = limitNumbers[i]
            twoSigmaBandX.append(mass)
            twoSigmaBandY.append(limits[4])
            oneSigmaBandX.append(mass)
            oneSigmaBandY.append(limits[3])
    
        valid_masses_array = array('d', valid_masses)


        # --------------------------------------
        # Draw Brazilian plot
        # --------------------------------------
        # Create canvas
        canvas = ROOT.TCanvas("canvas", r"W' $\rightarrow \tau\nu$ Limits", 800, 600)
        canvas.cd()
        canvas.SetLeftMargin(0.12)
        canvas.SetRightMargin(0.05)
        canvas.SetBottomMargin(0.12)
        canvas.SetTopMargin(0.08)
        canvas.SetGrid()
        
        n_valid = len(valid_masses)
        
        # Create the graphs first
        CentralGraph = ROOT.TGraph(n_valid, valid_masses_array, central)
        OneSigmaGraph = ROOT.TGraph(2*n_valid, oneSigmaBandX, oneSigmaBandY)
        TwoSigmaGraph = ROOT.TGraph(2*n_valid, twoSigmaBandX, twoSigmaBandY)
        TheoryGraph = ROOT.TGraph(n_valid, valid_masses_array, theory)
        
        # Determine appropriate axis ranges
        x_min = min(valid_masses)
        x_max = max(valid_masses)
        
        # Find the maximum value among limits to set Y range
        all_limits = []
        for limits in limitNumbers:
            all_limits.extend(limits)
        y_max = max(all_limits) * 1.3  # 30% margin
        y_min = 0.0  # Start from 0 for limits
        
        # If all values are small, adjust y_min
        if min(all_limits) > 0:
            y_min = min(all_limits) * 0.5
        
        # Create a frame histogram to define axes
        frame_hist = ROOT.TH1F("frame_hist", "frame_hist", 100, x_min, x_max)
        frame_hist.SetMinimum(y_min)
        frame_hist.SetMaximum(y_max)
        frame_hist.SetStats(0)
        frame_hist.SetTitle(r";m_{W'} (GeV);95% CL upper limit (pb)")
        
        x_margin = (x_max - x_min) * 0.05
        frame_hist.GetXaxis().SetLimits(x_min - x_margin, x_max + x_margin)
        frame_hist.GetXaxis().SetTitleSize(0.045)
        frame_hist.GetXaxis().SetLabelSize(0.04)
        frame_hist.GetXaxis().CenterTitle(True)
        frame_hist.GetXaxis().SetTitleOffset(1.1)
        
        frame_hist.GetYaxis().SetTitleSize(0.045)
        frame_hist.GetYaxis().SetLabelSize(0.04)
        frame_hist.GetYaxis().CenterTitle(True)
        frame_hist.GetYaxis().SetTitleOffset(1.3)
        frame_hist.GetYaxis().SetTitle("95% CL upper limits median exp. (pb)")
        

        # Draw frame histogram 
        if grid:
            frame_hist.Draw("AXIG")  # AXIS + Grid
        else:
            frame_hist.Draw("AXIS")        
        #frame_hist.Draw("AXIG")  # AXIS + Grid
        
        # Graph styles
        CentralGraph.SetLineColor(ROOT.kBlack)
        CentralGraph.SetLineStyle(1)  # Solid line
        CentralGraph.SetLineWidth(2)
        
        # Use standard opaque colors for bands
        OneSigmaGraph.SetFillColor(ROOT.kGreen+1)
        OneSigmaGraph.SetFillStyle(1001)  # Solid
        OneSigmaGraph.SetLineWidth(0)      # No border
        
        TwoSigmaGraph.SetFillColor(ROOT.kYellow)
        TwoSigmaGraph.SetFillStyle(1001)   # Solid
        TwoSigmaGraph.SetLineWidth(0)      # No border
        
        TheoryGraph.SetLineColor(ROOT.kRed)
        TheoryGraph.SetLineWidth(3)
        TheoryGraph.SetLineStyle(2)  # Dashed line
        
        # Draw in the correct order (widest band first)
        TwoSigmaGraph.Draw("F SAME")
        OneSigmaGraph.Draw("F SAME")
        CentralGraph.Draw("L SAME")
        TheoryGraph.Draw("L SAME")
        
        # Redraw axes
        if grid:
            frame_hist.Draw("AXIGS SAME")  # AXIS + Grid
        else:
            frame_hist.Draw("AXIS SAME")   
            
        #frame_hist.Draw("AXIS SAME")
        #frame_hist.Draw("AXIGS SAME")  # AXIS + Grid
        canvas.RedrawAxis()

        
        # --------------------------------------
        # Legend - make it more visible
        # --------------------------------------
        # Move legend higher and to the left
        legend = ROOT.TLegend(0.55, 0.70, 0.89, 0.89)
        legend.SetHeader("95% CL upper limits", "C")
        legend.SetBorderSize(0)  # No border for cleaner look
        legend.SetFillStyle(0)   # Transparent
        legend.SetTextSize(0.035)
        legend.SetTextFont(42)
        legend.SetFillColor(0)
        legend.SetLineColor(0)
        legend.AddEntry(CentralGraph, "Median expected", "l")
        legend.AddEntry(OneSigmaGraph, "68% expected", "f")
        legend.AddEntry(TwoSigmaGraph, "95% expected", "f")
        legend.AddEntry(TheoryGraph, r"W' #rightarrow #tau#nu theory", "l")
        legend.Draw()
        
        # Professional CMS text
        lumi_text = ROOT.TLatex()
        lumi_text.SetNDC()
        lumi_text.SetTextFont(42)
        lumi_text.SetTextSize(0.035)
        lumi_text.DrawLatex(0.16, 0.955, f"{self.lumi} fb^{{-1}} ({self.year}, 13 TeV)")
        
        # CMS in bold
        collab_text = ROOT.TLatex()
        collab_text.SetNDC()
        collab_text.SetTextFont(61)  # 61 = Helvetica-Bold
        collab_text.SetTextSize(0.045)
        collab_text.DrawLatex(0.16, 0.885, "CMS")
        
        # "Work in Progress" in smaller font
        status_text = ROOT.TLatex()
        status_text.SetNDC()
        status_text.SetTextFont(52)  # 52 = Helvetica-Italic
        status_text.SetTextSize(0.035)
        status_text.DrawLatex(0.16, 0.835, "Simulation Preliminary")
        
        # Update canvas
        canvas.SetGridx()
        canvas.SetGridy()
        canvas.SetLogy(1)
        canvas.Update()
        canvas.RedrawAxis()
        
        # --------------------------------------
        # Save outputs
        # --------------------------------------
        output_base = f"Wprime_{self.channel}_{self.year}"
        if runBlinded:
            output_base += "_blinded"
        
        # Save in multiple formats
        canvas.SaveAs(f"{output_base}.pdf")
        print(f"Plot saved as: {output_base}.pdf")

        # --------------------------------------
        # Summary
        # --------------------------------------
        print("\n")
        print("SUMMARY OF LIMITS (pb)")
        print(f"{'Mass [GeV]':<12} {'Theory [pb]':<14} {'Median':<14} {'-2σ':<14} {'-1σ':<14} {'+1σ':<14} {'+2σ':<14}")
        print("-" * 90)
        
        for i, mass in enumerate(valid_masses):
            limits = limitNumbers[i]
            theory_val = self.map_xsec[mass]
        
            print(
                f"{mass:<12d}"
                f"{theory_val:<14.6f}"
                f"{limits[2]:<14.6f}"
                f"{limits[0]:<14.6f}"
                f"{limits[1]:<14.6f}"
                f"{limits[3]:<14.6f}"
                f"{limits[4]:<14.6f}"
            )
        

if __name__ == "__main__":  
    plot = BrazilianPlot(year="2017", binS= "All", channel="tau")
    plot.main(runBlinded = True, grid=False)
