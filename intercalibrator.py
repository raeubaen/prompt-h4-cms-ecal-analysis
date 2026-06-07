import os,json,uproot,argparse,sys,ROOT
import numpy as np
import array
import glob
from math import sqrt
import csv
import ctypes
import math

def has_branch(fname, branch):
    f = ROOT.TFile.Open(fname)
    if not f or f.IsZombie():
        return False
    t = f.Get("tree")
    if not t:
        return False

    return t.GetBranchStatus(branch)

def cbFit(h,name,Run,output_dir,seed_channel,xmin=-1,xmax=-1):

    x = ROOT.RooRealVar(f"x_{name}_{Run}", "E/E_{True}", h.GetXaxis().GetXmin(), h.GetXaxis().GetXmax())

    data = ROOT.RooDataHist(f"data_{name}_{Run}", "data", ROOT.RooArgList(x), h)

    peak = h.GetBinCenter(h.GetMaximumBin())

    mean  = ROOT.RooRealVar(f"mean_{name}", "DCB mean",peak,peak-3,peak+3)

    sigma = ROOT.RooRealVar(f"sigma_{name}", "DCB sigma",h.GetRMS(),0.1*h.GetRMS(),5*h.GetRMS())

    alphaL = ROOT.RooRealVar(f"alphaL_{name}", "alphaL", 1.5, 0.1, 5.0)
    nL     = ROOT.RooRealVar(f"nL_{name}",     "nL",     3.0, 0.5, 20.0)

    alphaR = ROOT.RooRealVar(f"alphaR_{name}", "alphaR", 1.5, 0.1, 5.0)
    nR     = ROOT.RooRealVar(f"nR_{name}",     "nR",     3.0, 0.5, 20.0)

    dcb = ROOT.RooCrystalBall(f"dcb_{name}", "Double Crystal Ball",x,mean,sigma,alphaL, nL,alphaR, nR)

    nsig = ROOT.RooRealVar(f"nsig_{name}", "signal yield",h.Integral(),0.0,10.0*h.Integral())
    model = ROOT.RooAddPdf(f"model_{name}_{Run}", "extended DCB model",ROOT.RooArgList(dcb),ROOT.RooArgList(nsig))

    fitArgs = [
        ROOT.RooFit.Extended(True),
        ROOT.RooFit.Save(),
        ROOT.RooFit.PrintLevel(-1)
    ]

    if xmin >= 0 and xmax >= 0:
        fitArgs.insert(0, ROOT.RooFit.Range("fitRange"))
        x.setRange("fitRange", xmin, xmax)

    result = model.fitTo(data, *fitArgs)

    c = ROOT.TCanvas()

    frame = x.frame()
    data.plotOn(frame)
    model.plotOn(frame, ROOT.RooFit.Range("fitRange"),ROOT.RooFit.NormRange("fitRange"))

    frame.Draw()

    chi2 = frame.chiSquare()

    pt = ROOT.TPaveText(0.60, 0.65, 0.88, 0.88, "NDC")
    pt.SetFillColor(0)
    pt.SetTextFont(42)
    pt.SetBorderSize(0)
    pt.SetTextSize(0.05)

    pt.AddText(f"m_{{core}} = {mean.getVal():.3g} #pm {mean.getError():.3g}")
    pt.AddText(f"#sigma_{{core}} = {sigma.getVal():.3g} #pm {sigma.getError():.3g}")
    pt.AddText(f"#chi^2_{{core}} = {chi2:.3g}" )

    pt.Draw()

    c.Update()

    filename_h = f"SeedChannelHistoWithMask_{seed_channel}"
    subdir = f"Run_{Run}_Seed_{seed_channel}"
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    output_path_h = os.path.join(output_dir,subdir, filename_h)
    c.SaveAs(output_path_h + ".pdf")
    c.SaveAs(output_path_h + ".root")
    c.Clear()

    return {
        "mean": (mean.getVal(), mean.getError()),
        "sigma": (sigma.getVal(), sigma.getError())
    }


def main(arguments):

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i",  f"--input-dir", type=str, required=True, help="input directory containing ROOT file with unpacked tree")
    parser.add_argument("-ro", f"--plot-output-dir", type=str, required=True, help="directory for output plots")
    parser.add_argument("-j", f"--run-info-json", type=str, required=False, help="run and energy sample")

    args = parser.parse_args(arguments)

    json_dict = json.load(open(args.run_info_json, "r"))
    input_dir = args.input_dir
    plot_output_dir = args.plot_output_dir
    Run = json_dict["global"]["run info"]["run list"]
    SeedChannel = json_dict["global"]["run info"]["seed channel"]
    EtaCenter = json_dict["global"]["run info"]["eta center"]
    PhiCenter = json_dict["global"]["run info"]["phi center"]
    MaxX = json_dict["global"]["run info"]["maxx"]
    MinX = json_dict["global"]["run info"]["minx"]
    MaxY = json_dict["global"]["run info"]["maxy"]
    MinY = json_dict["global"]["run info"]["miny"]
    roofit_objects = []
    rows = []
    charge_dict = {}
    intercalib_dict = {}

    ROOT.gStyle.SetTitleSize(0.045, "XYZ")
    ROOT.gErrorIgnoreLevel = ROOT.kError

    for ie in range(len(Run)):

        c = ROOT.TCanvas()
        c.SetGrid()

        run = Run[ie]
        seed_channel = SeedChannel[ie]
        eta_center = EtaCenter[ie]
        phi_center = PhiCenter[ie]

        chain = ROOT.TChain("tree")

        pattern = os.path.join(input_dir, f"run_{run}/{run}_*_reco.root")

        for f in glob.glob(pattern):
            if has_branch(f, "ecal_charge_sum_5x5"):
                chain.Add(f)
            else:
                print("Skipping:", f)

        print(f"Run {run}: added {chain.GetNtrees()} files")

    ######                         Cetroid_eta vs hodox profile and fit

        centroid_par = 3.8

        h1 = ROOT.TH2F(f"ieta_centroid_vs_hodox_{run}", "", 30,-15,15,2000,50,57)
        chain.Draw(f"Sum$( (abs(ecal_iphi_within_5x5) < 2)*(abs(ecal_ieta_within_5x5) < 2)*max(0, {centroid_par} + log(max(ecal_charge_divided_5x5, 1e-8)))*ecal_ieta )/Sum$( (abs(ecal_iphi_within_5x5) < 2)*(abs(ecal_ieta_within_5x5) < 2)*max(0, {centroid_par} + log(max(ecal_charge_divided_5x5, 1e-8))) ):(hodo_x1_cl0_pos + hodo_x2_cl0_pos)/2>>ieta_centroid_vs_hodox_{run}", "hodo_x1_single_cl_flag && hodo_x2_single_cl_flag", "goff")

        h1.SetStats(0)
        h1.SetTitle("iEta_{centroid} vs Hodo_x;HodoX [mm];ieta_{{centroid}}")
        h1.SetMarkerStyle(24)
        h1.SetMarkerSize(0.8)
        h1.SetMarkerColor(ROOT.kBlack)
        ROOT.gStyle.SetOptTitle(1)
        ROOT.gStyle.SetTitleAlign(23)
        ROOT.gStyle.SetTitleX(0.5)
        h1.Draw("COLZ")

        hprof1 = h1.ProfileX()
        hprof1.Draw("same")

        fit1 = ROOT.TF1("fit", "pol1",MinX[ie],MaxX[ie])  #interval to verify the correct centroid parameter
        fit1.FixParameter(1,1/22)
        hprof1.Fit(fit1,"R")

        slope1 = fit1.GetParameter(1)
        #eslope1 = fit1.GetParError(1)
        const1 = fit1.GetParameter(0)
        econst1 = fit1.GetParError(0)
        chi2_1  = fit1.GetChisquare()
        ndf1 = fit1.GetNDF()
        prob1 = ROOT.TMath.Prob(chi2_1, ndf1)

        xcenter_hodo = (eta_center-const1)/slope1

        pave1 = ROOT.TPaveText(0.15, 0.7, 0.35, 0.88, "NDC")
        pave1.SetFillColor(0)
        pave1.SetTextFont(42)
        pave1.SetTextSize(0.03)
        pave1.SetBorderSize(1)

        pave1.AddText(f"Offset_eta = {const1:.4f} #pm {econst1:.4f}")
        pave1.AddText(f"Slope = {slope1:.3f} (fixed)")
        #pave1.AddText(f"Centroid par = {centroid_par:.2f}")
        pave1.AddText(f"#chi^2/Ndf = {chi2_1/ndf1:.2f}")
        pave1.AddText(f"HodoX_{{center}} = {xcenter_hodo:.3f} mm")
       # pave1.AddText(f"Ndof = {ndf1}")
    #            pave1.AddText(f"FitProb = {prob1}")
        pave1.Draw()
        fit1.Draw("same")

        filename_h1 = f"EtaCentroidvsHodoX_{run}"
        subdir = f"Run_{run}_Seed_{seed_channel}"
        os.makedirs(os.path.join(plot_output_dir, subdir), exist_ok=True)
        output_path_h1 = os.path.join(plot_output_dir,subdir, filename_h1)
        c.SaveAs(output_path_h1 + ".pdf")
        c.SaveAs(output_path_h1 + ".root")

        eta_min = eta_center - 4*abs(slope1)
        eta_max = eta_center + 4*abs(slope1)
        hodox_min = xcenter_hodo - 4
        hodox_max = xcenter_hodo + 4
        print("etamin/etamax/hodoxmin/hodoxmax",eta_min, eta_max,hodox_min,hodox_max)

    ######                             Cetroid_phi vs hodoy profile and fit

        c = ROOT.TCanvas()
        c.SetGrid()

        h2 = ROOT.TH2F(f"iphi_centroid_vs_hodoy_{run}","",30,-15,15,2500,2,8)
    #            chain.Draw(f"Sum$( (abs(ecal_iphi_within_5x5) < 2)*(abs(ecal_ieta_within_5x5) < 2)*({centroid_par} + log(ecal_charge_divided_5x5))*ecal_iphi )/Sum$( (abs(ecal_iphi_within_5x5) < 2)*(abs(ecal_ieta_within_5x5) < 2)*({centroid_par} + log(ecal_charge_divided_5x5)) ):(hodo_y1_cl0_pos + hodo_y2_cl0_pos)/2>>iphi_centroid_vs_hodoy_{run}", "hodo_y1_single_cl_flag && hodo_y2_single_cl_flag", "goff")
        chain.Draw(f"Sum$( (abs(ecal_iphi_within_5x5) < 2)*(abs(ecal_ieta_within_5x5) < 2)*max(0, {centroid_par} + log(max(ecal_charge_divided_5x5, 1e-8)))*ecal_iphi )/Sum$( (abs(ecal_iphi_within_5x5) < 2)*(abs(ecal_ieta_within_5x5) < 2)*max(0, {centroid_par} + log(max(ecal_charge_divided_5x5, 1e-8))) ):(hodo_y1_cl0_pos + hodo_y2_cl0_pos)/2>>iphi_centroid_vs_hodoy_{run}", "hodo_y1_single_cl_flag && hodo_y2_single_cl_flag", "goff")

        h2.SetStats(0)
        h2.SetTitle("Phi_{centroid} vs HodoY;HodoY [mm];iphi_{{centroid}}")
        h2.SetMarkerStyle(24)
        h2.SetMarkerSize(0.8)
        h2.SetMarkerColor(ROOT.kBlack)
        ROOT.gStyle.SetOptTitle(1)
        ROOT.gStyle.SetTitleAlign(23)
        ROOT.gStyle.SetTitleX(0.5)
        h2.Draw("COLZ")

        hprof2 = h2.ProfileX()
        hprof2.Draw("same")

        fit2 = ROOT.TF1("fit", "pol1",MinY[ie],MaxY[ie])
        fit2.FixParameter(1,-1/22)
        hprof2.Fit(fit2,"R")

        const2 = fit2.GetParameter(0)
        econst2 = fit2.GetParError(0)
        slope2 = fit2.GetParameter(1)
        chi2_2  = fit2.GetChisquare()
        ndf2 = fit2.GetNDF()
        prob2 = ROOT.TMath.Prob(chi2_2, ndf2)

        ycenter_hodo = (phi_center-const2)/slope2

        pave2 = ROOT.TPaveText(0.15, 0.7, 0.35, 0.88, "NDC")
        pave2.SetFillColor(0)
        pave2.SetTextFont(42)
        pave2.SetTextSize(0.03)
        pave2.SetBorderSize(1)

        pave2.AddText(f"Offset_phi = {const2:.4f} #pm {econst2:.4f}")
        pave2.AddText(f"Slope = {slope2:.3f} (fixed)")
        #pave2.AddText(f"Centroid par = {centroid_par:.2f}")
        pave2.AddText(f"#chi^2/Ndf = {chi2_2/ndf2:.2f}")
        pave2.AddText(f"HodoY_{{center}} = {ycenter_hodo:.3f} mm")
        #pave2.AddText(f"Ndof = {ndf2}")
    #            pave2.AddText(f"FitProb = {prob2}")

        pave2.Draw()
        fit2.Draw("same")

        filename_h2 = f"PhiCentroidvsHodoY_{run}"
        subdir = f"Run_{run}_Seed_{seed_channel}"
        os.makedirs(os.path.join(plot_output_dir, subdir), exist_ok=True)
        output_path_h2 = os.path.join(plot_output_dir,subdir, filename_h2)
        c.SaveAs(output_path_h2 + ".pdf")
        c.SaveAs(output_path_h2 + ".root")
        c.Clear()

        phi_min = phi_center - 4*abs(slope2)    #Warning:abs is present because slope2<0
        phi_max = phi_center + 4*abs(slope2)
        hodoy_min = ycenter_hodo - 4
        hodoy_max = ycenter_hodo + 4
        print("phimin/phimax/hodoymin/hodoymax",phi_min, phi_max,hodoy_min,hodoy_max)

    #######
    #
    #            c = ROOT.TCanvas()
    #            c.SetGrid()
    #
    #            h3 = ROOT.TH2F(f"seedchargevshodox_{run}","",100,-15,15,1000,0,20000)
    #            chain.Draw(f"ecal_charge_seed:(hodo_x1_cl0_pos + hodo_x2_cl0_pos)/2>>seedchargevshodox_{run}","hodo_x1_single_cl_flag && hodo_x2_single_cl_flag")
    #
    #            h3.SetStats(0)
    #            h3.SetTitle("Seed Charge vs HodoX;X [mm];Charge [ADC]")
    #            h3.SetMarkerStyle(24)
    #            h3.SetMarkerSize(0.8)
    #            h3.SetMarkerColor(ROOT.kBlack)
    #            ROOT.gStyle.SetOptTitle(1)
    #            ROOT.gStyle.SetTitleAlign(23)
    #            ROOT.gStyle.SetTitleX(0.5)
    #            h3.Draw("COLZ")
    #
    #            hprof3 = h3.ProfileX()
    #            hprof3.Draw("same")
    #
    #            filename_h3 = f"SeedChargevsHodoX_{run}"
    #            subdir = f"Run_{run}_Seed_{seed_channel}"
    #            os.makedirs(os.path.join(plot_output_dir, subdir), exist_ok=True)
    #            output_path_h3 = os.path.join(plot_output_dir,subdir, filename_h3)
    #            c.SaveAs(output_path_h3 + ".pdf")
    #            c.SaveAs(output_path_h3 + ".root")
    #            c.Clear()
    #
    #            c = ROOT.TCanvas()
    #            c.SetGrid()
    #
    #            h4 = ROOT.TH2F(f"seedchargevshodoy_{run}","",100,-20,20,1000,0,20000)
    #            chain.Draw(f"ecal_charge_seed:(hodo_y1_cl0_pos + hodo_y2_cl0_pos)/2>>seedchargevshodoy_{run}","hodo_y1_single_cl_flag && hodo_y2_single_cl_flag")
    #
    #            h4.SetStats(0)
    #            h4.SetTitle("Seed Charge vs HodoY;Y [mm];Charge [ADC]")
    #            h4.SetMarkerStyle(24)
    #            h4.SetMarkerSize(0.8)
    #            h4.SetMarkerColor(ROOT.kBlack)
    #            ROOT.gStyle.SetOptTitle(1)
    #            ROOT.gStyle.SetTitleAlign(23)
    #            ROOT.gStyle.SetTitleX(0.5)
    #            h4.Draw("COLZ")
    #
    #            hprof4 = h4.ProfileX()
    #            hprof4.Draw("same")
    #
    #            filename_h4 = f"SeedChargevsHodoY_{run}"
    #            subdir = f"Run_{run}_Seed_{seed_channel}"
    #            os.makedirs(os.path.join(plot_output_dir, subdir), exist_ok=True)
    #            output_path_h4 = os.path.join(plot_output_dir,subdir, filename_h4)
    #            c.SaveAs(output_path_h4 + ".pdf")
    #            c.SaveAs(output_path_h4 + ".root")
    #            c.Clear()

        ROOT.gStyle.SetOptFit(0)
        ROOT.gStyle.SetOptStat(0)

        c = ROOT.TCanvas(f"c_{run}", "", 800, 600)
        c.SetGrid()
        c_3d = ROOT.TCanvas(f"c_3d_{run}", "3D view", 800, 600)
        c_3d.SetGrid()

        hnew = ROOT.TProfile2D(f"ChargeSeedvshodoXY_{run}","SeedChannelvshodoXY;Hodo X [mm];Hodo Y [mm];Charge Seed [ADC]",30,-15,15,30,-15,15)
        chain.Draw(f"ecal_charge_seed:(hodo_y1_cl0_pos+hodo_y2_cl0_pos)/2:(hodo_x1_cl0_pos+hodo_x2_cl0_pos)/2>>ChargeSeedvshodoXY_{run}","hodo_x1_single_cl_flag && hodo_x2_single_cl_flag && hodo_y1_single_cl_flag && hodo_y2_single_cl_flag")

        h2d = ROOT.TH2D(f"h2d_{run}", "SeedChannevshodoXY;Hodo X [mm];Hodo Y [mm];Charge Seed",
                        30, -15, 15, 30, -15, 15)

        for bx in range(1, hnew.GetNbinsX() + 1):
            for by in range(1, hnew.GetNbinsY() + 1):
                entries = hnew.GetBinEntries(hnew.GetBin(bx, by))
                content = hnew.GetBinContent(bx, by)
                error   = hnew.GetBinError(bx, by)
                if entries < 5:
                    h2d.SetBinContent(bx, by, content)
                    h2d.SetBinError(bx, by, 1e6)   # too sparse: ignore
                else:
                    h2d.SetBinContent(bx, by, content)
                    h2d.SetBinError(bx, by, error if error > 0 else 1e6)

        bin_x = hnew.GetXaxis().FindBin(xcenter_hodo)
        bin_y = hnew.GetYaxis().FindBin(ycenter_hodo)
        x_center = hnew.GetXaxis().GetBinCenter(bin_x)
        y_center = hnew.GetYaxis().GetBinCenter(bin_y)
        z_value_crystalcenter = hnew.GetBinContent(bin_x, bin_y)

        fit_range = 3

        f2 = ROOT.TF2(f"gauss2d_{run}",
                      "[0] * TMath::Gaus(x, [1], [2]) * TMath::Gaus(y, [3], [4])",
                      xcenter_hodo - fit_range, xcenter_hodo + fit_range,
                      ycenter_hodo - fit_range, ycenter_hodo + fit_range)

        f2.SetParameter(0, z_value_crystalcenter)
        f2.SetParLimits(0, z_value_crystalcenter*0.5, z_value_crystalcenter*1.5)
        f2.SetParameter(1, xcenter_hodo)
        f2.SetParameter(2, 2)
        f2.SetParLimits(2, 0.5, 18.0)
#        f2.SetParLimits(2, 0.3, fit_range)
        f2.SetParameter(3, ycenter_hodo)
        f2.SetParameter(4, 1.5)
        f2.SetParLimits(4, 0.5, 23.0)
#        f2.SetParLimits(4, 0.3, fit_range)
        f2.SetLineColor(ROOT.kRed)
        f2.SetLineWidth(2)

        h2d.Fit(f2, "REN")

        p0 = f2.GetParameter(0)
        ep0 = f2.GetParError(0)
        p1 = f2.GetParameter(1)
        ep1 = f2.GetParError(1)
        p2 = f2.GetParameter(2)
        ep2 = f2.GetParError(2)
        p3 = f2.GetParameter(3)
        ep3 = f2.GetParError(3)
        p4 = f2.GetParameter(4)
        ep4 = f2.GetParError(4)

        x_peak = p1
        y_peak = p3
        chi2   = f2.GetChisquare()
        ndf    = f2.GetNDF()

        Gaussian_intercal=f2.Eval(xcenter_hodo, ycenter_hodo)

    ######

        c_3d.cd()                             #drawing 3d histo with fit on top
        c_3d.SetGrid()
        hnew.SetLineColor(ROOT.kBlack)
        hnew.SetLineWidth(1)
        hnew.Draw("LEGO2")

        f2.SetLineColor(ROOT.kRed)
        f2.SetLineWidth(2)
        f2.SetNpx(30)
        f2.SetNpy(30)
        f2.Draw("SURF1 SAME")

        c_3d.SetTheta(30)
        c_3d.SetPhi(230)
        c_3d.Update()
        filename_hnew3d = f"SeedChargevsHodo_3dfit_{run}"
        subdir = f"Run_{run}_Seed_{seed_channel}"
        os.makedirs(os.path.join(plot_output_dir, subdir), exist_ok=True)
        output_path_3d = os.path.join(plot_output_dir,subdir, filename_hnew3d)
        c_3d.SaveAs(output_path_3d + ".pdf")
        c_3d.SaveAs(output_path_3d + ".root")
        c_3d.Clear()

    #######

        c_px = ROOT.TCanvas(f"c_px_{run}", "ProfileX", 800, 600)    #projection of the fit on x
        c_px.SetGrid()

        bin_y_low  = hnew.GetYaxis().FindBin(ycenter_hodo - fit_range)
        bin_y_high = hnew.GetYaxis().FindBin(ycenter_hodo + fit_range)
        px = hnew.ProfileX(f"ProfileX_{run}", bin_y_low, bin_y_high)
        px.SetTitle("Profile X  [bins on the y axis in fit range];Hodo X [mm];Mean Charge Seed [ADC]")
        px.SetMarkerStyle(20)
        px.SetMarkerColor(ROOT.kBlue)
        px.Draw()

        f1x = ROOT.TF1(f"f1x_{run}", 
                       f"[0] * TMath::Gaus(x, [1], [2])",
                       xcenter_hodo - fit_range, xcenter_hodo + fit_range)

        sum_gy = 0.0
        n_ybins = 0
        for by in range(bin_y_low, bin_y_high + 1):
            yval = hnew.GetYaxis().GetBinCenter(by)
            sum_gy += math.exp(-0.5 * ((yval - p3) / p4) ** 2)
            n_ybins += 1
        avg_gy = sum_gy / n_ybins if n_ybins > 0 else 1.0
        f1x.SetParameter(0, p0*avg_gy)
        f1x.SetParameter(1, p1)
        f1x.SetParameter(2, p2)
        f1x.SetLineColor(ROOT.kRed)
        f1x.SetLineWidth(2)
        f1x.Draw("SAME")

        line2 = ROOT.TLine(xcenter_hodo, px.GetMinimum(), xcenter_hodo, px.GetMaximum())
        line2.SetLineColor(ROOT.kGreen+2)
        line2.SetLineWidth(2)
        line2.SetLineStyle(2)
        line2.Draw("SAME")

        leg_px = ROOT.TLegend(0.15, 0.15, 0.55, 0.35)
        leg_px.SetTextSize(0.030)
        leg_px.AddEntry(f1x, "xyGauss slice", "l")
        leg_px.AddEntry(line2, f"HodoX_{{center}} = {xcenter_hodo:.2f} mm", "l")
        leg_px.Draw()

        output_path_px = os.path.join(plot_output_dir, subdir, f"ProfileX_projection_{run}")
        c_px.SaveAs(output_path_px + ".pdf")
        c_px.SaveAs(output_path_px + ".root")

    #

        c_py = ROOT.TCanvas(f"c_py_{run}", "ProfileY", 800, 600)       #projection of the fit on y
        c_py.SetGrid()

        bin_x_low  = hnew.GetXaxis().FindBin(xcenter_hodo - fit_range)
        bin_x_high = hnew.GetXaxis().FindBin(xcenter_hodo + fit_range)
        py = hnew.ProfileY(f"ProfileY_{run}", bin_x_low, bin_x_high)
        py.SetTitle("Profile Y [bins on x axis in fit range];Hodo Y [mm];Mean Charge Seed [ADC]")
        py.SetMarkerStyle(20)
        py.SetMarkerColor(ROOT.kBlue)
        py.Draw()

        f1y = ROOT.TF1(f"f1y_{run}",
                       f"[0] * TMath::Gaus(x, [1], [2])",
                       ycenter_hodo - fit_range, ycenter_hodo + fit_range)

        sum_gx = 0.0
        n_xbins = 0
        for bx in range(bin_x_low, bin_x_high + 1):
            xval = hnew.GetXaxis().GetBinCenter(bx)
            sum_gx += math.exp(-0.5 * ((xval - p1) / p2) ** 2)
            n_xbins += 1
        avg_gx = sum_gx / n_xbins if n_xbins > 0 else 1.0
        f1y.SetParameter(0, p0*avg_gx)
        f1y.SetParameter(1, p3)
        f1y.SetParameter(2, p4)
        f1y.SetLineColor(ROOT.kRed)
        f1y.SetLineWidth(2)
        f1y.Draw("SAME")

        line = ROOT.TLine(ycenter_hodo, py.GetMinimum(), ycenter_hodo, py.GetMaximum())
        line.SetLineColor(ROOT.kGreen+2)
        line.SetLineWidth(2)
        line.SetLineStyle(2)
        line.Draw("SAME")

        leg_py = ROOT.TLegend(0.15, 0.15, 0.55, 0.35)
        leg_py.SetTextSize(0.030)
        leg_py.AddEntry(f1y, "xyGauss slice", "l")
        leg_py.AddEntry(line, f"HodoY_{{center}} = {ycenter_hodo:.2f} mm", "l")
        leg_py.Draw()

        output_path_py = os.path.join(plot_output_dir, subdir, f"ProfileY_projection_{run}")
        c_py.SaveAs(output_path_py + ".pdf")
        c_py.SaveAs(output_path_py + ".root")

    #######

        c.cd()                          #drawing standard 2DProfile
        hnew.SetLineColor(ROOT.kBlack)
        hnew.SetLineWidth(1)
        hnew.GetZaxis().SetRangeUser(0, 12000)
        hnew.Draw("COLZ")
        marker = ROOT.TMarker(x_center, y_center, 29)     #marker red star, nominal center drom nominal eta/phi values
        marker.SetMarkerColor(ROOT.kRed)
        marker.SetMarkerSize(2.5)
        marker.Draw("SAME")
#        marker_fit = ROOT.TMarker(x_peak, y_peak, 29)       #marker black star, coordinates of the max from the fit
#        marker_fit.SetMarkerColor(ROOT.kBlack)
#        marker_fit.SetMarkerSize(2.5)
#        marker_fit.Draw("SAME")

        h_contour = ROOT.TH2D(f"h_contour_{run}", "",
                              100, -15, 15,
                              100, -15, 15)
        h_contour.Add(f2)
        levels = array.array('d', [0.3*p0, 0.5*p0, 0.7*p0, 0.85*p0, 0.95*p0])
        h_contour.SetContour(len(levels), levels)
        h_contour.SetLineColor(ROOT.kRed)
        h_contour.SetLineWidth(2)
        h_contour.Draw("CONT3 SAME")

        leg = ROOT.TLegend(0.15, 0.60, 0.50, 0.88)
        leg.SetBorderSize(1)
        leg.SetFillColor(0)
        leg.SetTextSize(0.028)
        leg.AddEntry(marker, "nominal crystal center", "P")
        leg.AddEntry(ROOT.nullptr, f"Nominal center hodox = {xcenter_hodo:.2f} mm","")
        leg.AddEntry(ROOT.nullptr, f"Nominal center hodoy = {ycenter_hodo:.2f} mm","")
        #leg.AddEntry(marker_fit, "max charge position from paraboloid fit", "P")
        leg.AddEntry(f2, "xyGauss fit", "l")
        leg.AddEntry(ROOT.nullptr, f"Amp  = {p0:.0f} #pm {ep0:.0f} ADC","")
        leg.AddEntry(ROOT.nullptr, f"#mu_{{x}} = {p1:.2f} #pm {ep1:.2f} mm","")
        leg.AddEntry(ROOT.nullptr, f"#sigma_{{x}} = {p2:.1f} #pm {ep2:.1f} mm","")
        leg.AddEntry(ROOT.nullptr, f"#mu_{{y}} = {p3:.2f} #pm {ep3:.2f} mm","")
        leg.AddEntry(ROOT.nullptr, f"#sigma_{{y}} = {p4:.1f} #pm {ep4:.1f} mm","")
        leg.AddEntry(ROOT.nullptr, f"#chi^{{2}} / ndf = {chi2:.2f} / {ndf}","")
        leg.Draw()

        filename_hnew = f"SeedChargevsHodo_{run}"
        subdir = f"Run_{run}_Seed_{seed_channel}"
        os.makedirs(os.path.join(plot_output_dir, subdir), exist_ok=True)
        output_path_hnew = os.path.join(plot_output_dir,subdir, filename_hnew)
        c.SaveAs(output_path_hnew + ".pdf")
        c.SaveAs(output_path_hnew + ".root")
        c.Clear()


#    #######                                   other way to compute the intercalibration constants, not in use now
#        c = ROOT.TCanvas()
#        c.SetGrid()
#
#        h = ROOT.TH1F(f"SeedChannelHistoWithMask_{seed_channel}", "", 500,0,20000)
#        chain.Draw(f"ecal_charge_seed>>SeedChannelHistoWithMask_{seed_channel}", f"ecal_ieta > {eta_min} && ecal_ieta < {eta_max} && ecal_iphi > {phi_min} && ecal_iphi < {phi_max} && (hodo_x1_cl0_pos + hodo_x2_cl0_pos)/2 > {hodox_min} && (hodo_x1_cl0_pos + hodo_x2_cl0_pos)/2 < {hodox_max} && (hodo_y1_cl0_pos + hodo_y2_cl0_pos)/2 > {hodoy_min} && (hodo_y1_cl0_pos + hodo_y2_cl0_pos)/2 < {hodoy_max} && hodo_y1_single_cl_flag && hodo_y2_single_cl_flag && hodo_x1_single_cl_flag && hodo_x2_single_cl_flag")
#        h.SetStats(0)
#        h.SetTitle("Seed channel charge;Charge [ADC];Events")
#        h.SetMarkerStyle(24)
#        h.SetMarkerSize(0.8)
#        h.SetMarkerColor(ROOT.kBlack)
#        ROOT.gStyle.SetOptTitle(1)
#        ROOT.gStyle.SetTitleAlign(23)
#        ROOT.gStyle.SetTitleX(0.5)
#        #h.Draw("COLZ")
#
#        max_bin = h.GetMaximumBin()
#        max_position = h.GetBinCenter(max_bin)
#        max_value = h.GetBinContent(max_bin)
#        bin1 = h.FindFirstBinAbove(max_value/2)
#        bin2 = h.FindLastBinAbove(max_value/2)
#        fwhm = h.GetBinCenter(bin2) - h.GetBinCenter(bin1)
#
#        fit_min = max_position - 2.5*fwhm
#        fit_max = max_position + 1.5*fwhm
#
#        results = cbFit(h,h.GetName(),run,plot_output_dir,seed_channel,fit_min,fit_max)
#
#        roofit_objects.append(results)
#
#        mu_val, emu_val = results["mean"]
#        sig_val, esig_val = results["sigma"]

        print("Seed/Fit value xyGauss:",seed_channel,Gaussian_intercal)

        rows.append({"seed_channel": seed_channel,
                "x_center_hodo": xcenter_hodo,
                "y_center_hodo": ycenter_hodo,
                "a_eta": const1,
                "b_eta": slope1,
                "a_phi": const2,
                "b_phi": slope2,
                "ChargeSeedCenterCrystal": z_value_crystalcenter,
                "GaussianFitValue": Gaussian_intercal})
    #            writer.writerow([seed_channel,f"{float(xcenter_hodo):.5f}",f"{float(ycenter_hodo):.5f}",f"{float(const1):.5f}",f"{float(slope1):.5f}",f"{float(const2):.5f}",f"{float(slope2):.5f}",f"{int(z_value_crystalcenter)}",f"{float(Gaussian_intercal):.10f}"])


    Gaussian_intercal_seed = rows[4]["GaussianFitValue"]

    with open("intercalibration_info.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed_channel", "x_center_hodo", "y_center_hodo",
                         "a_eta", "b_eta", "a_phi", "b_phi","ChargeSeedHistoCenter",
                         "GaussianFitValueCenter", "IntercalibrationFactor"])
        for row in rows:
            intercal_factor = row["GaussianFitValue"] / Gaussian_intercal_seed
            writer.writerow([
                row["seed_channel"],
                f"{float(row['x_center_hodo']):.5f}",
                f"{float(row['y_center_hodo']):.5f}",
                f"{float(row['a_eta']):.5f}",
                f"{float(row['b_eta']):.5f}",
                f"{float(row['a_phi']):.5f}",
                f"{float(row['b_phi']):.5f}",
                f"{int(row['ChargeSeedCenterCrystal'])}",
                f"{float(row['GaussianFitValue']):.10f}",
                f"{float(intercal_factor):.5f}"
            ])


    input("finito")
if __name__ == "__main__":
    main(sys.argv[1:])
