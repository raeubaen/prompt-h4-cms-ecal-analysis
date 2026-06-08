#!/usr/bin/env python3
import os, json, uproot, argparse, sys, ROOT
import numpy as np
import array
import glob
from math import sqrt
import csv
import time

def has_branch(fname, branch):
    try:
      t_start = time.time()
      uproot.concatenate(fname)
    except:
      return False
    t_start = time.time()
    f = ROOT.TFile.Open(fname)
    if not f or f.IsZombie():
        return False
    t = f.Get("tree")
    if not t:
        return False
    return t.GetBranchStatus(branch)

def cbFit(h, name, Run, energy, output_dir, xmin=-1, xmax=-1):

    x = ROOT.RooRealVar(f"x_{name}_{energy}", "Energy [ADC]", h.GetXaxis().GetXmin(), h.GetXaxis().GetXmax())

    data = ROOT.RooDataHist(f"data_{name}_{energy}", "data", ROOT.RooArgList(x), h)




    h.GetXaxis().SetRangeUser(500,8000)
    h.GetXaxis().SetRangeUser(h.GetMean()-3*h.GetRMS(),h.GetMean()+3*h.GetRMS())
    h.GetXaxis().SetRangeUser(h.GetMean()-3*h.GetRMS(),h.GetMean()+3*h.GetRMS())
    h.GetXaxis().SetRangeUser(h.GetMean()-3*h.GetRMS(),h.GetMean()+3*h.GetRMS())
    peak = h.GetMean()


    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Peak center is {peak}")
    print(f"//////////////////////////////////////////////")
    print(f"//////////////////////////////////////////////")
    print(f"//////////////////////////////////////////////")

    mean  = ROOT.RooRealVar(f"mean_{name}", "DCB mean", peak, peak-501, peak+500)
    sigma = ROOT.RooRealVar(f"sigma_{name}", "DCB sigma", h.GetRMS(), 1, 200)

    alphaL = ROOT.RooRealVar(f"alphaL_{name}", "alphaL", 1.5, 0.1, 5.0)
    nL     = ROOT.RooRealVar(f"nL_{name}",     "nL",     3.0, 0.5, 20.0)

    alphaR = ROOT.RooRealVar(f"alphaR_{name}", "alphaR", 1.5, 0.1, 5.0)
    nR     = ROOT.RooRealVar(f"nR_{name}",     "nR",     3.0, 0.5, 20.0)

    dcb = ROOT.RooCrystalBall(f"dcb_{name}", "Double Crystal Ball", x, mean, sigma, alphaL, nL, alphaR, nR)

    nsig = ROOT.RooRealVar(f"nsig_{name}", "signal yield", h.Integral(), 0.0, 10.0*h.Integral())
    model = ROOT.RooAddPdf(f"model_{name}_{energy}", "extended DCB model", ROOT.RooArgList(dcb), ROOT.RooArgList(nsig))

    fitArgs = [
        ROOT.RooFit.Extended(True),
        ROOT.RooFit.Save(),
        ROOT.RooFit.PrintLevel(-1)
    ]

    if xmin >= 0 and xmax >= 0:
        fitArgs.insert(0, ROOT.RooFit.Range("fitRange"))
        x.setRange("fitRange", xmin, xmax)

    result = model.fitTo(data, *fitArgs)

    canvas = ROOT.TCanvas()

    frame = x.frame()
    data.plotOn(frame,ROOT.RooFit.Range("fitRange"))
    model.plotOn(frame, ROOT.RooFit.Range("fitRange"))

    ROOT.gStyle.SetOptTitle(1)
    frame.SetTitle(f"Energy {energy} - dcb fit")
    frame.Draw()

    npar = result.floatParsFinal().getSize()
    chi2_ndf = frame.chiSquare(npar)

    pt = ROOT.TPaveText(0.60, 0.65, 0.88, 0.88, "NDC")
    pt.SetFillColor(0)
    pt.SetTextFont(42)
    pt.SetBorderSize(0)
    pt.SetTextSize(0.05)

    pt.AddText(f"#mu = {mean.getVal():.5g} #pm {mean.getError():.1g}")
    pt.AddText(f"#sigma = {sigma.getVal():.4g} #pm {sigma.getError():.1g}")
    pt.AddText(f"Resolution = {sigma.getVal()/mean.getVal():.2g}" )
    pt.AddText(f"#chi^2/Ndf = {chi2_ndf:.3g}" )
    pt.AddText(f"Energy = {energy:.3g} GeV" )

    pt.Draw()

    canvas.Update()

    filename = f"{name}_fit_dcb"
    output_path = os.path.join(output_dir, filename)
    canvas.SaveAs(output_path + ".pdf")
    canvas.SaveAs(output_path + ".root")

    return {
        "mean": (mean.getVal(), mean.getError()),
        "sigma": (sigma.getVal(), sigma.getError())
    }


def main(arguments):

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i",  f"--input-dir", type=str, default = "/eos/cms/store/group/dpg_ecal/comm_ecal/upgrade/testbeam/ECALTB_H4_Jun2026/reco_dqm/", help="input directory containing ROOT file with unpacked tree")
    parser.add_argument("-ro", f"--plot-output-dir", type=str, default = "/eos/user/m/mcampana/www/h4dqm/prompt_2026_fit", help="directory for output plots")
    parser.add_argument("-f", f"--fit-output-dir", type=str, default= "/eos/user/m/mcampana/www/h4dqm/prompt_2026_fit" , help="directory for fits")
    parser.add_argument("-j", f"--run-info-json", type=str, default= "run_info_with_cuts.json", help="run and energy sample")

    args = parser.parse_args(arguments)

    json_dict = json.load(open(args.run_info_json, "r"))
    input_dir=args.input_dir
    plot_output_dir=args.plot_output_dir
    fit_output_dir=args.fit_output_dir
    os.makedirs(plot_output_dir, exist_ok=True)
    os.makedirs(fit_output_dir, exist_ok=True)

    dd = json_dict["global"]["run info"]

    cut = json_dict["cuts"]

    Run_list, Ebins, do_fitamp, do_channel_matrix_3x3 = [dd[k] for k in ["run list", "run energies", "do fitamp", "do matrix 3x3"]]
    rows_resolution = []

    lin = ROOT.TGraphErrors(len(Ebins))
    res = ROOT.TGraphErrors(len(Ebins))

    ROOT.gStyle.SetTitleSize(0.045, "XYZ")

    for ie in range(len(Ebins)):

        c = ROOT.TCanvas()
        c.SetGrid()

        run_list = Run_list[ie]
        energy = Ebins[ie]
        
        chain = ROOT.TChain("tree")
        
        for run in run_list:
            pattern = os.path.join(input_dir, f"run_{run}/{run}_*_reco.root")

            for f in glob.glob(pattern):
                print("testing: ", f)
                if has_branch(f, "ecal_charge_sum_5x5"):
                    chain.Add(f)
                else:
                    print("Skipping:", f)

        print(f"Runs {run_list}: added {chain.GetNtrees()} files")

        if do_fitamp:
            if do_channel_matrix_3x3:
                h = ROOT.TH1F(f"FitAmp_3x3_{energy}_uncalibrated", "", 8000, 0, 8000)
                FitAmp_sum_3x3_string = "Sum$(ecal_lsfit_amp* (abs(ecal_iphi_within_5x5) < 2) * (abs(ecal_ieta_within_5x5) < 2))"
                print("drawing", f"{FitAmp_sum_3x3_string}>>FitAmp_3x3_{energy}_uncalibrated", cut[ie], "goff")
                chain.Draw(f"{FitAmp_sum_3x3_string}>>FitAmp_3x3_{energy}_uncalibrated", cut[ie], "goff")
            else:
                h = ROOT.TH1F(f"FitAmp_5x5_{energy}_uncalibrated", "", 8000, 0, 8000)
                FitAmp_sum_5x5_string = "Sum$(ecal_lsfit_amp * (abs(ecal_iphi_within_5x5) < 3) * (abs(ecal_ieta_within_5x5) < 3))"
                print("drawing", f"{FitAmp_sum_5x5_string}>>FitAmp_5x5_{energy}_uncalibrated", cut[ie], "goff")
                chain.Draw(f"{FitAmp_sum_5x5_string}>>FitAmp_5x5_{energy}_uncalibrated", cut[ie], "goff")
        else:
            if do_channel_matrix_3x3:
                h = ROOT.TH1F(f"Charge_3x3_{energy}_uncalibrated", "", 8000, 0, 8000)
                Charge_sum_3x3_string = "Sum$(ecal_charge * (abs(ecal_iphi_within_5x5) < 2) * (abs(ecal_ieta_within_5x5) < 2))"
                print("drawing", f"{Charge_sum_3x3_string}>>Charge_3x3_{energy}_uncalibrated", cut[ie], "goff")
                chain.Draw(f"{Charge_sum_3x3_string}>>Charge_3x3_{energy}_uncalibrated", cut[ie], "goff")
            else:
                h = ROOT.TH1F(f"Charge_5x5_{energy}_uncalibrated", "", 8000, 0, 8000)
                Charge_sum_5x5_string = "Sum$(ecal_charge * (abs(ecal_iphi_within_5x5) < 3) * (abs(ecal_ieta_within_5x5) < 3))"
                print("drawing", f"{Charge_sum_5x5_string}>>Charge_5x5_{energy}_uncalibrated", cut[ie], "goff")
                chain.Draw(f"{Charge_sum_5x5_string}>>Charge_5x5_{energy}_uncalibrated", cut[ie], "goff")

        print("chain draw done")
        h.Draw()

#####   uncalibrated histo
        h.GetXaxis().SetRangeUser(500,8000)
        h.GetXaxis().SetRangeUser(h.GetMean()-3*h.GetRMS(),h.GetMean()+3*h.GetRMS())
        h.GetXaxis().SetRangeUser(h.GetMean()-3*h.GetRMS(),h.GetMean()+3*h.GetRMS())
        h.GetXaxis().SetRangeUser(h.GetMean()-3*h.GetRMS(),h.GetMean()+3*h.GetRMS())


        x_min = h.GetMean() - 4*h.GetRMS()
        x_max = h.GetMean() + 4*h.GetRMS()

        print(f"Min before functinon {x_min}")
        print(f"Man before functinon {x_max}")

        results = cbFit(h, h.GetName(), run_list, energy, fit_output_dir, x_min, x_max)
        mu_val, emu_val = results["mean"]
        sig_val, esig_val = results["sigma"]

#####   filling the plots

        resolution_error = sqrt((esig_val/mu_val)**2+emu_val**2*(sig_val/mu_val**2)**2+(5e-4)**2)

        lin.SetPoint(ie, mu_val, energy)                  #energy linearity
        lin.SetPointError(ie, emu_val, energy*0.025)

        res.SetPoint(ie, energy, 100*(sig_val/mu_val))      #resolution vs beam energy
        res.SetPointError(ie, 0, 100*resolution_error)

        rows_resolution.append({
            "run": run,
            "energy": energy,
            "resolution_uncalib": 100*(sig_val/mu_val),
            "resolution_uncalib_err": 100*resolution_error,
        })

#####

    #saving data
    with open("resolution_points.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "energy", "resolution_uncalib", "resolution_uncalib_err"])
        for row in rows_resolution:
            writer.writerow([
                row["energy"],
                f"{row['resolution_uncalib']:.5f}",
                f"{row['resolution_uncalib_err']:.5f}",
            ])

    canvas = ROOT.TCanvas()
    canvas.SetGrid()

    lin.SetMarkerStyle(24)
    lin.SetMarkerSize(0.8)
    lin.SetMarkerColor(ROOT.kBlack)
    res.SetMarkerStyle(22)
    res.SetMarkerSize(1.4)
    res.SetMarkerColor(ROOT.kBlue)
    
    ROOT.gStyle.SetOptTitle(1)
    ROOT.gStyle.SetTitleAlign(23)
    ROOT.gStyle.SetTitleX(0.5)

#####   saving linearity plot

    if do_fitamp:
        if do_channel_matrix_3x3:
            lin.SetTitle(f"Energy linearity ;#Mu_fitAmp_3x3 [ADC];Beam energy [GeV]")
        else:
            lin.SetTitle(f"Energy linearity ;#Mu_fitAmp_5x5 [ADC];Beam energy [GeV]")
    else:
        if do_channel_matrix_3x3:
            lin.SetTitle(f"Energy linearity ;#Mu_charge_3x3 [ADC];Beam energy [GeV]")
        else:
            lin.SetTitle(f"Energy linearity ;#Mu_charge_5x5 [ADC];Beam energy [GeV]")

    lin.Draw("AP")
    canvas.Update()

    filename_lin = f"Energy_linearity"
    output_path_lin = os.path.join(plot_output_dir, filename_lin)
    canvas.SaveAs(output_path_lin + ".pdf")
    canvas.SaveAs(output_path_lin + ".root")
    canvas.Clear()

#####   saving resolution plot

    if do_fitamp:
        if do_channel_matrix_3x3:
            res.SetTitle(f"Resolution 3x3 ;Beam energy [GeV];(#sigma/#mu)_{{fitAmp_3x3}} %")
        else:
            res.SetTitle(f"Resolution 5x5 ;Beam energy [GeV];(#sigma/#mu)_{{fitAmp_5x5}} %")
    else:
        if do_channel_matrix_3x3:
            res.SetTitle(f"Resolution 3x3 ;Beam energy [GeV];(#sigma/#mu)_{{charge_3x3}} %")
        else:
            res.SetTitle(f"Resolution 5x5 ;Beam energy [GeV];(#sigma/#mu)_{{charge_5x5}} %")

    ROOT.gStyle.SetOptFit(0)
    ROOT.gStyle.SetOptStat(0)
    res.Draw("AP")

    fit = ROOT.TF1("fit", "sqrt(([0]/x)**2 + ([1]/sqrt(x))**2 + [2]**2 )", 0, 250)
    fit.SetLineColor(ROOT.kRed)
    fit_result = res.Fit(fit, "RS")

    p0    = fit.GetParameter(0)
    p0e   = fit.GetParError(0)
    p1    = fit.GetParameter(1)
    p1e   = fit.GetParError(1)
    p2    = fit.GetParameter(2)
    p2e   = fit.GetParError(2)
    chi2  = fit.GetChisquare()
    ndf   = fit.GetNDF()

    leg = ROOT.TLegend(0.15, 0.60, 0.60, 0.88)
    leg.SetBorderSize(1)
    leg.SetFillColor(0)
    leg.SetTextSize(0.032)
    if do_channel_matrix_3x3:
        leg.AddEntry(res, "Ecal 3x3", "lep")
    else:
        leg.AddEntry(res, "Ecal 5x5", "lep")
    leg.AddEntry(fit, "#sqrt{(N/E)^{2} + (S/#sqrt{E})^{2} + C^{2}}", "l")
    leg.AddEntry(ROOT.nullptr, f"N = ({p0/100:.4f} #pm {p0e/100:.4f}) GeV", "")
    leg.AddEntry(ROOT.nullptr, f"S = ({p1/100:.6f} #pm {p1e/100:.6f}) GeV^{{1/2}}", "")
    leg.AddEntry(ROOT.nullptr, f"C = {p2/100:.8f} #pm {p2e/100:.8f} ", "")
    leg.AddEntry(ROOT.nullptr, f"#chi^{{2}} / ndf = {chi2:.2f} / {ndf}", "")
    leg.Draw()

    status = fit_result.Status()
    covstatus = fit_result.CovMatrixStatus()
    print(f"Fit status: {status} (0=converged)")
    print(f"Covariance matrix status: {covstatus} (3=accurate)")

    if status != 0:
        print("WARNING: fit did not converge!")
    if covstatus != 3:
        print("WARNING: covariance matrix not accurate!")

    filename_res = f"Resolution_uncalib"
    output_path_res = os.path.join(plot_output_dir, filename_res)
    for ext in [".pdf",".png",".C",".root"]:
        canvas.SaveAs(output_path_res + ext)
    canvas.Clear()

    input("finito")

if __name__ == "__main__":
    main(sys.argv[1:])
