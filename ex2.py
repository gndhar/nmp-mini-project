import numpy as np
import ROOT
import multiprocessing
from numba import njit

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetTitleFontSize(0.06)

N0 = 5_000_000
N_STEPS = 500

CASES = {
    "secular": (0.0001, 0.10),
    "transient": (0.0100, 0.10),
    "noequil": (0.1000, 0.01),
}


@njit(fastmath=True, cache=True)
def simulate(pP, pQ, n0=N0, n_steps=N_STEPS, seed=0):
    NP = n0
    NQ = 0

    times = np.arange(n_steps + 1, dtype=np.float64)
    actP = np.zeros(n_steps + 1, dtype=np.float64)
    actQ = np.zeros(n_steps + 1, dtype=np.float64)

    np.random.seed(seed)
    actP[0] = pP * NP

    for t in range(1, n_steps + 1):
        decayP = np.random.binomial(NP, pP)
        decayQ = np.random.binomial(NQ, pQ)

        NP -= decayP
        NQ += decayP - decayQ

        actP[t] = pP * NP
        actQ[t] = pQ * NQ

    return times, actP, actQ


@njit(fastmath=True, cache=True)
def analytic_actP(t, lP, M0P):
    return M0P * np.exp(-lP * t)


@njit(fastmath=True, cache=True)
def analytic_actQ(t, lP, lQ, M0P):
    out = np.empty(len(t), dtype=np.float64)
    if abs(lQ - lP) < 1e-12:
        for i in range(len(t)):
            out[i] = M0P * lQ * t[i] * np.exp(-lP * t[i])
    else:
        k = lQ / (lQ - lP)
        for i in range(len(t)):
            out[i] = k * M0P * (np.exp(-lP * t[i]) - np.exp(-lQ * t[i]))
    return out


def _run_case(args):
    name, pP, pQ, seed = args
    times, actP, actQ = simulate(pP, pQ, seed=seed)
    return name, {"t": times, "actP": actP, "actQ": actQ, "pP": pP, "pQ": pQ}


def _warmup():
    simulate(0.01, 0.1, n0=10, n_steps=5, seed=0)
    t = np.linspace(0, 10, 5, dtype=np.float64)
    analytic_actP(t, 0.01, 1.0)
    analytic_actQ(t, 0.01, 0.1, 1.0)


def make_figure(data_dict, outfile="fig_radioactive_equilibrium.png"):
    c = ROOT.TCanvas("c", "Radioactive Decay Series P -> Q -> R", 1300, 1000)
    c.Divide(2, 2)
    refs = []

    t_dense = np.linspace(0, 500, 2000, dtype=np.float64)

    c.cd(1)
    ROOT.gPad.SetLogy(True)

    mg_a = ROOT.TMultiGraph()
    mg_a.SetTitle("(a) Radioactivity - different decay probabilities;Time;Activity (P)")
    leg_a = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kBlack]
    labels = ["p_{P} = 0.0001", "p_{P} = 0.01", "p_{P} = 0.1"]

    for key, lbl, col in zip(["secular", "transient", "noequil"], labels, colors):
        t = data_dict[key]["t"]
        y = np.maximum(data_dict[key]["actP"], 1e-1)
        gr = ROOT.TGraph(len(t), t, y)
        gr.SetMarkerStyle(ROOT.kFullCircle)
        gr.SetMarkerSize(0.6)
        gr.SetMarkerColor(col)
        mg_a.Add(gr, "P")
        leg_a.AddEntry(gr, lbl, "p")
        refs.append(gr)

    mg_a.Draw("A")
    mg_a.GetXaxis().SetLimits(0, 500)
    mg_a.SetMinimum(1e-1)
    leg_a.Draw()
    refs.extend([mg_a, leg_a])

    def draw_equilibrium_panel(pad_idx, key, title, show_tmax=False):
        c.cd(pad_idx)
        ROOT.gPad.SetLogy(True)
        d = data_dict[key]

        mg = ROOT.TMultiGraph()
        mg.SetTitle(title)
        t = d["t"]
        lP, lQ = d["pP"], d["pQ"]
        M0P = d["actP"][0]

        actP = np.maximum(d["actP"], 1.0)
        actQ = np.maximum(d["actQ"], 1.0)

        gr_P = ROOT.TGraph(len(t), t, actP)
        gr_P.SetMarkerStyle(ROOT.kFullCircle)
        gr_P.SetMarkerSize(0.6)
        gr_P.SetMarkerColor(ROOT.kBlue)

        gr_Q = ROOT.TGraph(len(t), t, actQ)
        gr_Q.SetMarkerStyle(ROOT.kFullCircle)
        gr_Q.SetMarkerSize(0.6)
        gr_Q.SetMarkerColor(ROOT.kRed)

        yP_ana = analytic_actP(t_dense, lP, M0P)
        yQ_ana = np.maximum(analytic_actQ(t_dense, lP, lQ, M0P), 1.0)

        gr_P_ana = ROOT.TGraph(len(t_dense), t_dense, yP_ana)
        gr_P_ana.SetLineColor(ROOT.kBlue)
        gr_P_ana.SetLineWidth(2)

        gr_Q_ana = ROOT.TGraph(len(t_dense), t_dense, yQ_ana)
        gr_Q_ana.SetLineColor(ROOT.kRed)
        gr_Q_ana.SetLineWidth(2)

        mg.Add(gr_P_ana, "L")
        mg.Add(gr_Q_ana, "L")
        mg.Add(gr_P, "P")
        mg.Add(gr_Q, "P")
        mg.Draw("A")
        mg.GetXaxis().SetLimits(0, 500)
        mg.SetMinimum(1.0)

        leg = ROOT.TLegend(0.65, 0.75, 0.9, 0.9)
        leg.AddEntry(gr_P, "Activity of P (Sim)", "p")
        leg.AddEntry(gr_Q, "Activity of Q (Sim)", "p")
        leg.AddEntry(gr_P_ana, "Theory P", "l")
        leg.AddEntry(gr_Q_ana, "Theory Q", "l")

        if show_tmax:
            t_max = np.log(lQ / lP) / (lQ - lP)
            line = ROOT.TLine(
                t_max, mg.GetYaxis().GetXmin(), t_max, mg.GetYaxis().GetXmax()
            )
            line.SetLineColor(ROOT.kBlack)
            line.SetLineStyle(2)
            line.Draw()
            leg.AddEntry(line, f"t_{{max}} #approx {t_max:.1f}", "l")
            refs.append(line)

        leg.Draw()
        refs.extend([mg, gr_P, gr_Q, gr_P_ana, gr_Q_ana, leg])

    draw_equilibrium_panel(
        2, "secular", "(b) Secular equilibrium (p_{P}=0.0001, p_{Q}=0.1);Time;Activity"
    )
    draw_equilibrium_panel(
        3,
        "transient",
        "(c) Transient equilibrium (p_{P}=0.01, p_{Q}=0.1);Time;Activity",
    )
    draw_equilibrium_panel(
        4,
        "noequil",
        "(d) No equilibrium (p_{P}=0.1, p_{Q}=0.01);Time;Activity",
        show_tmax=True,
    )

    c.SaveAs(outfile)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    _warmup()

    args = [(name, pP, pQ, i) for i, (name, (pP, pQ)) in enumerate(CASES.items())]

    with multiprocessing.Pool(processes=len(CASES)) as pool:
        results = pool.map(_run_case, args)

    sim_data = {name: d for name, d in results}
    make_figure(sim_data)
