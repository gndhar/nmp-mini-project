import numpy as np
import tqdm
import ROOT
import multiprocessing
from numba import njit

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

M_PI, SIG_PI = 139.57018, 0.00035
M_MU = 105.65837
M_E = 0.51100
EXP_MOM_MEAN = 60.0

CHUNK_SIZE = 2_000_000
N_CHUNKS = 250
N_WORKERS = max(1, multiprocessing.cpu_count() - 1)

MOM_BINS = (0.0, 500.0, 100)
ANG_BINS = (0.0, 180.0, 90)
PROF_BINS = (0.0, 500.0, 25)


@njit(fastmath=True, cache=True)
def fast_binning(data, bin_params):
    low, high, n_bins = bin_params
    inv_width = n_bins / (high - low)
    counts = np.zeros(n_bins, dtype=np.float64)
    for i in range(len(data)):
        val = data[i]
        if low <= val < high:
            idx = int((val - low) * inv_width)
            counts[idx] += 1
    return counts


@njit(fastmath=True, cache=True)
def run_physics_chunk(seed):
    np.random.seed(seed)
    n = CHUNK_SIZE

    pi_mass = np.random.normal(M_PI, SIG_PI, n)
    pi_mom = np.random.exponential(EXP_MOM_MEAN, n)

    E_pi = np.sqrt(pi_mass**2 + pi_mom**2)
    gamma = E_pi / pi_mass
    bg = pi_mom / pi_mass

    cos_cm = np.random.uniform(-1.0, 1.0, n)
    sin_cm = np.sqrt(1.0 - cos_cm**2)
    theta_cm_deg = np.arccos(cos_cm) * (180.0 / np.pi)

    res_mu = [np.zeros(1, dtype=np.float64) for _ in range(10)]
    res_e = [np.zeros(1, dtype=np.float64) for _ in range(10)]

    for mB, res in [(M_MU, res_mu), (M_E, res_e)]:
        pCM = (pi_mass**2 - mB**2) / (2.0 * pi_mass)
        EB_cm = np.sqrt(mB**2 + pCM**2)

        pB_par_lab = gamma * (pCM * cos_cm) + bg * EB_cm
        pC_par_lab = gamma * (-pCM * cos_cm) + bg * pCM
        p_per_lab = pCM * sin_cm

        pB_lab = np.sqrt(pB_par_lab**2 + p_per_lab**2)
        pC_lab = np.sqrt(pC_par_lab**2 + p_per_lab**2)

        tB_lab = np.arccos(np.clip(pB_par_lab / (pB_lab + 1e-30), -1.0, 1.0)) * (
            180.0 / np.pi
        )
        tC_lab = np.arccos(np.clip(pC_par_lab / (pC_lab + 1e-30), -1.0, 1.0)) * (
            180.0 / np.pi
        )

        cos_open = (pB_par_lab * pC_par_lab + p_per_lab**2) / (pB_lab * pC_lab + 1e-30)
        open_ang = np.arccos(np.clip(cos_open, -1.0, 1.0)) * (180.0 / np.pi)

        res[0] = fast_binning(pi_mom, MOM_BINS)
        res[1] = fast_binning(theta_cm_deg, ANG_BINS)
        res[2] = fast_binning(pB_lab, MOM_BINS)
        res[3] = fast_binning(pC_lab, MOM_BINS)
        res[4] = fast_binning(tB_lab, ANG_BINS)
        res[5] = fast_binning(tC_lab, ANG_BINS)
        res[6] = fast_binning(open_ang, ANG_BINS)

        low, high, n_prof = PROF_BINS
        inv_w = n_prof / (high - low)
        pn = np.zeros(n_prof, dtype=np.float64)
        psy = np.zeros(n_prof, dtype=np.float64)
        psy2 = np.zeros(n_prof, dtype=np.float64)

        for i in range(n):
            pm = pi_mom[i]
            if low <= pm < high:
                idx = int((pm - low) * inv_w)
                oa = open_ang[i]
                pn[idx] += 1
                psy[idx] += oa
                psy2[idx] += oa**2

        res[7], res[8], res[9] = pn, psy, psy2

    return res_mu, res_e


def make_accumulator():
    return {
        "pPi": np.zeros(MOM_BINS[2]),
        "tB_cm": np.zeros(ANG_BINS[2]),
        "pB_lab": np.zeros(MOM_BINS[2]),
        "pC_lab": np.zeros(MOM_BINS[2]),
        "tB_lab": np.zeros(ANG_BINS[2]),
        "tC_lab": np.zeros(ANG_BINS[2]),
        "open": np.zeros(ANG_BINS[2]),
        "pn": np.zeros(PROF_BINS[2]),
        "psy": np.zeros(PROF_BINS[2]),
        "psy2": np.zeros(PROF_BINS[2]),
    }


def add_to_acc(acc, data):
    acc["pPi"] += data[0]
    acc["tB_cm"] += data[1]
    acc["pB_lab"] += data[2]
    acc["pC_lab"] += data[3]
    acc["tB_lab"] += data[4]
    acc["tC_lab"] += data[5]
    acc["open"] += data[6]
    acc["pn"] += data[7]
    acc["psy"] += data[8]
    acc["psy2"] += data[9]


def _np_to_th1(name, title, params, counts):
    low, high, bins = params
    h = ROOT.TH1D(name, title, bins, low, high)
    h.SetDirectory(0)
    for i in range(bins):
        h.SetBinContent(i + 1, float(counts[i]))
        h.SetBinError(i + 1, np.sqrt(float(counts[i])))
    return h


def draw_and_save(acc, prefix, label_decay, label_B, outfile):
    hists = {
        "pPi": _np_to_th1(
            f"{prefix}_pPi",
            f"(a) #pi momentum distribution [{label_decay}];Momentum (#pi) [MeV/c];Counts (a.u.)",
            MOM_BINS,
            acc["pPi"],
        ),
        "tB_cm": _np_to_th1(
            f"{prefix}_tB_cm",
            f"(b) Isotropic CM decay angle of {label_B};#theta_{{CM}} ({label_B}) [deg];Normalized counts",
            ANG_BINS,
            acc["tB_cm"],
        ),
        "pB_lab": _np_to_th1(
            f"{prefix}_pB_lab",
            f"(c) LAB momentum: {label_B} vs #nu;Momentum [MeV/c];Counts",
            MOM_BINS,
            acc["pB_lab"],
        ),
        "pC_lab": _np_to_th1(f"{prefix}_pC_lab", "C lab", MOM_BINS, acc["pC_lab"]),
        "tB_lab": _np_to_th1(
            f"{prefix}_tB_lab",
            f"(d) LAB angle: {label_B} vs #nu;#theta_{{LAB}} [deg];Counts",
            ANG_BINS,
            acc["tB_lab"],
        ),
        "tC_lab": _np_to_th1(f"{prefix}_tC_lab", "C ang", ANG_BINS, acc["tC_lab"]),
        "open": _np_to_th1(
            f"{prefix}_open",
            f"(e) Opening angle ({label_B}, #nu) in LAB;#Delta#theta_{{LAB}} [deg];Normalized counts",
            ANG_BINS,
            acc["open"],
        ),
    }

    h_prof = ROOT.TH1D(
        f"{prefix}_prof",
        "(f) Avg. opening angle vs #pi momentum;Momentum (#pi) [MeV/c];<#Delta#theta_{{LAB}}> [deg]",
        PROF_BINS[2],
        PROF_BINS[0],
        PROF_BINS[1],
    )
    h_prof.SetDirectory(0)
    for i in range(PROF_BINS[2]):
        n = acc["pn"][i]
        if n > 10:
            mean = acc["psy"][i] / n
            err = np.sqrt(max(0, acc["psy2"][i] / n - mean**2) / n)
            h_prof.SetBinContent(i + 1, mean)
            h_prof.SetBinError(i + 1, err)
    hists["prof"] = h_prof

    for k in ["tB_cm", "open"]:
        if hists[k].Integral() > 0:
            hists[k].Scale(1.0 / hists[k].Integral("width"))

    c = ROOT.TCanvas(f"c_{prefix}", f"Kinematics: {label_decay}", 1200, 1300)
    c.Divide(2, 3)
    refs = []

    # Pad 1 — pi momentum
    c.cd(1)
    ROOT.gPad.SetLogy(True)
    hists["pPi"].SetLineColor(ROOT.kBlue + 1)
    hists["pPi"].SetLineWidth(2)
    hists["pPi"].Draw("HIST")
    refs.append(hists["pPi"])

    # Pad 2 — CM angle
    c.cd(2)
    hists["tB_cm"].SetLineColor(ROOT.kBlue + 1)
    hists["tB_cm"].SetLineWidth(2)
    hists["tB_cm"].Draw("HIST")
    refs.append(hists["tB_cm"])

    # Pad 3 — LAB momentum (B + neutrino overlaid)
    c.cd(3)
    ROOT.gPad.SetLogy(True)
    hists["pB_lab"].SetLineColor(ROOT.kBlack)
    hists["pB_lab"].SetLineWidth(2)
    hists["pC_lab"].SetLineColor(ROOT.kRed)
    hists["pC_lab"].SetLineStyle(3)
    hists["pC_lab"].SetLineWidth(2)
    max_y = max(hists["pB_lab"].GetMaximum(), hists["pC_lab"].GetMaximum())
    hists["pB_lab"].SetMaximum(max_y * 10)
    hists["pB_lab"].Draw("HIST")
    hists["pC_lab"].Draw("HIST SAME")
    leg3 = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
    leg3.AddEntry(hists["pB_lab"], label_B, "l")
    leg3.AddEntry(hists["pC_lab"], "#nu", "l")
    leg3.Draw()
    refs.extend([hists["pB_lab"], hists["pC_lab"], leg3])

    # Pad 4 — LAB angle (B + neutrino overlaid)
    c.cd(4)
    ROOT.gPad.SetLogy(True)
    hists["tB_lab"].SetLineColor(ROOT.kBlack)
    hists["tB_lab"].SetLineWidth(2)
    hists["tC_lab"].SetLineColor(ROOT.kRed)
    hists["tC_lab"].SetLineStyle(3)
    hists["tC_lab"].SetLineWidth(2)
    max_y = max(hists["tB_lab"].GetMaximum(), hists["tC_lab"].GetMaximum())
    hists["tB_lab"].SetMaximum(max_y * 10)
    hists["tB_lab"].Draw("HIST")
    hists["tC_lab"].Draw("HIST SAME")
    leg4 = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
    leg4.AddEntry(hists["tB_lab"], label_B, "l")
    leg4.AddEntry(hists["tC_lab"], "#nu", "l")
    leg4.Draw()
    refs.extend([hists["tB_lab"], hists["tC_lab"], leg4])

    # Pad 5 — opening angle
    c.cd(5)
    hists["open"].SetLineColor(ROOT.kBlue + 1)
    hists["open"].SetLineWidth(2)
    hists["open"].Draw("HIST")
    refs.append(hists["open"])

    # Pad 6 — profile
    c.cd(6)
    hists["prof"].SetLineColor(ROOT.kBlue + 1)
    hists["prof"].SetLineWidth(2)
    hists["prof"].Draw("HIST E")
    refs.append(hists["prof"])

    c.SaveAs(outfile)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except:
        pass

    seeds = np.random.SeedSequence(42).generate_state(N_CHUNKS)
    mu_acc = make_accumulator()
    e_acc = make_accumulator()

    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        for res_mu, res_e in tqdm.tqdm(
            pool.imap_unordered(run_physics_chunk, seeds), total=N_CHUNKS
        ):
            add_to_acc(mu_acc, res_mu)
            add_to_acc(e_acc, res_e)

    draw_and_save(mu_acc, "mu", "#pi #rightarrow #mu + #nu", "#mu", "fig_pi_mu_nu.png")
    draw_and_save(e_acc, "e", "#pi #rightarrow e + #nu", "e", "fig_pi_e_nu.png")
