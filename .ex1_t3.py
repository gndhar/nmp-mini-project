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

# ---------------------------------------------------------------------------
# Numba Kernels
# ---------------------------------------------------------------------------


@njit(fastmath=True, cache=True)
def fast_binning(data, bin_params):
    low, high, n_bins = bin_params
    inv_width = n_bins / (high - low)
    counts = np.zeros(n_bins, dtype=np.float64)
    for i in range(len(data)):
        val = data[i]
        if low <= val < high:
            counts[int((val - low) * inv_width)] += 1.0
    return counts


@njit(fastmath=True, cache=True)
def _profile_loop(pi_mom, open_ang, prof_bins):
    low, high, n_prof = prof_bins
    inv_w = n_prof / (high - low)
    pn = np.zeros(n_prof, dtype=np.float64)
    psy = np.zeros(n_prof, dtype=np.float64)
    psy2 = np.zeros(n_prof, dtype=np.float64)
    for i in range(len(pi_mom)):
        pm = pi_mom[i]
        if low <= pm < high:
            idx = int((pm - low) * inv_w)
            oa = open_ang[i]
            pn[idx] += 1.0
            psy[idx] += oa
            psy2[idx] += oa * oa
    return pn, psy, psy2


@njit(fastmath=True, cache=True)
def _decay_branch(pi_mass, pi_mom, gamma, bg, cos_cm, sin_cm, theta_cm_deg, mB):
    pCM = (pi_mass * pi_mass - mB * mB) / (2.0 * pi_mass)
    EB_cm = np.sqrt(mB * mB + pCM * pCM)

    pB_par = gamma * (pCM * cos_cm) + bg * EB_cm
    pC_par = gamma * (-pCM * cos_cm) + bg * pCM
    p_per = pCM * sin_cm

    pB_lab = np.sqrt(pB_par * pB_par + p_per * p_per)
    pC_lab = np.sqrt(pC_par * pC_par + p_per * p_per)

    tB_lab = np.arccos(np.clip(pB_par / (pB_lab + 1e-30), -1.0, 1.0)) * (180.0 / np.pi)
    tC_lab = np.arccos(np.clip(pC_par / (pC_lab + 1e-30), -1.0, 1.0)) * (180.0 / np.pi)

    cos_open = (pB_par * pC_par + p_per * p_per) / (pB_lab * pC_lab + 1e-30)
    open_ang = np.arccos(np.clip(cos_open, -1.0, 1.0)) * (180.0 / np.pi)

    h_pB = fast_binning(pB_lab, MOM_BINS)
    h_pC = fast_binning(pC_lab, MOM_BINS)
    h_tBl = fast_binning(tB_lab, ANG_BINS)
    h_tCl = fast_binning(tC_lab, ANG_BINS)
    h_tBc = fast_binning(theta_cm_deg, ANG_BINS)
    h_oa = fast_binning(open_ang, ANG_BINS)

    pn, psy, psy2 = _profile_loop(pi_mom, open_ang, PROF_BINS)
    return (h_pB, h_pC, h_tBl, h_tCl, h_tBc, h_oa, pn, psy, psy2)


@njit(fastmath=True, cache=True)
def run_physics_chunk(seed):
    np.random.seed(seed)
    n = CHUNK_SIZE

    pi_mass = np.random.normal(M_PI, SIG_PI, n)
    pi_mom = np.random.exponential(EXP_MOM_MEAN, n)
    E_pi = np.sqrt(pi_mass * pi_mass + pi_mom * pi_mom)
    gamma = E_pi / pi_mass
    bg = pi_mom / pi_mass

    cos_cm = np.random.uniform(-1.0, 1.0, n)
    sin_cm = np.sqrt(1.0 - cos_cm * cos_cm)
    theta_cm_deg = np.arccos(cos_cm) * (180.0 / np.pi)

    h_pPi = fast_binning(pi_mom, MOM_BINS)  # computed once, shared
    res_mu = _decay_branch(
        pi_mass, pi_mom, gamma, bg, cos_cm, sin_cm, theta_cm_deg, M_MU
    )
    res_e = _decay_branch(pi_mass, pi_mom, gamma, bg, cos_cm, sin_cm, theta_cm_deg, M_E)
    return h_pPi, res_mu, res_e


def _warmup():
    """
    Force full JIT compilation in the main process.
    cache=True writes the compiled objects to disk so spawned
    workers load them instantly instead of recompiling.
    """
    print("Pre-warming Numba JIT cache (first run only)...")
    seed = np.random.SeedSequence(0).generate_state(1)[0]
    run_physics_chunk(int(seed))
    print("JIT cache ready.")


# ---------------------------------------------------------------------------
# ROOT Bridge & Accumulation
# ---------------------------------------------------------------------------


def make_accumulator():
    return {
        k: np.zeros(b[2])
        for k, b in [
            ("pPi", MOM_BINS),
            ("tB_cm", ANG_BINS),
            ("pB", MOM_BINS),
            ("pC", MOM_BINS),
            ("tB", ANG_BINS),
            ("tC", ANG_BINS),
            ("open", ANG_BINS),
            ("pn", PROF_BINS),
            ("psy", PROF_BINS),
            ("psy2", PROF_BINS),
        ]
    }


def add_to_acc(acc, h_pPi, res):
    h_pB, h_pC, h_tBl, h_tCl, h_tBc, h_oa, pn, psy, psy2 = res
    acc["pPi"] += h_pPi
    acc["tB_cm"] += h_tBc
    acc["pB"] += h_pB
    acc["pC"] += h_pC
    acc["tB"] += h_tBl
    acc["tC"] += h_tCl
    acc["open"] += h_oa
    acc["pn"] += pn
    acc["psy"] += psy
    acc["psy2"] += psy2


def _np_to_th1(name, title, params, counts):
    low, high, bins = params
    h = ROOT.TH1D(name, title, bins, low, high)
    h.SetDirectory(0)
    for i in range(bins):
        c = float(counts[i])
        h.SetBinContent(i + 1, c)
        h.SetBinError(i + 1, c**0.5)
    return h


def draw_and_save(acc, prefix, label_decay, label_B, outfile):
    hists = {
        "pPi": _np_to_th1(
            f"{prefix}_pPi",
            f"(a) #pi momentum [{label_decay}];[MeV/c];Counts",
            MOM_BINS,
            acc["pPi"],
        ),
        "tB_cm": _np_to_th1(
            f"{prefix}_tBcm",
            f"(b) CM angle {label_B};[deg];Normalized",
            ANG_BINS,
            acc["tB_cm"],
        ),
        "pB": _np_to_th1(
            f"{prefix}_pB", "(c) LAB momentum;[MeV/c];Counts", MOM_BINS, acc["pB"]
        ),
        "pC": _np_to_th1(f"{prefix}_pC", "C lab", MOM_BINS, acc["pC"]),
        "tB": _np_to_th1(
            f"{prefix}_tB",
            f"(d) LAB angle;#theta_{{LAB}} [deg];Counts",
            ANG_BINS,
            acc["tB"],
        ),
        "tC": _np_to_th1(f"{prefix}_tC", "C ang", ANG_BINS, acc["tC"]),
        "open": _np_to_th1(
            f"{prefix}_open",
            f"(e) Opening angle;#Delta#theta [deg];Normalized",
            ANG_BINS,
            acc["open"],
        ),
    }

    h_prof = ROOT.TH1D(
        f"{prefix}_prof",
        "(f) Avg. Opening Angle;P_{#pi} [MeV/c];<#Delta#theta>",
        PROF_BINS[2],
        PROF_BINS[0],
        PROF_BINS[1],
    )
    h_prof.SetDirectory(0)
    for i in range(PROF_BINS[2]):
        n = acc["pn"][i]
        if n > 10:
            mean = acc["psy"][i] / n
            var = max(0.0, acc["psy2"][i] / n - mean * mean)
            h_prof.SetBinContent(i + 1, mean)
            h_prof.SetBinError(i + 1, (var / n) ** 0.5)
    hists["prof"] = h_prof

    for k in ("tB_cm", "open"):
        if hists[k].Integral() > 0:
            hists[k].Scale(1.0 / hists[k].Integral("width"))

    c = ROOT.TCanvas(f"c_{prefix}", "", 1200, 1300)
    c.Divide(2, 3)
    for pad_i, k in enumerate(["pPi", "tB_cm", "pB", "tB", "open", "prof"], 1):
        c.cd(pad_i)
        if k in ("pPi", "pB", "tB"):
            ROOT.gPad.SetLogy(True)
        hists[k].SetLineColor(ROOT.kBlue + 1)
        hists[k].Draw("HIST" if k != "prof" else "HIST E")
        if k == "pB":
            hists["pC"].SetLineColor(ROOT.kRed)
            hists["pC"].SetLineStyle(2)
            hists["pC"].Draw("HIST SAME")
        if k == "tB":
            hists["tC"].SetLineColor(ROOT.kRed)
            hists["tC"].SetLineStyle(2)
            hists["tC"].Draw("HIST SAME")
    c.SaveAs(outfile)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # KEY: compile everything in the main process first.
    # Spawned workers find the cache on disk and skip recompilation entirely.
    _warmup()

    seeds = np.random.SeedSequence(42).generate_state(N_CHUNKS)
    mu_acc = make_accumulator()
    e_acc = make_accumulator()

    print(
        f"Running {N_CHUNKS} chunks × {CHUNK_SIZE:,} events on {N_WORKERS} workers..."
    )
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        for h_pPi, res_mu, res_e in tqdm.tqdm(
            pool.imap_unordered(run_physics_chunk, [int(s) for s in seeds]),
            total=N_CHUNKS,
        ):
            add_to_acc(mu_acc, h_pPi, res_mu)
            add_to_acc(e_acc, h_pPi, res_e)

    draw_and_save(mu_acc, "mu", "#pi #rightarrow #mu + #nu", "#mu", "fig_pi_mu_nu.png")
    draw_and_save(e_acc, "e", "#pi #rightarrow e + #nu", "e", "fig_pi_e_nu.png")
    print("Done.")
