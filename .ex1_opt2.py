import numpy as np
import tqdm
import ROOT
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

M_PI = 139.57018
SIG_PI = 0.00035
M_MU = 105.65837
M_E = 0.51100
M_NU = 0.0

CHUNK_SIZE = 2_000_000
N_CHUNKS = 250  # same for both species
EXP_MOM_MEAN = 60.0

N_WORKERS = max(1, multiprocessing.cpu_count() - 1)
_SEED_RNG = np.random.default_rng(seed=42)

# Bin edges at module level — never pickled across the IPC boundary
MOM_EDGES = np.linspace(0.0, 500.0, 101, dtype=np.float64)
ANG_EDGES = np.linspace(0.0, 180.0, 91, dtype=np.float64)
PROF_EDGES = np.linspace(0.0, 500.0, 26, dtype=np.float64)
N_PROF = 25


# ---------------------------------------------------------------------------
# Physics kernel + histogramming — runs in worker processes
# ---------------------------------------------------------------------------


def process_and_histogram(args):
    """
    Physics + numpy histogramming in one worker call.
    Returns the species tag and ~6 KB of bin arrays (not raw events).
    """
    seed, n, mB_val, tag = args
    rng = np.random.default_rng(seed)
    f32 = np.float32

    pi_mass = rng.normal(M_PI, SIG_PI, n).astype(f32)
    pi_mom = rng.exponential(EXP_MOM_MEAN, n).astype(f32)
    mB = f32(mB_val)

    disc = (pi_mass**2 - (mB + M_NU) ** 2) * (pi_mass**2 - (mB - M_NU) ** 2)
    pCM = np.sqrt(np.clip(disc, f32(0), None)) / (f32(2) * pi_mass)

    EB_cm = np.sqrt(mB**2 + pCM**2)

    cos_cm = rng.uniform(-1.0, 1.0, n).astype(f32)
    per_cm = pCM * np.sqrt(np.clip(f32(1) - cos_cm**2, f32(0), None))

    E_pi = np.sqrt(pi_mass**2 + pi_mom**2)
    gamma = E_pi / pi_mass
    bg = pi_mom / pi_mass

    pB_par_lab = gamma * (pCM * cos_cm) + bg * EB_cm
    pC_par_lab = gamma * (-pCM * cos_cm) + bg * pCM
    pB_per_lab = per_cm
    pC_per_lab = per_cm

    pB_lab = np.sqrt(pB_par_lab**2 + pB_per_lab**2)
    pC_lab = np.sqrt(pC_par_lab**2 + pC_per_lab**2)

    EPS = f32(1e-30)
    theta_B_cm = np.degrees(np.arccos(np.clip(cos_cm, f32(-1), f32(1)))).astype(
        np.float64
    )
    theta_B_lab = np.degrees(
        np.arccos(np.clip(pB_par_lab / np.maximum(pB_lab, EPS), f32(-1), f32(1)))
    ).astype(np.float64)
    theta_C_lab = np.degrees(
        np.arccos(np.clip(pC_par_lab / np.maximum(pC_lab, EPS), f32(-1), f32(1)))
    ).astype(np.float64)

    cos_open = (pB_par_lab * pC_par_lab + pB_per_lab * pC_per_lab) / np.maximum(
        pB_lab * pC_lab, EPS
    )
    open_ang = np.degrees(np.arccos(np.clip(cos_open, f32(-1), f32(1)))).astype(
        np.float64
    )

    pi_mom_f64 = pi_mom.astype(np.float64)

    c_pPi, _ = np.histogram(pi_mom_f64, bins=MOM_EDGES)
    c_thetaB_cm, _ = np.histogram(theta_B_cm, bins=ANG_EDGES)
    c_pB_lab, _ = np.histogram(pB_lab.astype(np.float64), bins=MOM_EDGES)
    c_pC_lab, _ = np.histogram(pC_lab.astype(np.float64), bins=MOM_EDGES)
    c_tB_lab, _ = np.histogram(theta_B_lab, bins=ANG_EDGES)
    c_tC_lab, _ = np.histogram(theta_C_lab, bins=ANG_EDGES)
    c_open_ang, _ = np.histogram(open_ang, bins=ANG_EDGES)

    bin_idx = np.searchsorted(PROF_EDGES, pi_mom_f64, side="right") - 1
    valid = (bin_idx >= 0) & (bin_idx < N_PROF)
    bi = bin_idx[valid]
    oa = open_ang[valid]
    prof_n = np.bincount(bi, minlength=N_PROF).astype(np.int64)
    prof_sy = np.bincount(bi, weights=oa, minlength=N_PROF)
    prof_sy2 = np.bincount(bi, weights=oa**2, minlength=N_PROF)

    return tag, (
        c_pPi,
        c_thetaB_cm,
        c_pB_lab,
        c_pC_lab,
        c_tB_lab,
        c_tC_lab,
        c_open_ang,
        prof_n,
        prof_sy,
        prof_sy2,
    )


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------


def make_accumulators():
    return dict(
        pPi=np.zeros(100, dtype=np.int64),
        thetaB_cm=np.zeros(90, dtype=np.int64),
        pB_lab=np.zeros(100, dtype=np.int64),
        pC_lab=np.zeros(100, dtype=np.int64),
        tB_lab=np.zeros(90, dtype=np.int64),
        tC_lab=np.zeros(90, dtype=np.int64),
        open_ang=np.zeros(90, dtype=np.int64),
        prof_n=np.zeros(N_PROF, dtype=np.int64),
        prof_sy=np.zeros(N_PROF, dtype=np.float64),
        prof_sy2=np.zeros(N_PROF, dtype=np.float64),
    )


def accumulate(acc, result):
    (
        c_pPi,
        c_thetaB_cm,
        c_pB_lab,
        c_pC_lab,
        c_tB_lab,
        c_tC_lab,
        c_open_ang,
        prof_n,
        prof_sy,
        prof_sy2,
    ) = result

    acc["pPi"] += c_pPi
    acc["thetaB_cm"] += c_thetaB_cm
    acc["pB_lab"] += c_pB_lab
    acc["pC_lab"] += c_pC_lab
    acc["tB_lab"] += c_tB_lab
    acc["tC_lab"] += c_tC_lab
    acc["open_ang"] += c_open_ang
    acc["prof_n"] += prof_n
    acc["prof_sy"] += prof_sy
    acc["prof_sy2"] += prof_sy2


# ---------------------------------------------------------------------------
# ROOT histogram construction from accumulators
# ---------------------------------------------------------------------------


def _np_to_th1(name, title, edges, counts):
    h = ROOT.TH1D(name, title, len(counts), edges[0], edges[-1])
    h.SetDirectory(0)
    for i, c in enumerate(counts):
        h.SetBinContent(i + 1, float(c))
        h.SetBinError(i + 1, float(np.sqrt(c)))
    h.SetEntries(float(counts.sum()))
    return h


def build_root_histograms(acc, prefix, label_decay, label_B):
    hists = {}
    hists["pPi"] = _np_to_th1(
        f"{prefix}_pPi",
        f"(a) #pi momentum distribution [{label_decay}];"
        "Momentum (#pi) [MeV/c];Counts (a.u.)",
        MOM_EDGES,
        acc["pPi"],
    )
    hists["thetaB_cm"] = _np_to_th1(
        f"{prefix}_thetaB_cm",
        f"(b) Isotropic CM decay angle of {label_B};"
        f"#theta_{{CM}} ({label_B}) [deg];Normalized counts",
        ANG_EDGES,
        acc["thetaB_cm"],
    )
    hists["pB_lab"] = _np_to_th1(
        f"{prefix}_pB_lab",
        f"(c) LAB momentum: {label_B} vs #nu;Momentum [MeV/c];Counts",
        MOM_EDGES,
        acc["pB_lab"],
    )
    hists["pC_lab"] = _np_to_th1(f"{prefix}_pC_lab", "C lab", MOM_EDGES, acc["pC_lab"])
    hists["tB_lab"] = _np_to_th1(
        f"{prefix}_tB_lab",
        f"(d) LAB angle: {label_B} vs #nu;#theta_{{LAB}} [deg];Counts",
        ANG_EDGES,
        acc["tB_lab"],
    )
    hists["tC_lab"] = _np_to_th1(f"{prefix}_tC_lab", "C ang", ANG_EDGES, acc["tC_lab"])
    hists["open_ang"] = _np_to_th1(
        f"{prefix}_open_ang",
        f"(e) Opening angle ({label_B}, #nu) in LAB;"
        "#Delta#theta_{LAB} [deg];Normalized counts",
        ANG_EDGES,
        acc["open_ang"],
    )
    h_prof = ROOT.TH1D(
        f"{prefix}_prof",
        "(f) Avg. opening angle vs #pi momentum;"
        "Momentum (#pi) [MeV/c];<#Delta#theta_{LAB}> [deg]",
        N_PROF,
        PROF_EDGES[0],
        PROF_EDGES[-1],
    )
    h_prof.SetDirectory(0)
    for i in range(N_PROF):
        ni = acc["prof_n"][i]
        if ni > 10:
            mean = acc["prof_sy"][i] / ni
            var = acc["prof_sy2"][i] / ni - mean**2
            err = np.sqrt(max(var, 0.0) / ni)
            h_prof.SetBinContent(i + 1, mean)
            h_prof.SetBinError(i + 1, err)
    h_prof.SetEntries(float(acc["prof_n"].sum()))
    hists["prof"] = h_prof
    return hists


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_figure(hists, label_B, outfile):
    h_pPi = hists["pPi"]
    h_thetaB = hists["thetaB_cm"]
    h_pB_lab = hists["pB_lab"]
    h_pC_lab = hists["pC_lab"]
    h_tB_lab = hists["tB_lab"]
    h_tC_lab = hists["tC_lab"]
    h_open = hists["open_ang"]
    h_prof = hists["prof"]

    for h in (h_thetaB, h_open):
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral("width"))

    c = ROOT.TCanvas("c", "Kinematics", 1200, 1300)
    c.Divide(2, 3)

    c.cd(1)
    ROOT.gPad.SetLogy(True)
    h_pPi.SetLineColor(ROOT.kBlue + 1)
    h_pPi.SetLineWidth(2)
    h_pPi.Draw("HIST")

    c.cd(2)
    h_thetaB.SetLineColor(ROOT.kBlue + 1)
    h_thetaB.SetLineWidth(2)
    h_thetaB.Draw("HIST")

    c.cd(3)
    ROOT.gPad.SetLogy(True)
    h_pB_lab.SetLineColor(ROOT.kBlack)
    h_pB_lab.SetLineWidth(2)
    h_pC_lab.SetLineColor(ROOT.kRed)
    h_pC_lab.SetLineStyle(3)
    h_pC_lab.SetLineWidth(2)
    h_pB_lab.SetMaximum(max(h_pB_lab.GetMaximum(), h_pC_lab.GetMaximum()) * 10)
    h_pB_lab.Draw("HIST")
    h_pC_lab.Draw("HIST SAME")
    leg_c = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
    leg_c.AddEntry(h_pB_lab, label_B, "l")
    leg_c.AddEntry(h_pC_lab, "#nu", "l")
    leg_c.Draw()

    c.cd(4)
    ROOT.gPad.SetLogy(True)
    h_tB_lab.SetLineColor(ROOT.kBlack)
    h_tB_lab.SetLineWidth(2)
    h_tC_lab.SetLineColor(ROOT.kRed)
    h_tC_lab.SetLineStyle(3)
    h_tC_lab.SetLineWidth(2)
    h_tB_lab.SetMaximum(max(h_tB_lab.GetMaximum(), h_tC_lab.GetMaximum()) * 10)
    h_tB_lab.Draw("HIST")
    h_tC_lab.Draw("HIST SAME")
    leg_d = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
    leg_d.AddEntry(h_tB_lab, label_B, "l")
    leg_d.AddEntry(h_tC_lab, "#nu", "l")
    leg_d.Draw()

    c.cd(5)
    h_open.SetLineColor(ROOT.kBlue + 1)
    h_open.SetLineWidth(2)
    h_open.Draw("HIST")

    c.cd(6)
    h_prof.SetLineColor(ROOT.kBlue + 1)
    h_prof.SetLineWidth(2)
    h_prof.Draw("HIST E")

    c.SaveAs(outfile)


# ---------------------------------------------------------------------------
# Single combined pass — both species submitted to the pool together
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate all seeds upfront for both species
    seeds_mu = _SEED_RNG.integers(0, 2**32, size=N_CHUNKS)
    seeds_e = _SEED_RNG.integers(0, 2**32, size=N_CHUNKS)

    # Interleave mu/e jobs so workers stay busy on both species simultaneously.
    # Wall time → max(t_mu, t_e) instead of t_mu + t_e  (~2× faster).
    args = []
    for s_mu, s_e in zip(seeds_mu, seeds_e):
        args.append((int(s_mu), CHUNK_SIZE, M_MU, "mu"))
        args.append((int(s_e), CHUNK_SIZE, M_E, "e"))

    mu_acc = make_accumulators()
    e_acc = make_accumulators()
    acc_map = {"mu": mu_acc, "e": e_acc}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for tag, result in tqdm.tqdm(
            pool.map(process_and_histogram, args, chunksize=8),
            total=len(args),
            desc="mu+e chunks",
        ):
            accumulate(acc_map[tag], result)

    # Build histograms and draw — done once per species after all chunks finish
    for prefix, label_decay, label_B, outfile, acc in [
        ("mu", "#pi #rightarrow #mu + #nu", "#mu", "fig3_pi_mu_nu.png", mu_acc),
        ("e", "#pi #rightarrow e + #nu", "e", "fig4_pi_e_nu.png", e_acc),
    ]:
        hists = build_root_histograms(acc, prefix, label_decay, label_B)
        draw_figure(hists, label_B, outfile)
