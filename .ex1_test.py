import numpy as np
import tqdm
import ROOT
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Set ROOT to batch mode for performance
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Constants
M_PI, SIG_PI = 139.57018, 0.00035
M_MU = 105.65837
M_E = 0.51100
CHUNK_SIZE = 2_000_000
N_CHUNKS = 250
EXP_MOM_MEAN = 60.0

# macOS handling: Use "spawn" to avoid issues with ROOT/multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

N_WORKERS = max(1, multiprocessing.cpu_count() - 1)

# Binning Config
MOM_EDGES = (0.0, 500.0, 100)  # (min, max, n_bins)
ANG_EDGES = (0.0, 180.0, 90)
PROF_EDGES = (0.0, 500.0, 25)

# ---------------------------------------------------------------------------
# Fast Binning Utility
# ---------------------------------------------------------------------------


def fast_hist(data, edges_tuple):
    """Blazing fast histogramming for uniform bins using bincount."""
    low, high, bins = edges_tuple
    idx = ((data - low) * (bins / (high - low))).astype(np.int32)
    # Filter bounds efficiently
    mask = (idx >= 0) & (idx < bins)
    return np.bincount(idx[mask], minlength=bins)


# ---------------------------------------------------------------------------
# Combined Physics Kernel
# ---------------------------------------------------------------------------


def process_chunk(seed):
    rng = np.random.default_rng(seed)
    f32 = np.float32

    # 1. Shared Parent Kinematics (Sampled once for BOTH species)
    pi_mass = rng.normal(M_PI, SIG_PI, CHUNK_SIZE).astype(f32)
    pi_mom = rng.exponential(EXP_MOM_MEAN, CHUNK_SIZE).astype(f32)
    E_pi = np.sqrt(pi_mass**2 + pi_mom**2)
    gamma = E_pi / pi_mass
    bg = pi_mom / pi_mass
    cos_cm = rng.uniform(-1.0, 1.0, CHUNK_SIZE).astype(f32)
    sin_cm = np.sqrt(np.clip(f32(1) - cos_cm**2, 0, None))

    results = {}

    # 2. Process both species using the same parent events
    for mB, tag in [(M_MU, "mu"), (M_E, "e")]:
        # Since m_nu = 0, pCM simplifies:
        pCM = (pi_mass**2 - f32(mB**2)) / (f32(2) * pi_mass)
        EB_cm = np.sqrt(f32(mB**2) + pCM**2)

        # LAB Transformations
        pB_par_lab = gamma * (pCM * cos_cm) + bg * EB_cm
        pC_par_lab = gamma * (-pCM * cos_cm) + bg * pCM  # Nu has EB_cm = pCM
        p_per_lab = pCM * sin_cm

        pB_lab = np.sqrt(pB_par_lab**2 + p_per_lab**2)
        pC_lab = np.sqrt(pC_par_lab**2 + p_per_lab**2)

        # Angular calculations
        EPS = f32(1e-30)
        tB_lab = np.degrees(np.arccos(np.clip(pB_par_lab / (pB_lab + EPS), -1, 1)))
        tC_lab = np.degrees(np.arccos(np.clip(pC_par_lab / (pC_lab + EPS), -1, 1)))

        cos_open = (pB_par_lab * pC_par_lab + p_per_lab**2) / (pB_lab * pC_lab + EPS)
        open_ang = np.degrees(np.arccos(np.clip(cos_open, -1, 1)))

        # Profile logic (Opening angle vs Pi Momentum)
        p_idx = (
            (pi_mom - PROF_EDGES[0]) * (PROF_EDGES[2] / (PROF_EDGES[1] - PROF_EDGES[0]))
        ).astype(np.int32)
        p_mask = (p_idx >= 0) & (p_idx < PROF_EDGES[2])
        bi = p_idx[p_mask]
        oa = open_ang[p_mask]

        # Store hist data
        results[tag] = (
            fast_hist(pi_mom, MOM_EDGES),
            fast_hist(np.degrees(np.arccos(cos_cm)), ANG_EDGES),
            fast_hist(pB_lab, MOM_EDGES),
            fast_hist(pC_lab, MOM_EDGES),
            fast_hist(tB_lab, ANG_EDGES),
            fast_hist(tC_lab, ANG_EDGES),
            fast_hist(open_ang, ANG_EDGES),
            np.bincount(bi, minlength=PROF_EDGES[2]).astype(np.int64),  # prof_n
            np.bincount(bi, weights=oa, minlength=PROF_EDGES[2]),  # prof_sy
            np.bincount(bi, weights=oa**2, minlength=PROF_EDGES[2]),  # prof_sy2
        )

    return results


# ---------------------------------------------------------------------------
# Accumulators & ROOT Bridge
# ---------------------------------------------------------------------------


def make_acc():
    return {
        "pPi": np.zeros(MOM_EDGES[2], dtype=np.int64),
        "tB_cm": np.zeros(ANG_EDGES[2], dtype=np.int64),
        "pB_lab": np.zeros(MOM_EDGES[2], dtype=np.int64),
        "pC_lab": np.zeros(MOM_EDGES[2], dtype=np.int64),
        "tB_lab": np.zeros(ANG_EDGES[2], dtype=np.int64),
        "tC_lab": np.zeros(ANG_EDGES[2], dtype=np.int64),
        "open": np.zeros(ANG_EDGES[2], dtype=np.int64),
        "pn": np.zeros(PROF_EDGES[2], dtype=np.int64),
        "psy": np.zeros(PROF_EDGES[2], dtype=np.float64),
        "psy2": np.zeros(PROF_EDGES[2], dtype=np.float64),
    }


def update_acc(acc, data):
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


# (Drawing and TH1 conversion functions omitted for brevity, keep your original ones)
# Just ensure the bin edges passed to ROOT match MOM_EDGES, ANG_EDGES, etc.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seeds = np.random.default_rng(42).integers(0, 2**32, size=N_CHUNKS)

    mu_acc = make_acc()
    e_acc = make_acc()

    print(f"Starting simulation on {N_WORKERS} cores...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for res_dict in tqdm.tqdm(pool.map(process_chunk, seeds), total=N_CHUNKS):
            update_acc(mu_acc, res_dict["mu"])
            update_acc(e_acc, res_dict["e"])

    # Final Plotting Logic (Reuse your existing build_root_histograms and draw_figure)
    # ...
