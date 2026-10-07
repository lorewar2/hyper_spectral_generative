"""
Retrieve MCR results (finalSpectra + wavelengths) from a Sandia .MCRanalysis file.

The file is a MATLAB v7.3 (HDF5) save of an MCOS class object, so property names
like 'finalSpectra' are NOT stored as dataset names -- they live in the opaque
#subsystem#/MCOS blob. Instead of decoding MCOS, we identify the arrays by shape
and the documented property that finalSpectra is normalized to unit length.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

PATH = "/data/codes_git/hyper_spectral_generative/data/29Apr2021_142118 - SMA.MCRanalysis"  # adjust
OUT_PNG = "finalSpectra.png"
OUT_CSV = "finalSpectra.csv"

# Confirmed for this file; set to None to re-auto-detect on a different file.
PICK_SPECTRA = "#refs#/xm"   # 5-component, unit-length finalSpectra
PICK_WL      = "#refs#/bm"   # 512 wavelengths, 486-805 nm
N_CHANNELS   = 512


def scan(f):
    """Find candidate spectra (512 x N, N<=12) and wavelength axes (monotonic 512-vector)."""
    spec, wls = [], []
    def visit(name, o):
        if not isinstance(o, h5py.Dataset) or o.dtype.kind != "f" or o.ndim != 2:
            return
        if N_CHANNELS not in o.shape:
            return
        a = np.array(o)
        v = a if a.shape[0] == N_CHANNELS else a.T        # orient to (512, ncomp)
        ncomp = v.shape[1]
        col = v[:, 0]
        mono = np.all(np.diff(col) > 0) or np.all(np.diff(col) < 0)
        if ncomp == 1 and mono and 300 < np.nanmin(col) and np.nanmax(col) < 1200:
            wls.append(name)                               # looks like a wavelength axis
        elif 1 < ncomp <= 12:
            spec.append((name, ncomp, np.linalg.norm(v, axis=0)))
    f.visititems(visit)
    return spec, wls


def main():
    with h5py.File(PATH, "r") as f:
        spec, wls = scan(f)

        print("── spectra candidates (512 × N, N ≤ 12) ──")
        for name, n, norms in spec:
            tag = "UNIT-NORM" if np.allclose(norms, 1, atol=1e-2) else "not unit"
            print(f"  {name:14s} {n} comp   {tag}   first norms = {np.round(norms[:4], 3)}")
        print("── wavelength candidates ──\n  ", wls)

        # pick: explicit override, else the unit-norm spectra set, else the first candidate
        sp_name = PICK_SPECTRA or next(
            (n for n, _, q in spec if np.allclose(q, 1, atol=1e-2)), spec[0][0])
        wl_name = PICK_WL or (wls[0] if wls else None)
        print(f"\nusing finalSpectra = {sp_name} ,  wavelengths = {wl_name}")

        spec_arr = np.array(f[sp_name])
        wl = np.array(f[wl_name]).ravel() if wl_name else np.arange(float(N_CHANNELS))

    # orient to (nWav, nComp); each column is one component's pure spectrum
    if spec_arr.shape[0] != wl.size and spec_arr.shape[1] == wl.size:
        spec_arr = spec_arr.T

    print("finalSpectra:", spec_arr.shape, "| wavelengths:", wl.shape,
          f"({wl.min():.1f}-{wl.max():.1f} nm)")
    print("column L2 norms:", np.round(np.linalg.norm(spec_arr, axis=0), 3))

    # plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(spec_arr.shape[1]):
        ax.plot(wl, spec_arr[:, k], lw=1.3, label=f"comp {k + 1}")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("intensity (unit-norm, a.u.)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)

    # export: wavelength + one column per component
    np.savetxt(OUT_CSV, np.column_stack([wl, spec_arr]), delimiter=",",
               header="wavelength_nm," + ",".join(f"comp{i+1}" for i in range(spec_arr.shape[1])),
               comments="")
    print(f"wrote {OUT_PNG} and {OUT_CSV}")

    return wl, spec_arr


if __name__ == "__main__":
    wl, spec_arr = main()