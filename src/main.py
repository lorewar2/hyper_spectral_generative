import os
import h5py
from sklearn.decomposition import NMF
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg
import numpy as np
import h5py
import matplotlib.pyplot as plt

SUBTRACT_DARK    = False   # subtract stored 2-component background (darkConc @ darkSpectra), probably already done
APPLY_CORRECTION = False   # multiply by header/correctionFactor (spectral response), probably already done
N_COMPONENTS     = 5      # fluorophore components to resolve (keep <= maxComponents)
MAX_ITER         = 20    # MCR-ALS iterations

def main():
    # get all the files to process and save paths
    all_files = []
    for i in [4, 7, 9]:
        path = "./data/2020-08-1{}/Preprocessed".format(i)
        for file in os.listdir(path):
            save_path = "./result/" + path.split("/")[2] + "/" + file.split(".")[0] + ".png"
            file_path = path + "/" + file
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #print(file_path, save_path)
            all_files.append((file_path, save_path))
    print(len(all_files))
    for (file_path, save_path) in all_files:
        try:
            run_file(file_path, save_path)
        except:
            print("error " + file_path)
    return

def run_file(file_path, save_path):
    # load data
    with h5py.File(file_path, "r") as f:
        data   = np.asarray(f["data"], dtype=np.float64)             # (pixels, channels)
        wl     = np.asarray(f["header/channelWavelengths"]).ravel()  # (channels,) nm
        corr   = np.asarray(f["header/correctionFactor"]).ravel()    # (channels,)
        dark_S = np.asarray(f["header/darkPureSpectra"])             # (2, channels)
        dark_C = np.asarray(f["header/darkConcentrations"])          # (pixels, 2)
        nRows  = int(round(sca(f["header/nRows"])))
        nCols  = int(round(sca(f["header/nCols"])))
        nWav   = int(round(sca(f["header/nWavelengths"])))
        nFr    = int(round(sca(f["header/nFrames"])))
        maxC   = int(round(sca(f["header/maxComponents"])))
        preproc = mstr(f["header/preprocessType"])
        fname   = mstr(f["header/Filename"])

    print(f"file           : {fname}")
    print(f"preprocessType : {preproc}")
    print(f"data           : {data.shape}  (pixels x channels)")
    print(f"grid           : {nRows} x {nCols} = {nRows*nCols}   nFrames={nFr}  nWav={nWav}")
    print(f"maxComponents  : {maxC}")
    print(f"wavelengths    : {wl.min():.1f}-{wl.max():.1f} nm across {wl.size} channels")

    assert data.shape[0] == nRows * nCols, (
        "pixel count != nRows*nCols -- likely multiple frames; reshape as "
        "(nRows, nCols, nFr, nWav) and pick a frame before unmixing.")
    assert data.shape[1] == wl.size, "channel count mismatch"

    # ---- preprocessing ---------------------------------------------------------
    D = data.copy()
    if SUBTRACT_DARK:
        # background contribution per pixel = concentrations @ pure background spectra
        D = D - dark_C @ dark_S
    if APPLY_CORRECTION:
        # flatten the detector's wavelength-dependent response
        D = D * corr[None, :]           # if intensities look inverted vs nm, try /corr instead
    D = np.clip(D, 0.0, None)           # MCR requires non-negative data

    # image cube for inspection: order="F" inverts MATLAB's column-major pixel flattening
    cube = D.reshape(nRows, nCols, nWav, order="F")

    # ---- MCR-ALS ---------------------------------------------------------------
    # Initialize with NMF (good starting spectra), then refine with constrained
    # alternating least squares -- the same idea as Sandia's MCR.


    nmf = NMF(n_components=N_COMPONENTS, init="nndsvda", max_iter=500, random_state=0)
    _  = nmf.fit_transform(D)
    S0 = nmf.components_                 # (k, channels) initial pure spectra

    mcr = McrAR(max_iter=MAX_ITER, c_regr="NNLS", st_regr="NNLS",
                c_constraints=[ConstraintNonneg()],
                st_constraints=[ConstraintNonneg()], tol_err_change=1e-8)
    mcr.fit(D, ST=S0)

    C = mcr.C_opt_                       # (pixels, k)  abundances
    S = mcr.ST_opt_                      # (k, channels) resolved pure emission spectra
    print(f"MCR lack-of-fit: {mcr.err[-1]:.4g}   over {len(mcr.err)} iterations")

    # per-component abundance images (same reshape convention as the cube)
    maps = C.reshape(nRows, nCols, N_COMPONENTS, order="F")

    # ---- plot ------------------------------------------------------------------
    
    fig, ax = plt.subplots(2, N_COMPONENTS, figsize=(3 * N_COMPONENTS, 6),
                        squeeze=False)
    for k in range(N_COMPONENTS):
        ax[0, k].imshow(maps[:, :, k], cmap="inferno")
        ax[0, k].set_title(f"Component {k+1}")
        ax[0, k].axis("off")
        ax[1, k].plot(wl, S[k])
        ax[1, k].set_xlabel("emission (nm)")
        ax[1, k].set_ylabel("intensity" if k == 0 else "")
    fig.suptitle(save_path.split("/")[-1] + " " + "MCR-ALS: abundance maps (top) and pure spectra (bottom)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print("saved" + save_path)
    #plt.show()

# `cube`, `maps`, `S`, `wl` are now in memory for further analysis.

# helper functions
def check_data(PATH):
    with h5py.File(PATH, "r") as f:
        print("=== structure ===")
        f.visititems(show)
        print("\n=== root attrs ===")
        for k, v in f.attrs.items():
            print(f"  {k}: {v}")

def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            cls = obj.attrs.get("MATLAB_class", b"")
            cls = cls.decode() if isinstance(cls, bytes) else cls
            print(f"[data]  {name:35s} shape={obj.shape}  dtype={obj.dtype}  matlab_class={cls}")
        else:
            print(f"[group] {name}")

def mstr(ds):               # decode a MATLAB char array (stored as uint16) -> str
    a = np.asarray(ds).ravel()
    return "".join(chr(int(v)) for v in a if v != 0)


def sca(ds):                # read a 1x1 scalar
    return float(np.asarray(ds).ravel()[0])

if __name__ == "__main__":
    main()