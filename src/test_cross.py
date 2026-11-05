import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


# =====================================================================
# Bidirectional cross-attention layer (BiXT-style)
#   component tokens (K)  <->  pixel tokens (N)
#   one shared similarity matrix, normalised two ways
# =====================================================================
class BiCrossAttnLayer(nn.Module):
    def __init__(self, dim, n_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.h, self.dh = n_heads, dim // n_heads

        # components attend to each other (K is small -> cheap)
        self.norm_cs = nn.LayerNorm(dim)
        self.comp_sa = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

        # shared similarity + values
        self.norm_c   = nn.LayerNorm(dim)
        self.norm_p   = nn.LayerNorm(dim)
        self.q_comp   = nn.Linear(dim, dim)
        self.q_pix    = nn.Linear(dim, dim)
        self.v_comp   = nn.Linear(dim, dim)
        self.v_pix    = nn.Linear(dim, dim)
        self.out_comp = nn.Linear(dim, dim)
        self.out_pix  = nn.Linear(dim, dim)

        hidden = int(dim * mlp_ratio)
        self.norm_c2 = nn.LayerNorm(dim)
        self.norm_p2 = nn.LayerNorm(dim)
        self.mlp_c = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.mlp_p = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, comp, pix):
        # comp: (K, dim)   pix: (N, dim)
        K, N, H, dh = comp.shape[0], pix.shape[0], self.h, self.dh

        # 1) component self-attention
        cs = self.norm_cs(comp).unsqueeze(0)
        comp = comp + self.comp_sa(cs, cs, cs, need_weights=False)[0].squeeze(0)

        # 2) one shared similarity matrix
        c, p = self.norm_c(comp), self.norm_p(pix)
        qc = self.q_comp(c).view(K, H, dh).transpose(0, 1)   # (H, K, dh)
        qp = self.q_pix(p).view(N, H, dh).transpose(0, 1)    # (H, N, dh)
        vc = self.v_comp(c).view(K, H, dh).transpose(0, 1)   # (H, K, dh)
        vp = self.v_pix(p).view(N, H, dh).transpose(0, 1)    # (H, N, dh)
        sim = qc @ qp.transpose(-1, -2) / dh ** 0.5          # (H, K, N)

        # components gather from pixels (normalise over pixels)     -> refines spectra
        comp_upd = (sim.softmax(dim=-1) @ vp).transpose(0, 1).reshape(K, H * dh)
        # pixels gather from components (normalise over components) -> refines mixture
        pix_upd = (sim.softmax(dim=-2).transpose(-1, -2) @ vc).transpose(0, 1).reshape(N, H * dh)

        comp = comp + self.out_comp(comp_upd)
        pix  = pix  + self.out_pix(pix_upd)

        # 3) feed-forward
        comp = comp + self.mlp_c(self.norm_c2(comp))
        pix  = pix  + self.mlp_p(self.norm_p2(pix))
        return comp, pix


# =====================================================================
# Unmixing model:  D (N x C) -> C (N x K), S (K x C), D_hat = C @ S
#   S_init: starting spectra (K x C); fully learnable afterwards
# =====================================================================
def inv_softplus(x, eps=1e-6):
    x = torch.clamp(x, min=eps)
    return x + torch.log(-torch.expm1(-x))


class BiXTUnmix(nn.Module):
    def __init__(self, n_channels, n_components, S_init, dim=128, depth=4, n_heads=4):
        super().__init__()
        self.pix_embed   = nn.Sequential(nn.Linear(n_channels, dim), nn.GELU(), nn.Linear(dim, dim))
        self.comp_tokens = nn.Parameter(torch.randn(n_components, dim) * 0.02)
        self.layers      = nn.ModuleList([BiCrossAttnLayer(dim, n_heads) for _ in range(depth)])
        self.norm_c      = nn.LayerNorm(dim)
        self.norm_p      = nn.LayerNorm(dim)
        self.to_spectra  = nn.Linear(dim, n_channels)    # learned correction on top of the base
        self.to_conc     = nn.Linear(dim, n_components)

        S_init = torch.as_tensor(S_init, dtype=torch.float32)
        assert S_init.shape == (n_components, n_channels), \
            f'S_init must be ({n_components}, {n_channels}), got {tuple(S_init.shape)}'

        # learnable base spectra, starting at S_init (stored in softplus space)
        self.S_base = nn.Parameter(inv_softplus(S_init))
        # correction head starts at zero -> S == S_init at epoch 0
        nn.init.zeros_(self.to_spectra.weight)
        nn.init.zeros_(self.to_spectra.bias)
        # component tokens start as the embedding of their spectra
        with torch.no_grad():
            self.comp_tokens.copy_(self.pix_embed(S_init))

    def forward(self, D):
        pix  = self.pix_embed(D)       # (N, dim)
        comp = self.comp_tokens        # (K, dim)
        for layer in self.layers:
            comp, pix = layer(comp, pix)
        S = F.softplus(self.S_base + self.to_spectra(self.norm_c(comp)))   # (K, C)
        C = F.softplus(self.to_conc(self.norm_p(pix)))                     # (N, K)
        return C @ S, C, S


# =====================================================================
# Loss: reconstruction + MCR-style constraints
# =====================================================================
def unmix_loss(D, D_hat, C, S, w_smooth=1e-3, w_sparse=0.0, w_unit=1e-2, poisson=False):
    if poisson:
        rec = F.poisson_nll_loss(D_hat, D, log_input=False, full=False)
    else:
        rec = F.mse_loss(D_hat, D)
    smooth = ((S[:, 2:] - 2 * S[:, 1:-1] + S[:, :-2]) ** 2).mean()
    sparse = C.mean()
    unit   = ((S.norm(dim=1) - 1.0) ** 2).mean()   # keeps spectra unit-length (scale ambiguity)
    return rec + w_smooth * smooth + w_sparse * sparse + w_unit * unit


# =====================================================================
# Starting spectra: NMF initial guess
# =====================================================================
def unit_norm(S):
    S = np.clip(S, 0, None)
    return (S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)).astype('float32')


def nmf_init(D, n_components, seed=0):
    nmf = NMF(n_components=n_components, init="nndsvda", max_iter=1, random_state=seed)
    _ = nmf.fit_transform(D)
    S0 = nmf.components_                 # (k, channels) initial pure spectra
    return unit_norm(S0)


# =====================================================================
# Training
# =====================================================================
def train_unmix(D, n_components, S_init, dim=128, depth=4, n_heads=4,
                epochs=3000, lr=1e-3, weight_decay=0.0,
                w_smooth=1e-3, w_sparse=0.0, w_unit=1e-2, poisson=False,
                seed=0, device=None, print_every=100):
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    D = torch.as_tensor(D, dtype=torch.float32, device=device)
    model = BiXTUnmix(D.shape[1], n_components, S_init, dim, depth, n_heads).to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loss_list = np.zeros(epochs)
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        D_hat, C, S = model(D)            # full image each step
        loss = unmix_loss(D, D_hat, C, S, w_smooth, w_sparse, w_unit, poisson)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        loss_list[epoch] = loss.item()
        if epoch % print_every == 0:
            print(f'epoch {epoch:5d}  loss {loss.item():.6f}')

    model.eval()
    with torch.no_grad():
        D_hat, C, S = model(D)
        rel_resid = (torch.linalg.norm(D_hat - D) / torch.linalg.norm(D)).item()
    print(f'relative residual ||D - CS|| / ||D|| = {rel_resid:.4f}')
    return model, C.cpu().numpy(), S.cpu().numpy(), loss_list


# =====================================================================
# Data loading (.dspkd)
# =====================================================================
def mstr(ds):
    a = np.asarray(ds).ravel()
    return "".join(chr(int(v)) for v in a if v != 0)


def sca(ds):
    return float(np.asarray(ds).ravel()[0])


def load_cube(file_path, subtract_dark=False, apply_correction=False):
    with h5py.File(file_path, "r") as f:
        data   = np.asarray(f["data"], dtype=np.float64)
        wl     = np.asarray(f["header/channelWavelengths"]).ravel()
        corr   = np.asarray(f["header/correctionFactor"]).ravel()
        dark_S = np.asarray(f["header/darkPureSpectra"])
        dark_C = np.asarray(f["header/darkConcentrations"])
        nRows  = int(round(sca(f["header/nRows"])))
        nCols  = int(round(sca(f["header/nCols"])))
        nWav   = int(round(sca(f["header/nWavelengths"])))
        preproc = mstr(f["header/preprocessType"])
    print(f'preprocessType: {preproc} | grid {nRows}x{nCols} | {nWav} channels '
          f'({wl.min():.1f}-{wl.max():.1f} nm)')
    D = data.copy()
    if subtract_dark:
        D = D - dark_C @ dark_S
    if apply_correction:
        D = D * corr[None, :]
    D = np.clip(D, 0.0, None)
    cube = D.reshape(nRows, nCols, nWav, order="F")   # undo MATLAB column-major
    return cube, wl


# =====================================================================
# Plotting
# =====================================================================
def plot_results(C, S, wl, nRows, nCols, S_init=None, names=None, save_path=None):
    K = S.shape[0]
    names = names or [f'comp {k + 1}' for k in range(K)]
    maps = C.reshape(nRows, nCols, K)      # C-order, same as D built from the cube
    fig, axes = plt.subplots(2, K, figsize=(3.2 * K, 6), squeeze=False)
    for k in range(K):
        im = axes[0, k].imshow(maps[:, :, k], cmap='inferno')
        axes[0, k].set_title(names[k]); axes[0, k].axis('off')
        fig.colorbar(im, ax=axes[0, k], fraction=0.046, pad=0.04)
        if S_init is not None:
            axes[1, k].plot(wl, S_init[k], lw=1.0, ls='--', color='grey', label='NMF start')
        axes[1, k].plot(wl, S[k], lw=1.3, label='learned')
        axes[1, k].set_xlabel('wavelength (nm)')
    axes[1, 0].set_ylabel('intensity (unit-norm)')
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle('BiXT unmixing: abundance maps (top), spectra (bottom)')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


# =====================================================================
# Main
# =====================================================================
def main():
    file_path    = "./data/Plant1_L1_13.dspkd"
    N_COMPONENTS = 4

    cube, wl = load_cube(file_path, subtract_dark=False, apply_correction=False)
    nRows, nCols, nWav = cube.shape
    D = cube.reshape(nRows * nCols, nWav).astype('float32')   # (N, C), C-order

    scale = D.max()                  # global scale only -> stays non-negative
    D_n = D / scale

    S_init = nmf_init(D_n, N_COMPONENTS)     # NMF start, then learned by the network

    model, C, S, loss_list = train_unmix(
        D_n, N_COMPONENTS, S_init,
        dim=128, depth=4, n_heads=4,
        epochs=3000, lr=1e-3,
        w_smooth=1e-3, w_sparse=0.0, w_unit=1e-2,
        poisson=True,
    )

    C = C * scale                    # abundances back in original intensity units

    np.save('bixt_C.npy', C)         # (N, K)
    np.save('bixt_S.npy', S)         # (K, C)
    np.savetxt('bixt_spectra.csv', np.column_stack([wl, S.T]), delimiter=',',
               header='wavelength_nm,' + ','.join(f'comp{k + 1}' for k in range(N_COMPONENTS)),
               comments='')
    np.savetxt('bixt_loss.txt', loss_list)

    plot_results(C, S, wl, nRows, nCols, S_init=S_init, save_path='bixt_unmix.png')


if __name__ == '__main__':
    main()