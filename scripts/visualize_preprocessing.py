import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- helpers -----------------

def load_nii(path):
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.affine

def hu_window_to_01(x, lo=-100, hi=1000):
    x = np.clip(x, lo, hi)
    return (x - lo) / float(hi - lo)

def choose_middle_slices(vol):
    return (vol.shape[2] // 2, vol.shape[1] // 2, vol.shape[0] // 2)

# ----------------- main viz -----------------

def visualize_pair(raw_path, proc_path,
                   hu_min=-100, hu_max=1000,
                   title_suffix="",
                   save_png=None,
                   HIST_MODE="overlay01",    # "split" | "overlay01" | "dualaxis"
                   LOG_SCALE=False):
    raw, _  = load_nii(raw_path)
    proc, _ = load_nii(proc_path)

    raw_disp  = hu_window_to_01(raw, hu_min, hu_max)
    proc_disp = np.clip(proc, 0, 1)

    z_r, y_r, x_r = choose_middle_slices(raw)
    z_p, y_p, x_p = choose_middle_slices(proc_disp)

    planes = {
        "Axial (Z)":   (raw[:, :, z_r],  proc_disp[:, :, z_p]),
        "Coronal (Y)": (raw[:, y_r, :],  proc_disp[:, y_p, :]),
        "Sagittal (X)":(raw[x_r, :, :],  proc_disp[x_p, :, :]),
    }

    # ------- Figure 1: slice comparisons -------
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))
    fig.suptitle(f"Raw vs Preprocessed {title_suffix}", fontsize=14, y=0.98)
    for row, (plane_name, (r, p)) in enumerate(planes.items()):
        ax_r = axes[row, 0]; ax_p = axes[row, 1]
        ax_r.imshow(np.rot90(r), cmap="gray", vmin=0.0, vmax=1.0)
        ax_r.set_title(f"{plane_name} — RAW ")
        ax_r.axis("off")
        ax_p.imshow(np.rot90(p), cmap="gray", vmin=0.0, vmax=1.0)
        ax_p.set_title(f"{plane_name} — PREPROCESSED (0–1)")
        ax_p.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ------- Figure 2: histograms -------
    if HIST_MODE == "split":
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        ax1.hist(raw.flatten(), bins=300, density=True, alpha=0.8)
        ax1.set_title(f"RAW (HU) {title_suffix}")
        ax1.set_xlabel("HU"); ax1.set_ylabel("Density")
        ax1.set_xlim(hu_min, hu_max)
        if LOG_SCALE: ax1.set_yscale("log")

        ax2.hist(proc.flatten(), bins=300, density=True, alpha=0.8)
        ax2.set_title(f"PREPROCESSED (0–1) {title_suffix}")
        ax2.set_xlabel("Normalized Intensity"); ax2.set_ylabel("Density")
        ax2.set_xlim(0, 1)
        if LOG_SCALE: ax2.set_yscale("log")
        plt.tight_layout()

    elif HIST_MODE == "overlay01":
        # Normalize raw to [0,1] first so both share the same x-axis
        raw01 = hu_window_to_01(raw, hu_min, hu_max)
        fig2, ax = plt.subplots(figsize=(7, 4))
        ax.hist(raw01.flatten(), bins=300, density=True, alpha=0.6, label="RAW→[0,1]")
        ax.hist(proc.flatten(),  bins=300, density=True, alpha=0.6, label="PREPROC (0–1)")
        ax.set_title(f"Intensity Distributions (Both in 0–1) {title_suffix}")
        ax.set_xlabel("Normalized Intensity (0–1)"); ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        if LOG_SCALE: ax.set_yscale("log")
        ax.legend()
        plt.tight_layout()

    else:  # "dualaxis"
        fig2, ax1 = plt.subplots(figsize=(7, 4))
        ax1.hist(raw.flatten(), bins=300, alpha=0.6, label="RAW (HU)", density=True)
        ax1.set_xlabel("Raw HU values"); ax1.set_ylabel("Density (Raw)")
        ax1.set_xlim(hu_min, hu_max)
        if LOG_SCALE: ax1.set_yscale("log")

        ax2 = ax1.twiny()
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Preprocessed (0–1) scale")

        ax3 = ax1.twinx()
        ax3.hist(proc.flatten(), bins=300, alpha=0.5, color="tab:orange", label="PREPROC (0–1)", density=True)
        ax3.set_ylabel("Density (Preprocessed)")
        if LOG_SCALE: ax3.set_yscale("log")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax1.set_title(f"Intensity Distributions {title_suffix}")
        plt.tight_layout()

    # ------- Save or show -------
    if save_png:
        out1 = Path(save_png)
        out2 = out1.with_name(out1.stem + "_hist" + out1.suffix)
        out1.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out1, dpi=200)
        fig2.savefig(out2, dpi=200)
        plt.close(fig); plt.close(fig2)
        print(f"Saved: {out1}"); print(f"Saved: {out2}")
    else:
        plt.show()

def visualize_first_N(input_dir, processed_dir, N=20,
                      hu_min=-100, hu_max=1000,
                      save_dir=None,
                      HIST_MODE="split",
                      LOG_SCALE=False):
    input_dir = Path(input_dir)
    processed_dir = Path(processed_dir)
    files = sorted(list(input_dir.glob("*.nii")) + list(input_dir.glob("*.nii.gz")))
    if not files:
        print(f"No NIfTI files in: {input_dir}")
        return
    count = 0
    for raw_path in files:
        proc_path = processed_dir / raw_path.name
        if not proc_path.exists():
            print(f"Missing preprocessed file for: {raw_path.name}")
            continue
        title_suffix = f"({raw_path.name})"
        save_png = None if save_dir is None else (Path(save_dir) / f"{raw_path.stem}_compare.png")
        print(f"Visualizing: {raw_path.name}")
        visualize_pair(raw_path, proc_path,
                       hu_min=hu_min, hu_max=hu_max,
                       title_suffix=title_suffix, save_png=save_png,
                       HIST_MODE=HIST_MODE, LOG_SCALE=LOG_SCALE)
        count += 1
        if count >= N:
            break

if __name__ == "__main__":
    input_dir    = r"data\imageCAS_data\201-400"
    processed_dir= r"data\processed\201-400"
    N            = 20
    save_dir     = r"outputs\viz_compare"

    HIST_MODE = "overlay01"   # "split" | "overlay01" | "dualaxis"
    LOG_SCALE = False

    visualize_first_N(input_dir, processed_dir, N=N,
                      hu_min=-100, hu_max=1000,
                      save_dir=save_dir,
                      HIST_MODE=HIST_MODE,
                      LOG_SCALE=LOG_SCALE)
