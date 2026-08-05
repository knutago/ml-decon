"""Interactive viewer for the M31 FITS images under storage/M31/.

Run with:  uv run python dataset/data_viewer.py

A radio-button list on the left selects a file; the centre shows the image and a
histogram of pixel values, with dtype / min / max / resolution reported above the
image. The right panel lists the keyboard shortcuts (with [*] marking active
flags) and a sequential-colormap selector.

Keys:
  up / down  select file
  l          toggle log stretch
  r          toggle rescale to [0, 1] (min-max display range)
  q          quit
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from astropy.io import fits
from astropy.visualization import ImageNormalize

SEQUENTIAL_CMAPS = ["inferno", "viridis", "plasma", "magma", "cividis", "gray", "hot", "bone"]


def find_project_root(marker="pyproject.toml"):
    for directory in [Path.cwd(), *Path.cwd().parents]:
        if (directory / marker).exists():
            return directory
    raise FileNotFoundError(f"could not locate {marker} above {Path.cwd()}")


def list_fits_files(data_dir):
    # Skip macOS AppleDouble sidecars (._*.fits) created when copying onto a non-HFS drive.
    return sorted(p for p in data_dir.glob("*.fits") if not p.name.startswith("."))


def load_fits(path):
    """Read a single-HDU FITS image as a native-endian float32 array."""
    with fits.open(path) as hdul:
        data = hdul[0].data
    # FITS stores big-endian; cast to native float32 for downstream numpy/torch use.
    return np.ascontiguousarray(data, dtype=np.float32)


def main():
    data_dir = find_project_root() / "storage" / "M31"
    fits_files = list_fits_files(data_dir)
    if not fits_files:
        raise FileNotFoundError(f"no FITS files under {data_dir}")
    files_by_name = {path.name: path for path in fits_files}
    names = [path.name for path in fits_files]

    state = {"name": names[0], "cmap": SEQUENTIAL_CMAPS[0], "log": False, "rescale": False}

    fig = plt.figure(figsize=(15, 8))
    ax_radio = fig.add_axes([0.02, 0.05, 0.16, 0.9])
    ax_image = fig.add_axes([0.24, 0.40, 0.45, 0.52])
    ax_hist = fig.add_axes([0.24, 0.07, 0.45, 0.22])
    ax_hint = fig.add_axes([0.73, 0.55, 0.25, 0.40])
    ax_cmap = fig.add_axes([0.73, 0.07, 0.16, 0.42])
    ax_hint.axis("off")

    radio_files = RadioButtons(ax_radio, names)
    for label in radio_files.labels:
        label.set_fontsize(8)
    radio_cmap = RadioButtons(ax_cmap, SEQUENTIAL_CMAPS)
    ax_cmap.set_title("colormap", fontsize=10)

    hint_text = ax_hint.text(
        0.0, 1.0, "", transform=ax_hint.transAxes, va="top", family="monospace", fontsize=10
    )

    def hint_string():
        log_mark = "*" if state["log"] else " "
        rescale_mark = "*" if state["rescale"] else " "
        return (
            "keys\n"
            "--------------------\n"
            "up/down  select file\n"
            f"l        log stretch   [{log_mark}]\n"
            f"r        rescale 0-1   [{rescale_mark}]\n"
            "q        quit"
        )

    def render():
        path = files_by_name[state["name"]]
        data = load_fits(path)
        finite = data[np.isfinite(data)]

        # Transform order is fixed: log first, then rescale to [0, 1] (when both on).
        display = data.copy()
        if state["log"]:
            # log10 of values shifted so the minimum maps to 0; +1 avoids log(0).
            # Monotonic, tolerates zeros/negatives. NaNs pass through.
            display = np.log10(display - finite.min() + 1.0)
        if state["rescale"]:
            lo, hi = np.nanmin(display), np.nanmax(display)
            display = (display - lo) / (hi - lo) if hi > lo else np.zeros_like(display)

        # Colormap spans the full transformed data range.
        finite_display = display[np.isfinite(display)]
        norm = ImageNormalize(vmin=finite_display.min(), vmax=finite_display.max())

        ax_image.clear()
        ax_image.imshow(display, origin="lower", cmap=state["cmap"], norm=norm)
        ax_image.set_xticks([])
        ax_image.set_yticks([])
        height, width = data.shape
        ax_image.set_title(
            f"{state['name']}\n"
            f"dtype={data.dtype}  resolution={width}x{height}\n"
            f"min={finite.min():.4g}  max={finite.max():.4g}",
            fontsize=10,
        )

        ax_hist.clear()
        ax_hist.hist(display[np.isfinite(display)].ravel(), bins=200, color="steelblue")
        ax_hist.set_yscale("log")
        ax_hist.set_xlabel("pixel value")
        ax_hist.set_ylabel("count (log)")

        hint_text.set_text(hint_string())
        fig.canvas.draw_idle()

    def on_file(name):
        state["name"] = name
        render()

    def on_cmap(name):
        state["cmap"] = name
        render()

    def on_key(event):
        if event.key == "q":
            plt.close(fig)
            return
        if event.key in ("up", "down"):
            current = names.index(state["name"])
            step = -1 if event.key == "up" else 1
            new = min(max(current + step, 0), len(names) - 1)
            if new != current:
                radio_files.set_active(new)  # fires on_file -> render
            return
        if event.key == "l":
            state["log"] = not state["log"]
            render()
            return
        if event.key == "r":
            state["rescale"] = not state["rescale"]
            render()

    radio_files.on_clicked(on_file)
    radio_cmap.on_clicked(on_cmap)
    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.show()


if __name__ == "__main__":
    main()
