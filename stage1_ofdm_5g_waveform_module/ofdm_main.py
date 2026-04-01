import re
from dataclasses import dataclass
from pathlib import Path

from bit_modulation import get_symbols
from nr_resources import NRResourceConfig, get_active_bins
from ofdm_math import generate_ofdm_symbol_data
from ofdm_plots import plot_all_ofdm_views


@dataclass
class OFDMConfig: # Configuration for the OFDM orthogonality lesson
    n_fft: int = 64
    modulation: str = "QPSK"
    subcarrier_spacing: float = 15e3
    num_rbs: int = 2

# ============================================================
# FILE / SAVE HELPERS
# ============================================================

def get_output_folder():
    """
    Save relative to the location of this .py file.
    Falls back to current working directory if __file__ is unavailable.
    """
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()

    output_folder = base_dir / "images" / "ofdm_images"
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to: {output_folder}")
    return output_folder


def slugify_title(title: str) -> str:
    """
    Convert a plot title into a safe filename.
    Example:
    'Lesson Demo: Combined OFDM Symbol Spectrum'
    -> 'lesson_demo_combined_ofdm_symbol_spectrum.png'
    """
    title = title.lower().strip()
    title = re.sub(r"[^\w\s-]", "", title)   # remove punctuation
    title = re.sub(r"[-\s]+", "_", title)    # spaces/dashes -> underscores
    return f"{title}.png"


def save_figure(fig, output_folder, title: str):
    """
    Save figure using its title as the filename.
    """
    filename = slugify_title(title)
    file_path = output_folder / filename
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {file_path}")


# ============================================================
# LESSON RUNNERS
# ============================================================
def run_orthogonality_lesson(cfg: OFDMConfig):
    output_folder = get_output_folder()
    nr_cfg = NRResourceConfig(
        subcarrier_spacing_hz=cfg.subcarrier_spacing,
        num_rbs=cfg.num_rbs,
    )

    active_bins = get_active_bins(cfg.num_rbs)
    symbols = get_symbols(cfg.modulation, len(active_bins))

    print(f"Subcarrier spacing: {cfg.subcarrier_spacing / 1e3:.0f} kHz")
    print(f"Number of RBs: {cfg.num_rbs}")
    print(f"Subcarriers per RB: {nr_cfg.subcarriers_per_rb}")
    print(f"Total active subcarriers: {nr_cfg.total_subcarriers}")
    print(f"RB bandwidth: {nr_cfg.rb_bandwidth_hz / 1e3:.2f} kHz")
    print(f"Total occupied bandwidth: {nr_cfg.total_bandwidth_hz / 1e3:.2f} kHz")
    data = generate_ofdm_symbol_data(
        n_fft=cfg.n_fft,
        subcarrier_spacing=cfg.subcarrier_spacing,
        active_bins=list(active_bins),
        symbols=list(symbols),
    )

    plot_all_ofdm_views(
        data=data,
        output_folder=output_folder,
        save_figure=save_figure,
        show_plots=True,
    )

# ============================================================
# MAIN
# ============================================================

def main():
    cfg = OFDMConfig(
        n_fft= 1028,
        modulation="256QAM",
        num_rbs=1,
        subcarrier_spacing=15e3
        )
    run_orthogonality_lesson(cfg)

if __name__ == "__main__":
    main()