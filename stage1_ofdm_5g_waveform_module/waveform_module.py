import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ============================================================
# CONFIG
# ============================================================

@dataclass
class OFDMConfig:
    n_fft: int = 256
    subcarrier_spacing: float = 30e3
    num_ofdm_symbols: int = 14
    modulation: str = "16QAM"       # "QPSK" or "16QAM"
    num_active_subcarriers: int = 180
    cp_len: int = 18
    pilot_spacing_freq: int = 12
    pilot_spacing_time: int = 4
    pilot_value: complex = 1 + 1j


# ============================================================
# MODULATION
# ============================================================

def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    """Map bits of shape (N, 2) to QPSK symbols with unit average power."""
    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j,
    }
    syms = np.array([mapping[tuple(b)] for b in bits], dtype=complex)
    return syms / np.sqrt(2)


def qam16_mod(bits: np.ndarray) -> np.ndarray:
    """Map bits of shape (N, 4) to 16QAM symbols with unit average power."""

    def map_2bits_to_level(b1: int, b2: int) -> int:
        if (b1, b2) == (0, 0):
            return -3
        if (b1, b2) == (0, 1):
            return -1
        if (b1, b2) == (1, 1):
            return 1
        if (b1, b2) == (1, 0):
            return 3
        raise ValueError("Invalid bit pair")

    syms = []
    for b in bits:
        i = map_2bits_to_level(b[0], b[1])
        q = map_2bits_to_level(b[2], b[3])
        syms.append(i + 1j * q)

    return np.array(syms, dtype=complex) / np.sqrt(10)


def generate_random_symbols(num_symbols: int, modulation: str) -> np.ndarray:
    """Generate random modulation symbols."""
    modulation = modulation.upper()

    if modulation == "QPSK":
        bits = np.random.randint(0, 2, size=(num_symbols, 2))
        return qpsk_mod(bits)

    if modulation == "16QAM":
        bits = np.random.randint(0, 2, size=(num_symbols, 4))
        return qam16_mod(bits)

    raise ValueError("Unsupported modulation. Use 'QPSK' or '16QAM'.")


# ============================================================
# RESOURCE GRID / OFDM CORE
# ============================================================

def get_sample_rate(cfg: OFDMConfig) -> float:
    """Sampling rate implied by FFT size and subcarrier spacing."""
    return cfg.n_fft * cfg.subcarrier_spacing


def get_active_bins(cfg: OFDMConfig) -> np.ndarray:
    """
    Return active FFT bin indices centered around DC, excluding the DC bin.
    Uses array indexing in fftshifted form.
    """
    if cfg.num_active_subcarriers >= cfg.n_fft:
        raise ValueError("num_active_subcarriers must be less than n_fft")

    half = cfg.num_active_subcarriers // 2
    neg_bins = np.arange(cfg.n_fft // 2 - half, cfg.n_fft // 2)
    pos_bins = np.arange(cfg.n_fft // 2 + 1, cfg.n_fft // 2 + 1 + half)
    return np.concatenate([neg_bins, pos_bins])


def build_resource_grid(cfg: OFDMConfig):
    """
    Build the frequency-domain resource grid.

    Returns
    -------
    resource_grid : np.ndarray
        Shape (n_fft, num_ofdm_symbols)
    data_mask : np.ndarray
        True where data REs exist
    pilot_mask : np.ndarray
        True where pilot REs exist
    """
    resource_grid = np.zeros((cfg.n_fft, cfg.num_ofdm_symbols), dtype=complex)
    data_mask = np.zeros_like(resource_grid, dtype=bool)
    pilot_mask = np.zeros_like(resource_grid, dtype=bool)

    active_bins = get_active_bins(cfg)

    for sym_idx in range(cfg.num_ofdm_symbols):
        for k, sc in enumerate(active_bins):
            is_pilot = (
                (k % cfg.pilot_spacing_freq == 0)
                and (sym_idx % cfg.pilot_spacing_time == 0)
            )

            if is_pilot:
                resource_grid[sc, sym_idx] = cfg.pilot_value / np.sqrt(2)
                pilot_mask[sc, sym_idx] = True
            else:
                data_mask[sc, sym_idx] = True

    num_data_re = np.sum(data_mask)
    resource_grid[data_mask] = generate_random_symbols(num_data_re, cfg.modulation)

    return resource_grid, data_mask, pilot_mask


def ofdm_symbol_to_time_domain(freq_symbol: np.ndarray, n_fft: int, cp_len: int):
    """
    Convert one frequency-domain OFDM symbol to time domain and add CP.
    """
    ifft_in = np.fft.ifftshift(freq_symbol)
    time_symbol = np.fft.ifft(ifft_in, n=n_fft)
    cp = time_symbol[-cp_len:]
    time_with_cp = np.concatenate([cp, time_symbol])
    return time_symbol, time_with_cp


def ofdm_modulate(resource_grid: np.ndarray, cp_len: int) -> np.ndarray:
    """
    Convert the full resource grid to a serialized OFDM waveform.
    """
    n_fft, num_symbols = resource_grid.shape
    tx_symbols_time = []

    for sym_idx in range(num_symbols):
        freq_symbol = resource_grid[:, sym_idx]
        _, time_with_cp = ofdm_symbol_to_time_domain(freq_symbol, n_fft, cp_len)
        tx_symbols_time.append(time_with_cp)

    return np.concatenate(tx_symbols_time)


def compute_spectrum(x: np.ndarray, fs: float):
    """
    Compute centered normalized spectrum in dB.
    """
    n = len(x)
    window = np.hanning(n)
    xw = x * window
    X = np.fft.fftshift(np.fft.fft(xw, n=n))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs))
    psd = 20 * np.log10(np.abs(X) / np.max(np.abs(X)) + 1e-12)
    return freqs, psd


# ============================================================
# SUMMARY / INFO
# ============================================================

def print_waveform_summary(cfg: OFDMConfig, fs: float, tx_waveform: np.ndarray):
    occupied_bw = cfg.num_active_subcarriers * cfg.subcarrier_spacing / 1e6

    print("========== OFDM / 5G-like Waveform Generator ==========")
    print(f"FFT size                : {cfg.n_fft}")
    print(f"Subcarrier spacing      : {cfg.subcarrier_spacing / 1e3:.1f} kHz")
    print(f"Sample rate             : {fs / 1e6:.3f} MHz")
    print(f"OFDM symbols            : {cfg.num_ofdm_symbols}")
    print(f"Modulation              : {cfg.modulation}")
    print(f"Active subcarriers      : {cfg.num_active_subcarriers}")
    print(f"Cyclic prefix length    : {cfg.cp_len} samples")
    print(f"Pilot spacing (freq)    : every {cfg.pilot_spacing_freq} active SCs")
    print(f"Pilot spacing (time)    : every {cfg.pilot_spacing_time} OFDM symbols")
    print(f"Waveform length         : {len(tx_waveform)} samples")
    print(f"Occupied bandwidth ~    : {occupied_bw:.3f} MHz")
    print("=======================================================")


def make_title_suffix(cfg: OFDMConfig) -> str:
    return f"({cfg.modulation}, Δf={cfg.subcarrier_spacing / 1e3:.0f}kHz, Nfft={cfg.n_fft})"


# ============================================================
# PLOTTING: MAIN WAVEFORM LESSONS
# ============================================================

def plot_resource_grid_magnitude(resource_grid: np.ndarray, title_suffix: str = ""):
    plt.figure(figsize=(10, 6))
    plt.imshow(
        np.abs(resource_grid),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    plt.colorbar(label="Magnitude")
    plt.title(f"Resource Grid Magnitude {title_suffix}")
    plt.xlabel("OFDM Symbol Index")
    plt.ylabel("Subcarrier Index")
    plt.tight_layout()
    plt.show()


def plot_resource_grid_types(data_mask: np.ndarray, pilot_mask: np.ndarray, title_suffix: str = ""):
    grid_type = np.zeros(data_mask.shape, dtype=int)
    grid_type[data_mask] = 1
    grid_type[pilot_mask] = 2

    plt.figure(figsize=(10, 6))
    plt.imshow(
        grid_type,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    plt.title(f"Resource Grid Type Map {title_suffix}")
    plt.xlabel("OFDM Symbol Index")
    plt.ylabel("Subcarrier Index")

    cbar = plt.colorbar()
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["Null/Unused", "Data", "Pilot"])

    plt.tight_layout()
    plt.show()


def plot_constellation(resource_grid: np.ndarray, data_mask: np.ndarray, title_suffix: str = ""):
    data_symbols = resource_grid[data_mask]

    plt.figure(figsize=(6, 6))
    plt.scatter(data_symbols.real, data_symbols.imag, s=10, alpha=0.7)
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.title(f"Constellation Plot {title_suffix}")
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_time_domain_waveform(tx_waveform: np.ndarray, title_suffix: str = "", num_samples: int = 1000):
    plt.figure(figsize=(10, 4))
    plt.plot(np.real(tx_waveform[:num_samples]), label="Real")
    plt.plot(np.imag(tx_waveform[:num_samples]), label="Imag", alpha=0.8)
    plt.title(f"Time-Domain Waveform (First {num_samples} Samples) {title_suffix}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_waveform_spectrum(tx_waveform: np.ndarray, fs: float, title_suffix: str = ""):
    freqs, psd = compute_spectrum(tx_waveform, fs)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs / 1e6, psd)
    plt.title(f"Spectrum Plot {title_suffix}")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Normalized Magnitude (dB)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_frequency_domain_symbol(resource_grid: np.ndarray, symbol_index: int = 0, title_suffix: str = ""):
    freq_symbol = resource_grid[:, symbol_index]

    plt.figure(figsize=(10, 8))

    plt.subplot(4, 1, 1)
    plt.stem(np.abs(freq_symbol), basefmt=" ")
    plt.title(f"Frequency-Domain OFDM Symbol {symbol_index} {title_suffix}")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.stem(np.real(freq_symbol), basefmt=" ")
    plt.ylabel("Real")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.stem(np.imag(freq_symbol), basefmt=" ")
    plt.ylabel("Imag")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.stem(np.angle(freq_symbol), basefmt=" ")
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Phase (rad)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_symbol_domain_comparison(resource_grid: np.ndarray, cp_len: int, symbol_index: int = 0, title_suffix: str = ""):
    n_fft, _ = resource_grid.shape
    freq_symbol = resource_grid[:, symbol_index]
    time_symbol, time_with_cp = ofdm_symbol_to_time_domain(freq_symbol, n_fft, cp_len)

    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.stem(np.abs(freq_symbol), basefmt=" ")
    plt.title(f"OFDM Symbol {symbol_index}: Frequency Domain vs Time Domain {title_suffix}")
    plt.ylabel("|X[k]|")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.stem(np.angle(freq_symbol), basefmt=" ")
    plt.ylabel("∠X[k]")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(np.real(time_symbol), label="Real")
    plt.plot(np.imag(time_symbol), label="Imag", alpha=0.8)
    plt.ylabel("x[n]")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(np.real(time_with_cp), label="Real")
    plt.plot(np.imag(time_with_cp), label="Imag", alpha=0.8)
    plt.xlabel("Sample Index")
    plt.ylabel("x_cp[n]")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



# ============================================================
# LESSON RUNNERS
# ============================================================

def run_waveform_generation_lesson(cfg: OFDMConfig):
    """
    Main lesson:
    1. Build the resource grid
    2. Show what is inside the grid
    3. Convert it to time-domain OFDM waveform
    4. Visualize both domains
    """
    fs = get_sample_rate(cfg)

    resource_grid, data_mask, pilot_mask = build_resource_grid(cfg)
    tx_waveform = ofdm_modulate(resource_grid, cfg.cp_len)

    print_waveform_summary(cfg, fs, tx_waveform)
    title_suffix = make_title_suffix(cfg)

    plot_resource_grid_types(data_mask, pilot_mask, title_suffix)
    plot_resource_grid_magnitude(resource_grid, title_suffix)
    plot_constellation(resource_grid, data_mask, title_suffix)
    plot_frequency_domain_symbol(resource_grid, symbol_index=0, title_suffix=title_suffix)
    plot_symbol_domain_comparison(resource_grid, cfg.cp_len, symbol_index=0, title_suffix=title_suffix)
    plot_time_domain_waveform(tx_waveform, title_suffix)
    plot_waveform_spectrum(tx_waveform, fs, title_suffix)




# ============================================================
# MAIN
# ============================================================

def main():
    cfg = OFDMConfig(
        n_fft=256,
        subcarrier_spacing=30e3,
        num_ofdm_symbols=14,
        modulation="16QAM",
        num_active_subcarriers=180,
        cp_len=18,
        pilot_spacing_freq=12,
        pilot_spacing_time=4,
    )

    run_waveform_generation_lesson(cfg)


if __name__ == "__main__":
    main()