import numpy as np
import matplotlib.pyplot as plt


def qpsk_mod(bits):
    """
    Map bits to QPSK symbols with unit average power.
    Input: bits shape = (N, 2)
    """
    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j,
    }
    syms = np.array([mapping[tuple(b)] for b in bits], dtype=complex)
    syms /= np.sqrt(2)  # normalize average power to 1
    return syms


def qam16_mod(bits):
    """
    Map bits to 16QAM symbols with unit average power.
    Gray-like mapping.
    Input: bits shape = (N, 4)
    """
    def map_2bits_to_level(b1, b2):
        # Gray-coded levels: 00->-3, 01->-1, 11->+1, 10->+3
        if (b1, b2) == (0, 0):
            return -3
        elif (b1, b2) == (0, 1):
            return -1
        elif (b1, b2) == (1, 1):
            return 1
        elif (b1, b2) == (1, 0):
            return 3
        raise ValueError("Invalid bit pair")

    syms = []
    for b in bits:
        i = map_2bits_to_level(b[0], b[1])
        q = map_2bits_to_level(b[2], b[3])
        syms.append(i + 1j * q)

    syms = np.array(syms, dtype=complex)
    syms /= np.sqrt(10)  # normalize average power to 1
    return syms


def generate_random_symbols(num_symbols, modulation):
    """
    Generate random QPSK or 16QAM symbols.
    """
    modulation = modulation.upper()
    if modulation == "QPSK":
        bits = np.random.randint(0, 2, size=(num_symbols, 2))
        return qpsk_mod(bits)
    elif modulation == "16QAM":
        bits = np.random.randint(0, 2, size=(num_symbols, 4))
        return qam16_mod(bits)
    else:
        raise ValueError("Unsupported modulation. Use 'QPSK' or '16QAM'.")


def build_resource_grid(
    n_fft=256,
    subcarrier_spacing=15e3,
    num_ofdm_symbols=14,
    modulation="QPSK",
    num_active_subcarriers=180,
    cp_len=18,
    pilot_spacing_freq=12,
    pilot_spacing_time=4,
    pilot_value=1 + 1j
):
    """
    Build a simple OFDM resource grid centered around DC.

    Returns:
        resource_grid: shape (n_fft, num_ofdm_symbols)
        data_mask:     True where data symbols exist
        pilot_mask:    True where pilot symbols exist
        fs:            sample rate
        cp_len:        cyclic prefix length
    """
    if num_active_subcarriers >= n_fft:
        raise ValueError("num_active_subcarriers must be less than n_fft")

    fs = n_fft * subcarrier_spacing

    resource_grid = np.zeros((n_fft, num_ofdm_symbols), dtype=complex)
    data_mask = np.zeros_like(resource_grid, dtype=bool)
    pilot_mask = np.zeros_like(resource_grid, dtype=bool)

    # Center active subcarriers around DC, excluding DC
    half = num_active_subcarriers // 2
    neg_bins = np.arange(n_fft // 2 - half, n_fft // 2)
    pos_bins = np.arange(n_fft // 2 + 1, n_fft // 2 + 1 + half)

    active_bins = np.concatenate([neg_bins, pos_bins])

    # Define pilot locations
    for sym_idx in range(num_ofdm_symbols):
        for k, sc in enumerate(active_bins):
            # regular pilot pattern
            if (k % pilot_spacing_freq == 0) and (sym_idx % pilot_spacing_time == 0):
                resource_grid[sc, sym_idx] = pilot_value / np.sqrt(2)
                pilot_mask[sc, sym_idx] = True
            else:
                data_mask[sc, sym_idx] = True

    # Fill data
    num_data_re = np.sum(data_mask)
    data_symbols = generate_random_symbols(num_data_re, modulation)
    resource_grid[data_mask] = data_symbols

    return resource_grid, data_mask, pilot_mask, fs, cp_len


def ofdm_modulate(resource_grid, cp_len):
    """
    OFDM modulation:
    1. IFFT shift to move DC to index 0 for IFFT input
    2. IFFT per OFDM symbol
    3. Add cyclic prefix
    4. Serialize
    """
    n_fft, num_symbols = resource_grid.shape

    tx_symbols_time = []
    for sym_idx in range(num_symbols):
        freq_symbol = resource_grid[:, sym_idx]
        ifft_in = np.fft.ifftshift(freq_symbol)
        time_symbol = np.fft.ifft(ifft_in, n=n_fft)

        cp = time_symbol[-cp_len:]
        tx_with_cp = np.concatenate([cp, time_symbol])
        tx_symbols_time.append(tx_with_cp)

    tx_waveform = np.concatenate(tx_symbols_time)
    return tx_waveform


def compute_spectrum(x, fs):
    """
    Compute centered power spectrum in dB.
    """
    n = len(x)
    window = np.hanning(n)
    xw = x * window
    X = np.fft.fftshift(np.fft.fft(xw, n=n))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1/fs))
    psd = 20 * np.log10(np.abs(X) / np.max(np.abs(X)) + 1e-12)
    return freqs, psd


def plot_results(resource_grid, data_mask, pilot_mask, tx_waveform, fs, title_suffix=""):
    """
    Create:
    - resource-grid visualization
    - constellation plot
    - time-domain waveform plot
    - spectrum plot
    """
    # Extract data symbols only for constellation
    data_symbols = resource_grid[data_mask]

    fig1 = plt.figure(figsize=(10, 6))
    plt.imshow(
        np.abs(resource_grid),
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )
    plt.colorbar(label="Magnitude")
    plt.title(f"Resource Grid Magnitude {title_suffix}")
    plt.xlabel("OFDM Symbol Index")
    plt.ylabel("Subcarrier Index")
    plt.tight_layout()

    fig2 = plt.figure(figsize=(6, 6))
    plt.scatter(data_symbols.real, data_symbols.imag, s=10, alpha=0.7)
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.title(f"Constellation Plot {title_suffix}")
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.axis("equal")
    plt.tight_layout()

    fig3 = plt.figure(figsize=(10, 4))
    plt.plot(np.real(tx_waveform[:1000]), label="Real")
    plt.plot(np.imag(tx_waveform[:1000]), label="Imag", alpha=0.8)
    plt.title(f"Time-Domain Waveform (First 1000 Samples) {title_suffix}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    freqs, psd = compute_spectrum(tx_waveform, fs)
    fig4 = plt.figure(figsize=(10, 4))
    plt.plot(freqs / 1e6, psd)
    plt.title(f"Spectrum Plot {title_suffix}")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Normalized Magnitude (dB)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


def main():
    # ==============================
    # User-configurable parameters
    # ==============================
    n_fft = 256
    subcarrier_spacing = 30e3      # Hz
    num_ofdm_symbols = 14
    modulation = "16QAM"           # "QPSK" or "16QAM"
    num_active_subcarriers = 180
    cp_len = 18
    pilot_spacing_freq = 12
    pilot_spacing_time = 4

    # Build grid
    resource_grid, data_mask, pilot_mask, fs, cp_len = build_resource_grid(
        n_fft=n_fft,
        subcarrier_spacing=subcarrier_spacing,
        num_ofdm_symbols=num_ofdm_symbols,
        modulation=modulation,
        num_active_subcarriers=num_active_subcarriers,
        cp_len=cp_len,
        pilot_spacing_freq=pilot_spacing_freq,
        pilot_spacing_time=pilot_spacing_time,
        pilot_value=1 + 1j
    )

    # OFDM modulation
    tx_waveform = ofdm_modulate(resource_grid, cp_len)

    # Print useful info
    print("========== OFDM / 5G-like Waveform Generator ==========")
    print(f"FFT size                : {n_fft}")
    print(f"Subcarrier spacing      : {subcarrier_spacing/1e3:.1f} kHz")
    print(f"Sample rate             : {fs/1e6:.3f} MHz")
    print(f"OFDM symbols            : {num_ofdm_symbols}")
    print(f"Modulation              : {modulation}")
    print(f"Active subcarriers      : {num_active_subcarriers}")
    print(f"Cyclic prefix length    : {cp_len} samples")
    print(f"Pilot spacing (freq)    : every {pilot_spacing_freq} active SCs")
    print(f"Pilot spacing (time)    : every {pilot_spacing_time} OFDM symbols")
    print(f"Waveform length         : {len(tx_waveform)} samples")
    print(f"Occupied bandwidth ~    : {num_active_subcarriers * subcarrier_spacing / 1e6:.3f} MHz")
    print("=======================================================")

    # Plots
    plot_results(
        resource_grid=resource_grid,
        data_mask=data_mask,
        pilot_mask=pilot_mask,
        tx_waveform=tx_waveform,
        fs=fs,
        title_suffix=f"({modulation}, Δf={subcarrier_spacing/1e3:.0f}kHz, Nfft={n_fft})"
    )


if __name__ == "__main__":
    main()