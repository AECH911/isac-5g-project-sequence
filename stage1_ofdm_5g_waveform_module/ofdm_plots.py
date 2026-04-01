import matplotlib.pyplot as plt
import numpy as np


def _get_plot_bandwidth_info(data):
    """
    Use actual active subcarriers to determine occupied bandwidth for display.
    """
    active_bins = data["active_bins"]
    subcarrier_spacing = data["subcarrier_spacing"]

    occupied_bw_hz = len(active_bins) * subcarrier_spacing
    occupied_bw_khz = occupied_bw_hz / 1e3

    rb_left_khz = -occupied_bw_khz / 2
    rb_right_khz = occupied_bw_khz / 2

    margin_factor = 0.5
    plot_half_span_khz = (occupied_bw_khz / 2) * (1 + margin_factor)

    return {
        "occupied_bw_hz": occupied_bw_hz,
        "occupied_bw_khz": occupied_bw_khz,
        "rb_left_khz": rb_left_khz,
        "rb_right_khz": rb_right_khz,
        "plot_half_span_khz": plot_half_span_khz,
    }


def plot_subcarriers_time(data, output_folder, save_figure):
    t = data["t"]
    active_bins = data["active_bins"]
    subcarrier_waves = data["subcarrier_waves"]

    title = "Individual OFDM Subcarriers in Time Real Part"
    fig = plt.figure(figsize=(12, 6))

    for k, wave in zip(active_bins, subcarrier_waves):
        plt.plot(t * 1e6, np.real(wave), label=f"k={k}")

    plt.title(title)
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_figure(fig, output_folder, title)


def plot_tx_symbol_time(data, output_folder, save_figure):
    t = data["t"]
    tx_symbol = data["tx_symbol"]

    title = "Summed OFDM Symbol in Time"
    fig = plt.figure(figsize=(12, 4))

    plt.plot(t * 1e6, np.real(tx_symbol), label="Real")
    plt.plot(t * 1e6, np.imag(tx_symbol), label="Imag", alpha=0.8)

    plt.title(title)
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_figure(fig, output_folder, title)


def plot_subcarrier_spectra(data, output_folder, save_figure):
    active_bins = data["active_bins"]
    freq_axis = data["freq_axis"]
    subcarrier_spectra = data["subcarrier_spectra"]

    bw = _get_plot_bandwidth_info(data)

    title = "Individual Subcarrier Spectra"
    fig = plt.figure(figsize=(12, 6))

    for k, W in zip(active_bins, subcarrier_spectra):
        mag = np.abs(W)
        mag /= np.max(mag) + 1e-12
        plt.plot(freq_axis / 1e3, mag, label=f"k={k}")

    plt.axvspan(
        bw["rb_left_khz"],
        bw["rb_right_khz"],
        alpha=0.15,
        label="Occupied bandwidth",
    )
    plt.xlim(-bw["plot_half_span_khz"], bw["plot_half_span_khz"])

    plt.title(title)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Normalized Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_figure(fig, output_folder, title)


def plot_combined_spectrum(data, output_folder, save_figure):
    freq_axis = data["freq_axis"]
    TX = data["TX"]

    bw = _get_plot_bandwidth_info(data)

    title = "Combined OFDM Symbol Spectrum"
    fig = plt.figure(figsize=(12, 5))

    mag_tx = np.abs(TX)
    mag_tx /= np.max(mag_tx) + 1e-12
    plt.plot(freq_axis / 1e3, mag_tx)

    plt.axvspan(
        bw["rb_left_khz"],
        bw["rb_right_khz"],
        alpha=0.15,
        label="Occupied bandwidth",
    )
    plt.xlim(-bw["plot_half_span_khz"], bw["plot_half_span_khz"])

    plt.title(title)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Normalized Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_figure(fig, output_folder, title)


def plot_frequency_bins(data, output_folder, save_figure):
    freq_bins = data["freq_bins"]

    title = "Discrete Frequency Domain Bin Values"
    fig = plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.stem(np.abs(freq_bins), basefmt=" ")
    plt.title(title)
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.stem(np.real(freq_bins), basefmt=" ")
    plt.ylabel("Real")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.stem(np.imag(freq_bins), basefmt=" ")
    plt.xlabel("FFT Bin Index")
    plt.ylabel("Imag")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_folder, title)


def print_orthogonality_check(data):
    active_bins = data["active_bins"]
    subcarrier_waves = data["subcarrier_waves"]

    print("=== Orthogonality Check (Inner Products Over One OFDM Symbol) ===")
    for i, ki in enumerate(active_bins):
        for j, kj in enumerate(active_bins):
            val = np.vdot(subcarrier_waves[i], subcarrier_waves[j])
            if i == j:
                print(f"<k={ki}, k={kj}> = {val:.4f}")
            else:
                print(f"<k={ki}, k={kj}> = {val:.4e}")


def plot_all_ofdm_views(data, output_folder, save_figure, show_plots=True):
    plot_subcarriers_time(data, output_folder, save_figure)
    plot_tx_symbol_time(data, output_folder, save_figure)
    plot_subcarrier_spectra(data, output_folder, save_figure)
    plot_combined_spectrum(data, output_folder, save_figure)
    plot_frequency_bins(data, output_folder, save_figure)
    print_orthogonality_check(data)

    if show_plots:
        plt.show()