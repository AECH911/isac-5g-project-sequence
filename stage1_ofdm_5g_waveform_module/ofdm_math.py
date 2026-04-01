import numpy as np

def generate_ofdm_symbol_data(
    n_fft: int = 64,
    subcarrier_spacing: float = 15e3,
    active_bins=None,
    symbols=None,
    n_spectrum=None,
):
    """
    Generate OFDM data from the sythesis equation.
        $x[n] = \sum_{k \in K} a_k e^{j 2\pi k \Delta f t_n}$
            The discrete-time complex baseband OFDM transmit signal written as a sum of orthogonal subcarriers.
        a_k: symbol placed on subcarrier k
        $e^{j 2\pi k \Delta f t_n}$: the k-th orthogonal complex exponential
        Summation: add all active subcarriers together to form one OFDM symbol
    This equation is the same basic idea as saying OFDM = IFFT of mapped subcarrier symbols

    Useful OFDM symbol duration is set by the subcarrier spacing:
    $f_s = N_FFT * \Delta f$
    $t[n]= n / f_s$ for $n=0,1,...,N_FFT-1$
    Useful symbol durration:
    $T_{u} = N_FFT/f_s = N_FFT/(N_FFT* \Delta f) = 1 / \Delta f$ 
        
    Returns a dictionary containing all waveform/math data needed by the plots.
    """
    # Default to 5 active subcarriers around DC if not provided
    if active_bins is None:
        active_bins = [-2, -1, 0, 1, 2]
    #  Default to all active subcarriers having symbol value 1+0j if not provided
    if symbols is None:
        symbols = [1 + 0j] * len(active_bins)
    # Validate inputs
    if len(active_bins) != len(symbols):
        raise ValueError("active_bins and symbols must have the same length")
    # Default spectrum size for FFT-based spectra. Larger than n_fft for smoother plots.
    if n_spectrum is None:
        n_spectrum = max(8192, 16 * n_fft) # Use a large FFT size for smooth spectra visualization

    # Useful symbol duration
    fs = n_fft * subcarrier_spacing 
    t = np.arange(n_fft) / fs
    # Note n_fft is the number of samples (think discrete time points) they cancelled out see descritption equation above Useful OFDM symbol duration.  

    # Create the time-domain subcarrier waves
    subcarrier_waves = []
    for k, a in zip(active_bins, symbols):
        wave = a * np.exp(1j * 2 * np.pi * k * subcarrier_spacing * t) # individual subcarrier waves
        subcarrier_waves.append(wave)
    # Convert list to array for easier manipulation later
    subcarrier_waves = np.array(subcarrier_waves)

    # Sum them to get the OFDM symbol in time
    tx_symbol = np.sum(subcarrier_waves, axis=0) / np.sqrt(n_fft)

    # Frequency axis for plotting spectra
    freq_axis = np.fft.fftshift(np.fft.fftfreq(n_spectrum, d=1 / fs)) 
    # sets up frequecy axis for something that will have an fft of length n_spectrum, with sample spacing in time $1/f_s$ 
    #  fftshift rearanges the FFT output ordering so the plot has negative frequencies on the left 0 in middile and positive frequencies on the right.
    #   fftshit is only relvant for visualization, the math of OFDM does not require it.
    # fftfreq returns the corresponding frequencies for each FFT bin, fftshift rearranges it so that negative frequencies are on the left and positive on the right.

    # Compute spectra of individual subcarriers
    subcarrier_spectra = []
    for wave in subcarrier_waves:
        W = np.fft.fftshift(np.fft.fft(wave, n=n_spectrum))
        subcarrier_spectra.append(W)
    subcarrier_spectra = np.array(subcarrier_spectra)
    # Computes the FFT of each individual subcarrier waveform.
    #  Each wave is one finite-duration complex sinusoid
    #   Wave only has n_fft samples, n_specturm is used to get a smooth spectrum for visualization, this zero-pads the time-domain signal before the FFT

    # Combined OFDM symbol spectrum
    TX = np.fft.fftshift(np.fft.fft(tx_symbol, n=n_spectrum)) #
    # Computes the FFT of each combinded waveform.
    #  Each wave is one finite-duration complex sinusoid
    #   Wave only has n_fft samples, n_specturm is used to get a smooth spectrum for visualization, this zero-pads the time-domain signal before the FFT

    # Discrete frequency-domain bin values
    freq_bins = np.zeros(n_fft, dtype=complex)
    center = n_fft // 2
    for k, a in zip(active_bins, symbols):
        freq_bins[center + k] = a

    return {
        "n_fft": n_fft,
        "subcarrier_spacing": subcarrier_spacing,
        "active_bins": list(active_bins),
        "symbols": list(symbols),
        "n_spectrum": n_spectrum,
        "fs": fs,
        "t": t,
        "subcarrier_waves": subcarrier_waves,
        "tx_symbol": tx_symbol,
        "freq_axis": freq_axis,
        "subcarrier_spectra": subcarrier_spectra,
        "TX": TX,
        "freq_bins": freq_bins,
    }


def generate_cp_ofdm_symbol_data(
    n_fft: int = 256,
    subcarrier_spacing: float = 15e3,
    active_bins=None,
    symbols=None,
    cp_len: int = 18,
    n_spectrum=None,
):
    """
    Generate one CP-OFDM symbol using the implementation flow:
    frequency-bin mapping -> IFFT -> cyclic prefix addition.

    Returns:
        dict with time-domain and frequency-domain views
    """
    if active_bins is None:
        active_bins = [-2, -1, 0, 1, 2]

    if symbols is None:
        symbols = [1 + 0j] * len(active_bins)

    if len(active_bins) != len(symbols):
        raise ValueError("active_bins and symbols must have the same length")

    if n_spectrum is None:
        n_spectrum = max(8192, 16 * n_fft)

    if cp_len < 0:
        raise ValueError("cp_len must be non-negative")

    # --------------------------------------------------------
    # Sample rate and time axes
    # --------------------------------------------------------
    fs = n_fft * subcarrier_spacing
    ts = 1 / fs

    useful_symbol_duration = n_fft / fs
    cp_duration = cp_len / fs
    total_symbol_duration = (n_fft + cp_len) / fs

    # --------------------------------------------------------
    # Frequency-domain OFDM bin vector
    # Centered indexing for teaching/demo:
    # negative bins left of DC, positive bins right of DC
    # --------------------------------------------------------
    freq_bins_centered = np.zeros(n_fft, dtype=complex)
    center = n_fft // 2

    for k, a in zip(active_bins, symbols):
        idx = center + k
        if idx < 0 or idx >= n_fft:
            raise ValueError(f"Active bin k={k} falls outside the FFT size n_fft={n_fft}")
        freq_bins_centered[idx] = a

    # Convert centered bin layout into NumPy IFFT ordering
    # because np.fft.ifft expects DC at index 0
    freq_bins_ifft = np.fft.ifftshift(freq_bins_centered)

    # --------------------------------------------------------
    # CP-OFDM useful symbol: IFFT of mapped bins
    # --------------------------------------------------------
    useful_symbol = np.fft.ifft(freq_bins_ifft, n=n_fft) * np.sqrt(n_fft)

    # --------------------------------------------------------
    # Cyclic prefix
    # --------------------------------------------------------
    cp = useful_symbol[-cp_len:] if cp_len > 0 else np.array([], dtype=complex)
    tx_with_cp = np.concatenate([cp, useful_symbol])

    # --------------------------------------------------------
    # Time axes
    # --------------------------------------------------------
    t_useful = np.arange(n_fft) * ts
    t_with_cp = np.arange(n_fft + cp_len) * ts

    # --------------------------------------------------------
    # Spectra for plotting
    # --------------------------------------------------------
    freq_axis = np.fft.fftshift(np.fft.fftfreq(n_spectrum, d=ts))

    useful_spectrum = np.fft.fftshift(np.fft.fft(useful_symbol, n=n_spectrum))
    cp_ofdm_spectrum = np.fft.fftshift(np.fft.fft(tx_with_cp, n=n_spectrum))

    # Optional: individual subcarrier waveforms over the useful interval
    # reconstructed from the centered bin placement for intuition
    subcarrier_waves = []
    for k, a in zip(active_bins, symbols):
        wave = a * np.exp(1j * 2 * np.pi * k * subcarrier_spacing * t_useful) / np.sqrt(n_fft)
        subcarrier_waves.append(wave)
    subcarrier_waves = np.array(subcarrier_waves)

    return {
        "n_fft": n_fft,
        "subcarrier_spacing": subcarrier_spacing,
        "active_bins": list(active_bins),
        "symbols": list(symbols),
        "cp_len": cp_len,
        "fs": fs,
        "ts": ts,
        "useful_symbol_duration": useful_symbol_duration,
        "cp_duration": cp_duration,
        "total_symbol_duration": total_symbol_duration,
        "freq_bins_centered": freq_bins_centered,
        "freq_bins_ifft": freq_bins_ifft,
        "subcarrier_waves": subcarrier_waves,
        "useful_symbol": useful_symbol,
        "cp": cp,
        "tx_with_cp": tx_with_cp,
        "t_useful": t_useful,
        "t_with_cp": t_with_cp,
        "freq_axis": freq_axis,
        "useful_spectrum": useful_spectrum,
        "cp_ofdm_spectrum": cp_ofdm_spectrum,
    }