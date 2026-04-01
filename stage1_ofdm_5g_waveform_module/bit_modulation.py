import numpy as np


def get_constellation(modulation: str) -> np.ndarray:
    modulation = modulation.upper()

    if modulation == "BPSK":
        return np.array([1+0j, -1+0j], dtype=complex)

    if modulation == "QPSK":
        return np.array([
            1+1j, 1-1j, -1+1j, -1-1j
        ], dtype=complex) / np.sqrt(2)

    if modulation == "16QAM":
        levels = np.array([-3, -1, 1, 3], dtype=float)
        return np.array([i + 1j*q for i in levels for q in levels], dtype=complex) / np.sqrt(10)

    if modulation == "64QAM":
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=float)
        return np.array([i + 1j*q for i in levels for q in levels], dtype=complex) / np.sqrt(42)

    if modulation == "256QAM":
        levels = np.array([-15, -13, -11, -9, -7, -5, -3, -1,
                            1,   3,   5,   7,  9, 11, 13, 15], dtype=float)
        return np.array([i + 1j*q for i in levels for q in levels], dtype=complex) / np.sqrt(170)

    raise ValueError(f"Unsupported modulation: {modulation}")


def get_symbols(modulation: str, num_symbols: int) -> list[complex]:
    constellation = get_constellation(modulation)

    if num_symbols <= len(constellation):
        return list(constellation[:num_symbols])

    repeats = int(np.ceil(num_symbols / len(constellation)))
    expanded = np.tile(constellation, repeats)
    return list(expanded[:num_symbols])