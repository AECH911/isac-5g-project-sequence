# OFDM / 5G-Like Waveform Learning Module

A Python-based lesson-driven OFDM waveform generator built to help develop intuition for 5G-style wireless systems and provide a foundation for future ISAC (Integrated Sensing and Communications) work.

This project is structured as a **learning module**, not just a single plotting script. It walks through the major ideas behind OFDM waveform generation:

1. modulation symbols
2. resource-grid construction
3. frequency-domain OFDM symbols
4. time-domain waveform generation using IFFT
5. cyclic prefix insertion
6. spectrum visualization
7. orthogonality intuition

---

## Project purpose

The goal of this project is to make OFDM more visually and conceptually understandable.

Many beginners can follow the code for:
- FFT
- IFFT
- QPSK
- 16QAM
- subcarrier mapping

but still struggle to connect that with:

- resource grids
- subcarriers
- OFDM symbols
- time-domain waveforms
- spectrum overlap
- orthogonality

This module is designed to bridge that gap.

---

## What this project teaches

This project is a hands-on Python learning environment for understanding OFDM concepts, including:

- individual subcarriers in time
- summed OFDM symbols
- individual subcarrier spectra
- combined OFDM symbol spectra
- discrete frequency-domain bin values

---

## Current structure

- `main.py` - runs the lesson/demo
- `ofdm_math.py` - waveform and OFDM math generation
- `ofdm_plots.py` - plotting functions
- `modulation.py` - modulation constellation helpers
- `nr_resources.py` - resource block and subcarrier spacing helpers

---

## Planned Waveform Expansion

The current implementation begins with the OFDM synthesis equation to build intuition for
orthogonal subcarriers, discrete frequency-domain bin placement, and spectrum formation.

The long-term goal is to expand `ofdm_math.py` into a waveform-generation module that supports:

- Plain OFDM / synthesis-equation view
- CP-OFDM
- DFT-s-OFDM
- LTE and 5G NR-oriented numerology and resource-block examples
- Additional waveform-specific transmit processing as the project matures

This allows the same plotting tools to be reused while comparing different waveform generation methods.

---

## OFDM Math Background

The current implementation begins with the OFDM synthesis equation, which is a direct
time-domain representation of an OFDM symbol as a sum of orthogonal subcarriers.

\[
x[n] = \frac{1}{\sqrt{N}} \sum_{k \in \mathcal{K}} a_k e^{j 2 \pi k \Delta f t_n}
\]

Where:

- \(a_k\) is the complex symbol placed on subcarrier \(k\)
- \(k\) is the subcarrier index
- \(\Delta f\) is the subcarrier spacing
- \(N\) is the FFT size
- \(t_n\) is the discrete-time sample location
- \(\mathcal{K}\) is the set of active subcarriers

This representation is useful for building intuition because it shows that an OFDM signal
can be interpreted as the sum of multiple orthogonal complex exponentials, each weighted
by a modulation symbol.

In implementation-focused systems such as LTE and 5G NR, OFDM is usually generated using
frequency-domain bin mapping followed by an IFFT. This project is being developed to show
both perspectives:

1. the synthesis-equation view for learning and intuition
2. the implementation view used for CP-OFDM and later waveform models

## Setup
```bash
pip install -r requirements.txt