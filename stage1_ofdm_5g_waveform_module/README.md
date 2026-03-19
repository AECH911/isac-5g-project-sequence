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

This module is organized around two main learning tracks:

### 1. Waveform generation lesson
This lesson shows how a 5G-like OFDM waveform is built from:

- modulation symbols
- active subcarriers
- pilot tones
- frequency-domain resource mapping
- IFFT-based OFDM modulation
- cyclic prefix insertion

### 2. Orthogonality lesson
This lesson provides a smaller, more intuitive OFDM demo to show:

- individual subcarriers in time
- summed OFDM symbols
- smooth subcarrier spectra
- combined OFDM spectrum
- the difference between smooth spectrum views and FFT-bin views

---

## Features

- Configurable FFT size
- Configurable subcarrier spacing
- Configurable number of OFDM symbols
- Supports:
  - QPSK
  - 16QAM
- Pilot tone insertion
- Frequency-domain resource grid generation
- OFDM modulation using IFFT
- Cyclic prefix insertion
- Time-domain waveform generation
- Spectrum visualization
- Resource-grid visualization
- Orthogonality teaching demo

---

## Current code structure

The cleaned-up script is organized into sections:

- `CONFIG`
- `MODULATION`
- `RESOURCE GRID / OFDM CORE`
- `SUMMARY / INFO`
- `PLOTTING: MAIN WAVEFORM LESSONS`
- `PLOTTING: ORTHOGONALITY LESSON`
- `LESSON RUNNERS`
- `MAIN`

This makes the project read more like a guided signal-generation module rather than a collection of unrelated experiments.

---

## Example project structure

```text
OFDM-5G-Waveform-Learning-Module/
│
├── ofdm_waveform_lesson.py
├── README.md
├── requirements.txt
└── images/
    ├── resource_grid_type_map.png
    ├── resource_grid_magnitude.png
    ├── constellation_plot.png
    ├── frequency_domain_symbol.png
    ├── symbol_domain_comparison.png
    ├── time_domain_waveform.png
    ├── waveform_spectrum.png
    ├── orthogonality_subcarriers_time.png
    ├── orthogonality_sum_time.png
    ├── orthogonality_subcarrier_spectra.png
    ├── orthogonality_combined_spectrum.png
    └── orthogonality_fft_bins.png