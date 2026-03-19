# OFDM / 5G-Like Waveform Generator

A Python-based OFDM waveform generator built to develop intuition for 5G-style waveforms and provide a starting point for future ISAC (Integrated Sensing and Communications) projects.

This project generates a simplified 5G-like OFDM frame with configurable waveform parameters and produces several useful visualizations for communications analysis.

## Features

- Configurable FFT size
- Configurable subcarrier spacing
- Configurable number of OFDM symbols
- Supports:
  - QPSK
  - 16QAM
- Pilot tone insertion
- OFDM modulation with cyclic prefix
- Visualization of:
  - time-domain waveform
  - constellation
  - spectrum
  - resource grid

## Why this project matters

This project builds the communications waveform foundation needed for more advanced wireless and ISAC work.

A large amount of modern communications and sensing research starts with OFDM because it allows you to study:

- subcarrier allocation
- modulation
- pilot placement
- time/frequency structure
- spectral behavior
- channel estimation foundations
- future sensing extensions such as range and Doppler processing

A 5G NR downlink-style waveform is also a common starting point for ISAC examples.

## Project structure

```text
OFDM-5G-Waveform-Generator/
│
├── ofdm_waveform_generator.py
├── requirements.txt
├── README.md
└── images/
    ├── resource_grid.png
    ├── constellation.png
    ├── spectrum.png
    └── time_waveform.png

# Expected outputs

When the script runs successfully, it should produce the following outputs.

### 1. Terminal / console summary
A short text summary is printed to the terminal showing the key waveform settings and derived values, such as:

- FFT size
- subcarrier spacing
- sample rate
- number of OFDM symbols
- modulation type
- number of active subcarriers
- cyclic prefix length
- pilot spacing
- waveform length
- approximate occupied bandwidth

Example:

```text
========== OFDM / 5G-like Waveform Generator ==========
FFT size                : 256
Subcarrier spacing      : 30.0 kHz
Sample rate             : 7.680 MHz
OFDM symbols            : 14
Modulation              : 16QAM
Active subcarriers      : 180
Cyclic prefix length    : 18 samples
Pilot spacing (freq)    : every 12 active SCs
Pilot spacing (time)    : every 4 OFDM symbols
Waveform length         : 3836 samples
Occupied bandwidth ~    : 5.400 MHz
=======================================================