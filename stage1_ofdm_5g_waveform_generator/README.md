# OFDM / 5G-Like Waveform Generator

A Python-based OFDM waveform generator designed to build intuition for 5G-style wireless systems and serve as a foundation for future ISAC (Integrated Sensing and Communications) work.

This project generates a simplified **5G-like OFDM frame** with configurable waveform parameters and visualizes the signal in both time and frequency domains.

## Features

- Configurable FFT size
- Configurable subcarrier spacing
- Configurable number of OFDM symbols
- Supports:
  - QPSK
  - 16QAM
- Pilot tone insertion
- Outputs:
  - Time-domain waveform
  - Constellation plot
  - Spectrum plot
  - Resource-grid visualization

## Why this project matters

This project builds the communication waveform foundation needed for more advanced wireless and ISAC work.

In modern communication and sensing research, OFDM waveforms are often used as the starting point because they allow you to:

- map data onto subcarriers
- model realistic resource grids
- study modulation behavior
- analyze spectral occupancy
- extend into channel estimation, equalization, and sensing

A 5G NR downlink waveform is also commonly used as the base signal in ISAC examples, including official MathWorks workflows.

## Project structure

A simple starter structure might look like this:

```text
OFDM-5G-Waveform-Generator/
│
├── ofdm_waveform_generator.py
├── README.md
└── requirements.txt