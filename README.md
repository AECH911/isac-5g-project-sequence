# ISAC 5G Project Sequence

A hands-on Python project sequence for learning and demonstrating **Integrated Sensing and Communications (ISAC)** concepts using a 5G-style waveform progression.

This repository is structured as a staged learning and portfolio project. Each stage builds on the previous one, moving from basic OFDM waveform generation toward a more complete joint communications-and-sensing workflow.

## Project Goals

The purpose of this project is to:

- build practical Python simulation skills
- strengthen understanding of OFDM and wireless communications
- explore sensing concepts using communications waveforms
- create a portfolio-quality technical project
- practice professional Git and GitHub workflow on a larger repo

## Planned Project Stages

### 1. OFDM / 5G Waveform Generation
Build a basic OFDM waveform generator with:
- configurable FFT size
- active subcarriers
- QPSK symbol mapping
- time-domain waveform generation
- constellation and spectrum visualization

### 2. Channel Simulation
Extend the waveform through a communications channel with:
- AWGN
- optional multipath extensions
- receive-side demodulation
- BER evaluation

### 3. Simple Radar / Sensing Extraction
Use the transmitted waveform for basic sensing concepts:
- target delay
- Doppler shift
- reflected-path modeling
- simple target response estimation

### 4. Communications Metrics
Evaluate communications-side performance using:
- BER
- SNR
- constellation quality
- received signal behavior

### 5. Joint Tradeoff Analysis
Study tradeoffs between sensing and communications performance, such as:
- bandwidth vs range resolution
- waveform settings vs BER
- sensing observability vs communications efficiency

## Repository Structure

```text
ISAC-5G-PROJECT-SEQUENCE/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
├── figures/
├── src/
├── stage1_ofdm_5g_waveform_generator/
├── stage2_channel_simulation/
├── stage3_simple_radar_sensing_extraction/
├── stage4_communications_metrics/
└── stage5_joint_tradeoff_analysis/