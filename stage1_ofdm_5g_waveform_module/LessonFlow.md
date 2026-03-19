# Main learning flow
The script is designed around a lesson flow.

## Lesson 1 — Build the resource grid
The code creates a frequency-domain grid where:
- rows = subcarriers
- columns = OFDM symbols
- entries = complex symbols, pilots, or unused/nulls

## Lesson 2 — Understand resource types
The code separates resource elements into:
- null / unused
- data
- pilot
This helps distinguish structure from actual symbol values.

## Lesson 3 — Inspect one OFDM symbol in the frequency domain

A single column of the resource grid is examined to show:
- magnitude
- real part
- imaginary part
- phase
This shows what is actually loaded onto the subcarriers before the IFFT.

## Lesson 4 — Convert frequency domain to time domain
The script uses the IFFT to convert the frequency-domain OFDM symbol into a time-domain waveform.

## Lesson 5 — Add the cyclic prefix
A cyclic prefix is appended to each OFDM symbol to model practical OFDM transmission behavior.

## Lesson 6 — Serialize the full waveform
All OFDM symbols are concatenated into one transmit waveform.

## Lesson 7 — Visualize the final waveform and spectrum
The script plots:
- time-domain waveform
- overall spectrum
- constellation
- resource-grid views

## Lesson 8 — Orthogonality intuition demo
A separate small demo shows:
- individual subcarriers in time
- their sum in time
- their smooth spectra
- their discrete FFT-bin representation
This is meant to help connect textbook OFDM diagrams with the actual code implementation.