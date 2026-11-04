# LoRa Fingerprinting Simulator

Desktop application that simulates RSSI fingerprinting localization with LoRa (868 MHz): four fixed
base stations and one hand-held receiver. It estimates the localization error and the time needed for
the field work for a given site, radio configuration and grid.

![Map tab](docs/app_map.png)

## Simulated system

- **4 base stations** transmit short frames in turn.
- **1 receiver** averages RSSI and SNR over N frames from each station.
- **Scan:** the receiver measures at every node of a grid over the area; the averaged values form the
  node's fingerprint.
- **Localization:** a new fingerprint measured at an unknown position is compared with the map.
  **NN** returns the most similar node, **WkNN** the weighted mean of the k most similar nodes.

## Model

Received power of one frame from station *s* at position *x*:

```
P_rx = P_tx + 2·G − PL(d) − S_s(x) − M_s(x) − N
```

| Term | Model |
|---|---|
| PL(d) | **two-ray** (default, open ground): direct wave plus the wave reflected by flat ground, from the station and receiver antenna heights and the ground type (vertical polarization, ground permittivity and conductivity from ITU-R P.2040 Table 3); about free space up to the breakpoint `4·h_station·h_receiver / λ` (28 m for 2 m and 1.2 m), then 40 dB per decade. **Log-distance**: `PL(d0) + 10·n·log10(d / d0)` on the 3-D distance, never below free-space loss |
| S_s(x) | shadowing: Gaussian random field with standard deviation σ and correlation `exp(−Δ / d_c)`; `S_s = √ρ·C + √(1−ρ)·I_s`, a field C shared by all stations plus the station's own field I_s, so any two stations are correlated with ρ |
| M_s(x) | multipath: Ricean channel with K-factor (Rayleigh for no line of sight). A fixed part per receiver position (independent between positions) plus a share of the power that is new in every frame |
| N | frame noise: normal, independent for every frame |

**Total σ.** Published path-loss models such as Bor et al. 2016 have one random term, so their σ is the
whole spread of single frames around the curve. With *σ is the total spread* ticked, the shadowing
simulated is `√(σ² − σ_multipath² − σ_frame²)`. If multipath and frame noise alone spread more than σ
(for example Rayleigh fading, 5.6 dB, inside Bor's 3.57 dB), the shadowing is set to 0 and a note is
shown.

**Receiver positions.** At every grid node (and at every test point) the receiver can be moved over
several positions more than half a wavelength apart, e.g. within 1 m; the frames are spread over them in
turn. Averaging over positions reduces the fixed multipath, averaging over frames only the part that
changes from frame to frame.

**Receiver**

- Noise floor `−174 dBm/Hz + 10·log10(BW) + NF`. A frame is received when its SNR is above the limit
  of the spreading factor (SF7 −7.5 dB … SF12 −20 dB).
- Reported RSSI = signal + noise in 0.5 dB steps; reported SNR in 0.25 dB steps, capped at a maximum.
- A fingerprint is the mean of the received frames; a station with no received frame is stored as
  −140 dBm.

**Localization:** Euclidean distance between fingerprints in dB; WkNN weights `1 / distance`.

**Timing**

- Time on air from the Semtech LoRa formula (coding rate 4/5, 8-symbol preamble, explicit header,
  CRC on, low data rate optimisation for symbols longer than 16 ms).
- Each station respects the duty cycle of the sub-band (1 % or 10 %). Scan time per grid node:
  `max(4 · N · (T_air + gap), N · T_air / duty cycle)`.

**Evaluation:** random test points over the whole area (not on grid nodes); mean, median and
90th-percentile error, frame loss, scan time, time of one localization and localizations per hour.

## Parameters

All parameters are set in the panel; every change is applied with **Run simulation**.

| Group | Parameters |
|---|---|
| Area and grid | width, height (100 m – 10 km), grid spacing (the grid always reaches all edges; limit 40 000 nodes) |
| Stations | corners, side midpoints, random or manual coordinates; station antenna height and receiver height |
| Radio | sub-band, spreading factor, bandwidth, Tx power (with an e.r.p. check), antenna gain, payload, gap between frames, noise figure, SNR report maximum |
| Propagation | preset, path-loss model (two-ray or log-distance), ground type, d0, PL(d0), n, shadowing σ (or total σ), correlation distance, station correlation ρ, K-factor, share of the multipath changing per frame, frame noise σ, seed |
| Frames | frames per station for the scan and for one localization, receiver positions per grid node and per localization, number of test points |
| Localization | NN or WkNN, k, RSSI or RSSI + SNR |

**Presets**

| Preset | For | Sources |
|---|---|---|
| Open area, line of sight (two-ray) — default | open ground without buildings, small obstacles (benches) | two-ray model; ground: ITU-R P.2040; shadowing 4 dB, correlation 10 m, K 9 dB: 3GPP TR 38.901 UMi LOS (base stations at 10 m, so a starting point); station correlation 0.5: 3GPP TR 36.942 |
| Free space | reference | ITU-R P.525 |
| Built-up, indoor nodes (Bor 2016) | nodes inside buildings, 40–100 m | Bor et al. 2016 (worst case by the authors' own description) |
| Urban, rooftop gateways (Rademacher 2021) | gateways on roofs, 1–13 km | Rademacher et al. 2021 |

The propagation parameters describe a particular site and should be measured there: path loss versus
distance gives d0, PL(d0), n and σ; the spatial correlation of the residuals gives the correlation
distance; repeated measurements at one position give the frame noise and the multipath spread. The
tooltip of every field says where its default value comes from or that it is an assumption.

## Application

| Tab | Content |
|---|---|
| **Map** | stations, grid nodes, true and estimated positions of the test points |
| **RSSI / coverage** | scan map of mean RSSI, mean SNR or percentage of received frames for one station |
| **Error** | error at each test point and the error distribution (CDF) of both methods |
| **Sweeps** | mean and 90th-percentile error versus grid spacing, zone size, scan frames, localization frames, scan positions, station correlation, station height or k |

The current figure can be saved as PNG and the fingerprint map exported as CSV
(`x_m, y_m, rssi_s1..s4, snr_s1..s4, frames_s1..s4`).

| | |
|---|---|
| ![RSSI map](docs/app_rssi.png) | ![Error](docs/app_error.png) |
| ![Grid spacing sweep](docs/app_sweep_grid.png) | ![k sweep](docs/app_sweep_k.png) |

## Running

Python 3.10 or newer. On Windows:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

After the first set-up, `uruchom.bat` starts the application without a console window. On Linux or
macOS use `.venv/bin/activate`.

Tests (`python -m pytest`) check the time on air, the sensitivity, the reported value steps, the
path-loss model and its free-space bound, the two-ray model (40 dB per decade beyond the breakpoint,
ground reflection, ground types), the 3-D distance, the standard deviation and correlation of the shadowing
field, the correlation between stations, the Rayleigh spread, the reduction of the multipath by
receiver positions and by frames, the total σ, the e.r.p. limit, the grid edges, the no-signal
handling, the duty-cycle timing, reproducibility for a given seed and the sweeps.

## Project layout

```
app.py                  desktop application (PySide6 + matplotlib)
uruchom.bat             starts app.py without a console window (Windows)
lorasim/config.py       settings and sub-bands
lorasim/radio.py        time on air, noise floor, sensitivity, reported values
lorasim/propagation.py  path loss (log-distance, two-ray), correlated shadowing, Ricean multipath
lorasim/simulation.py   scan, localization (NN, WkNN), timing, evaluation
lorasim/sweep.py        error as a function of one parameter
tests/                  pytest tests
docs/                   screenshots
```

## Limitations

- The two-ray model assumes flat ground and antennas that radiate equally at all elevation angles; a
  real antenna radiates less steeply downwards, so very close to a station the loss is underestimated.
  Single obstacles (benches) are only in the shadowing statistics, not placed on the map.
- The receiver is always held at the same height and its user's body is not modelled.
- The log-distance model has one slope for all distances.
- The correlation between stations is the same for every pair (it does not depend on the angle
  between the links).
- The part of the multipath that changes is independent from frame to frame (no coherence time).
- The environment does not change between the scan and the localization.
- Frames from different stations never collide.
- Scan time counts radio time only, not walking between grid nodes.
- For very large areas the shadowing raster is coarser than 1/10 of the correlation distance (a note
  is shown).
