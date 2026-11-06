"""LoRa Fingerprinting Simulator - desktop application (PySide6 + matplotlib)."""
import csv
import random
import sys

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6 import QtCore, QtGui, QtWidgets

from lorasim import radio
from lorasim.config import (ALGORITHMS, DEFAULT_ENVIRONMENT, ENVIRONMENTS, K_NONE, K_RAYLEIGH,
                            MAX_GRID_POINTS, STATION_LAYOUTS, SUB_BANDS, Config, Environment, Radio)
from lorasim.propagation import GROUND_TYPES, PATH_LOSS_MODELS
from lorasim.simulation import actual_spacing, grid_axis, grid_size, simulate
from lorasim.sweep import LOG_AXIS, PARAMETERS, UNITS, run_sweep

COLORS = {"WkNN": "#1f77b4", "NN": "#2ca02c"}
MAP_POINTS = 150
PANEL_WIDTH = 400

# Shown as tooltips: where each value comes from, or that it is an assumption.
TIPS = {
    "sub_band": "Source: ERC/REC 70-03 Annex 1 - duty cycle and maximum e.r.p. of the sub-band. Channel: "
                "LoRaWAN EU868 (868.1 MHz, 869.525 MHz); a bandwidth that does not fit there uses the band "
                "centre, and bandwidths wider than the sub-band are not available.",
    "tx": "Conducted output power. e.r.p. = Tx + antenna gain - 2.15 dB (e.r.p. is referred to a half-wave "
          "dipole of 2.15 dBi; ITU Radio Regulations, Article 1) must not exceed the sub-band limit.",
    "gain": "Project antenna: 868 MHz, 3 dBi, SMA with U.FL adapter (seller's specification, not measured). "
            "The same antenna is assumed on stations and receiver; adapter loss is not modelled.",
    "payload": "Design choice: bytes in one frame (firmware setting). Fixed: coding rate 4/5, "
               "8-symbol preamble, explicit header, CRC on.",
    "guard": "ASSUMPTION (no source): pause between consecutive frames.",
    "nf": "Receiver noise figure. Default 6 dB: with the demodulation SNR limits it gives -124.5 dBm at "
          "SF7 and -137 dBm at SF12 (125 kHz), in line with the SX1262 datasheet sensitivity. Measure the "
          "noise floor with the real SX1262 (instantaneous RSSI while no station transmits).",
    "d0": "Reference distance of the log-distance model (source: see the preset).",
    "pl0": "Mean path loss at the reference distance (source: see the preset).",
    "n": "Path-loss exponent of the log-distance model (source: see the preset).",
    "shadow": "Log-normal shadowing standard deviation. Default: 4 dB, 3GPP TR 38.901 UMi LOS.",
    "corr": "Distance at which shadowing correlation falls to 1/e. Exponential model "
            "(Gudmundson 1991; 3GPP TR 38.901). Default 10 m: 3GPP TR 38.901 Table 7.5-6, UMi LOS.",
    "k": "Ricean K-factor of the multipath at a fixed position. 3GPP TR 38.901 Table 7.5-6: "
         "LOS 7-9 dB, NLOS Rayleigh. At the minimum: Rayleigh; 40 dB or more: no multipath.",
    "noise": "ASSUMPTION (no source): frame-to-frame variation at one position. Measure it with "
             "the real boards (static link, several minutes).",
    "dynamic": "ASSUMPTION (no source): share of the multipath power that changes from frame to frame "
               "(people, vehicles, leaves). 0 %: the multipath at a position never changes, so more "
               "frames do not reduce it; 100 %: new fading in every frame. Measure it with a static "
               "link: spread of single frames versus spread of the per-position means.",
    "station_corr": "Correlation of the shadowing between any two stations: obstacles near the "
                    "receiver shadow the links to several stations at once. 0.5: 3GPP TR 36.942 "
                    "(between base station sites); 0: independent.",
    "preset": "Parameter sets with their sources. Choosing one fills the fields below; every field "
              "can then be changed.",
    "model": "Log-distance: PL(d0) + 10 n log10(d/d0) on the 3-D distance, fitted to measurements. "
             "Two-ray: direct wave plus the wave reflected by flat ground, from the antenna heights "
             "and the ground type (open ground with line of sight; d0, PL(d0) and n are not used).",
    "ground": "Ground under the links, for the reflected wave of the two-ray model. Permittivity and "
              "conductivity: ITU-R P.527-6 soil model (§ 5.2), silty loam from its Table 2 at 23 °C. "
              "7 % and 50 % water are the dry and wet examples of the Recommendation; 30 % is an "
              "assumption between them.",
    "station_height": "ASSUMPTION: height of the station antennas above ground. With the two-ray model "
                      "it sets the breakpoint 4 h_station h_receiver / wavelength (28 m for 2 m and 1.2 m).",
    "receiver_height": "ASSUMPTION: height at which the receiver is held. Keep it the same for the scan "
                       "and for localization: 1.0 m instead of 1.2 m changes the two-ray loss by about "
                       "1 dB on average and up to 5 dB near the stations.",
    "positions": "Receiver positions around the point, more than half a wavelength (17 cm) apart, "
                 "e.g. within 1 m. The frames are spread over the positions in turn, so averaging "
                 "also reduces the fixed multipath. 1: the receiver does not move.",
}


def fmt_time(seconds):
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def length_unit(cfg):
    return (1000.0, "km") if max(cfg.width, cfg.height) >= 2000 else (1.0, "m")


def fmt_len(metres):
    return f"{metres / 1000:.2f} km" if metres >= 1000 else f"{metres:.1f} m"


class Job(QtCore.QThread):
    """Runs a function in the background so the window stays responsive."""
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(self.progress.emit))
        except Exception as exc:          # shown to the user in a message box
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class WheelGuard(QtCore.QObject):
    """The mouse wheel never changes a value; over the settings panel it scrolls the panel instead."""

    def __init__(self, parent, scrollbar=None):
        super().__init__(parent)
        self.scrollbar = scrollbar

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Wheel:
            if self.scrollbar is not None:
                QtWidgets.QApplication.sendEvent(self.scrollbar, event)
            return True
        return False


class PlotTab(QtWidgets.QWidget):
    def __init__(self, controls=None):
        super().__init__()
        self.figure = Figure(figsize=(9, 6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        if controls is not None:
            layout.addLayout(controls)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)


def dspin(lo, hi, value, step=1.0, decimals=1, suffix=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    s.setSuffix(suffix)
    return s


def ispin(lo, hi, value, step=1, suffix=""):
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(value)
    s.setSuffix(suffix)
    return s


def combo(items, current=None):
    c = QtWidgets.QComboBox()
    c.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(8)
    c.addItems([str(i) for i in items])
    if current is not None:
        c.setCurrentText(str(current))
    return c


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.setLocale(QtCore.QLocale())
        self.setWindowTitle("LoRa Fingerprinting Simulator")
        self.resize(1480, 920)
        self.result = None
        self.job = None
        d = Config()

        # ---------- settings panel ----------
        panel = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(panel)

        g, f = self._group("Area and grid", form)
        self.width_ = dspin(100, 10000, d.width, 50, 0, " m")
        self.height_ = dspin(100, 10000, d.height, 50, 0, " m")
        self.spacing = dspin(1, 1000, d.grid_spacing, 1, 1, " m")
        self.grid_info = QtWidgets.QLabel()
        self.grid_info.setWordWrap(True)
        f.addRow("Width", self.width_)
        f.addRow("Height", self.height_)
        f.addRow("Grid spacing", self.spacing)
        f.addRow("", self.grid_info)

        g, f = self._group("Stations", form)
        self.layout_ = combo(STATION_LAYOUTS, d.station_layout)
        self.table = QtWidgets.QTableWidget(4, 2)
        self.table.setHorizontalHeaderLabels(["x [m]", "y [m]"])
        self.table.setVerticalHeaderLabels([f"S{i + 1}" for i in range(4)])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setFixedHeight(150)
        self._fill_table(d.manual_stations)
        self.station_height = dspin(0.2, 30, d.station_height, 0.1, 1, " m")
        self.receiver_height = dspin(0.2, 10, d.receiver_height, 0.1, 1, " m")
        self.station_height.setToolTip(TIPS["station_height"])
        self.receiver_height.setToolTip(TIPS["receiver_height"])
        f.addRow("Layout", self.layout_)
        f.addRow(self.table)
        f.addRow("Station antenna height", self.station_height)
        f.addRow("Receiver height", self.receiver_height)

        g, f = self._group("Radio (868 MHz)", form)
        r = d.radio
        self.sub_band = combo(SUB_BANDS, r.sub_band)
        self.sf = combo(range(7, 13), r.spreading_factor)
        self.bw = combo([125, 250, 500], int(r.bandwidth_khz))
        self.tx = dspin(-9, 22, r.tx_power_dbm, 1, 0, " dBm")
        self.gain = dspin(0, 10, r.antenna_gain_dbi, 0.5, 1, " dBi")
        self.payload = ispin(1, 255, r.payload_bytes, suffix=" B")
        self.guard = dspin(0, 1000, r.guard_ms, 1, 0, " ms")
        self.nf = dspin(0, 20, r.noise_figure_db, 0.5, 1, " dB")
        for w, key in ((self.sub_band, "sub_band"), (self.tx, "tx"), (self.gain, "gain"),
                       (self.payload, "payload"), (self.guard, "guard"), (self.nf, "nf")):
            w.setToolTip(TIPS[key])
        self.radio_info = QtWidgets.QLabel()
        self.radio_info.setWordWrap(True)
        f.addRow("Sub-band", self.sub_band)
        f.addRow("Spreading factor", self.sf)
        f.addRow("Bandwidth [kHz]", self.bw)
        f.addRow("Tx power", self.tx)
        f.addRow("Antenna gain", self.gain)
        f.addRow("Payload", self.payload)
        f.addRow("Gap between frames", self.guard)
        f.addRow("Noise figure", self.nf)
        f.addRow(self.radio_info)

        g, f = self._group("Propagation", form)
        self.preset = combo(ENVIRONMENTS, DEFAULT_ENVIRONMENT)
        self.preset.setToolTip(TIPS["preset"])
        self.model = combo(PATH_LOSS_MODELS)
        self.model.setToolTip(TIPS["model"])
        self.ground = combo(GROUND_TYPES)
        self.ground.setToolTip(TIPS["ground"])
        self.d0 = dspin(1, 10000, 1, 1, 0, " m")
        self.pl0 = dspin(20, 200, 100, 0.1, 2, " dB")
        self.n = dspin(1.0, 6, 2, 0.1, 2)
        self.shadow = dspin(0, 15, 0, 0.5, 2, " dB")
        self.corr = dspin(1, 500, 10, 1, 0, " m")
        self.kf = dspin(K_RAYLEIGH, K_NONE, K_RAYLEIGH, 1, 0, " dB")
        self.kf.setSpecialValueText("Rayleigh (no LOS)")
        self.noise = dspin(0, 10, 0, 0.5, 1, " dB")
        self.dynamic = dspin(0, 100, 0, 5, 0, " %")
        self.station_corr = dspin(0, 0.95, 0, 0.05, 2)
        self.seed = ispin(0, 999_999, d.seed)
        self.new_terrain = QtWidgets.QPushButton("New terrain")
        self.new_terrain.setToolTip("Draw a new random shadowing and multipath pattern (changes the seed)")
        for w, key in ((self.d0, "d0"), (self.pl0, "pl0"), (self.n, "n"), (self.shadow, "shadow"),
                       (self.corr, "corr"), (self.kf, "k"), (self.noise, "noise"), (self.dynamic, "dynamic"),
                       (self.station_corr, "station_corr")):
            w.setToolTip(TIPS[key])
        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setWordWrap(True)
        self.preset_note.setStyleSheet("color:#555; font-size:8pt")
        f.addRow("Preset", self.preset)
        f.addRow(self.preset_note)
        f.addRow("Path-loss model", self.model)
        f.addRow("Ground", self.ground)
        f.addRow("Reference distance d0", self.d0)
        f.addRow("Path loss at d0", self.pl0)
        f.addRow("Path-loss exponent n", self.n)
        f.addRow("Shadowing σ", self.shadow)
        f.addRow("Correlation distance", self.corr)
        f.addRow("Station correlation ρ", self.station_corr)
        f.addRow("Multipath K-factor", self.kf)
        f.addRow("Multipath changing per frame", self.dynamic)
        f.addRow("Frame noise σ", self.noise)
        f.addRow("Seed", self.seed)
        f.addRow("", self.new_terrain)
        self.set_environment(d.env)

        g, f = self._group("Frames", form)
        self.scan_frames = ispin(10, 500, d.scan_frames, 10)
        self.locate_frames = ispin(1, 200, d.locate_frames)
        self.test_points = ispin(100, 2000, d.test_points, 100)
        self.scan_positions = ispin(1, 20, d.scan_positions)
        self.locate_positions = ispin(1, 20, d.locate_positions)
        for w in (self.scan_positions, self.locate_positions):
            w.setToolTip(TIPS["positions"])
        f.addRow("Scan, per station", self.scan_frames)
        f.addRow("Localization, per station", self.locate_frames)
        f.addRow("Scan positions per node", self.scan_positions)
        f.addRow("Localization positions", self.locate_positions)
        f.addRow("Test points", self.test_points)

        g, f = self._group("Localization", form)
        self.algorithm = combo(ALGORITHMS, d.algorithm)
        self.k = ispin(1, 10, d.k)
        f.addRow("Algorithm", self.algorithm)
        f.addRow("k (WkNN)", self.k)
        form.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(PANEL_WIDTH)

        self.run_button = QtWidgets.QPushButton("Run simulation")
        self.run_button.setMinimumHeight(40)
        font = self.run_button.font()
        font.setBold(True)
        self.run_button.setFont(font)
        self.pending = QtWidgets.QLabel()
        self.pending.setStyleSheet("color:#c25e00")
        left = QtWidgets.QVBoxLayout()
        left.addWidget(scroll, 1)
        left.addWidget(self.pending)
        left.addWidget(self.run_button)

        # ---------- results ----------
        self.summary = QtWidgets.QLabel("Press Run simulation.")
        self.summary.setTextFormat(QtCore.Qt.RichText)
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("padding:6px; color:#1b1f24; background:#f3f5f8; border:1px solid #d8dde4;")

        self.tabs = QtWidgets.QTabWidget()
        self.map_tab = PlotTab()
        rssi_controls = QtWidgets.QHBoxLayout()
        self.rssi_station = combo([f"S{i + 1}" for i in range(4)])
        self.rssi_quantity = combo(["Mean RSSI", "Frames received"])
        rssi_controls.addWidget(QtWidgets.QLabel("Station"))
        rssi_controls.addWidget(self.rssi_station)
        rssi_controls.addWidget(QtWidgets.QLabel("Value"))
        rssi_controls.addWidget(self.rssi_quantity)
        rssi_controls.addStretch(1)
        self.rssi_tab = PlotTab(rssi_controls)
        self.error_tab = PlotTab()
        sweep_controls = QtWidgets.QHBoxLayout()
        self.sweep_param = combo(PARAMETERS)
        self.sweep_button = QtWidgets.QPushButton("Run sweep")
        self.sweep_progress = QtWidgets.QProgressBar()
        self.sweep_progress.setMaximumWidth(220)
        sweep_controls.addWidget(QtWidgets.QLabel("Parameter"))
        sweep_controls.addWidget(self.sweep_param)
        sweep_controls.addWidget(self.sweep_button)
        sweep_controls.addWidget(self.sweep_progress)
        sweep_controls.addStretch(1)
        self.sweep_tab = PlotTab(sweep_controls)
        self.tabs.addTab(self.map_tab, "Map")
        self.tabs.addTab(self.rssi_tab, "RSSI / coverage")
        self.tabs.addTab(self.error_tab, "Error")
        self.tabs.addTab(self.sweep_tab, "Sweeps")

        export = QtWidgets.QHBoxLayout()
        self.save_png = QtWidgets.QPushButton("Save figure (PNG)")
        self.save_csv = QtWidgets.QPushButton("Export fingerprint map (CSV)")
        export.addStretch(1)
        export.addWidget(self.save_png)
        export.addWidget(self.save_csv)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.summary)
        right.addWidget(self.tabs, 1)
        right.addLayout(export)

        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        root.addLayout(left)
        root.addLayout(right, 1)
        self.setCentralWidget(central)

        # ---------- mouse wheel: scrolls the panel, never changes a value ----------
        panel_guard = WheelGuard(self, scroll.verticalScrollBar())
        tab_guard = WheelGuard(self)
        for w in panel.findChildren(QtWidgets.QAbstractSpinBox) + panel.findChildren(QtWidgets.QComboBox):
            w.setFocusPolicy(QtCore.Qt.StrongFocus)
            w.installEventFilter(panel_guard)
        for w in (self.rssi_station, self.rssi_quantity, self.sweep_param):
            w.setFocusPolicy(QtCore.Qt.StrongFocus)
            w.installEventFilter(tab_guard)

        # ---------- signals ----------
        self.new_terrain.clicked.connect(lambda: self.seed.setValue(random.randint(0, 999_999)))
        self.preset.currentTextChanged.connect(self._preset_changed)
        self.model.currentTextChanged.connect(self._model_changed)
        self.sub_band.currentTextChanged.connect(self._sub_band_changed)
        self._sub_band_changed(self.sub_band.currentText())
        self.layout_.currentTextChanged.connect(self._layout_changed)
        self.run_button.clicked.connect(self.run_simulation)
        self.sweep_button.clicked.connect(self.run_sweep)
        self.rssi_station.currentIndexChanged.connect(self.draw_rssi)
        self.rssi_quantity.currentIndexChanged.connect(self.draw_rssi)
        self.save_png.clicked.connect(self.export_png)
        self.save_csv.clicked.connect(self.export_csv)
        for w in panel.findChildren(QtWidgets.QAbstractSpinBox):
            w.valueChanged.connect(self._changed)
        for w in panel.findChildren(QtWidgets.QComboBox):
            w.currentIndexChanged.connect(self._changed)
        self.table.itemChanged.connect(self._changed)
        self._layout_changed(self.layout_.currentText())
        self._update_info()
        QtCore.QTimer.singleShot(0, self.run_simulation)

    # ---------- panel helpers ----------
    def _group(self, title, parent):
        g = QtWidgets.QGroupBox(title)
        f = QtWidgets.QFormLayout(g)
        parent.addWidget(g)
        return g, f

    def _fill_table(self, positions):
        self.table.blockSignals(True)
        for i, (x, y) in enumerate(positions):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{x:g}"))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{y:g}"))
        self.table.blockSignals(False)

    def _layout_changed(self, name):
        manual = name == "Manual"
        self.table.setEnabled(manual)
        if manual and self.result is not None:
            self._fill_table(self.result.scenario.stations.tolist())

    @staticmethod
    def _spacing_text(cfg):
        sx, sy = (f"{v:.2f}".rstrip("0").rstrip(".") for v in actual_spacing(cfg))
        return f"{sx} m" if sx == sy else f"{sx} × {sy} m"

    def _preset_changed(self, name):
        self.set_environment(ENVIRONMENTS[name].env)

    def _sub_band_changed(self, name):
        """Offer only the bandwidths that fit into the sub-band."""
        band = SUB_BANDS[name]
        model = self.bw.model()
        for i in range(self.bw.count()):
            model.item(i).setEnabled(band.fits(float(self.bw.itemText(i))))
        if not band.fits(float(self.bw.currentText())):
            self.bw.setCurrentText("125")

    def _model_changed(self, name):
        two_ray = name == "Two-ray"
        for w in (self.d0, self.pl0, self.n):
            w.setEnabled(not two_ray)
        self.ground.setEnabled(two_ray)

    def set_environment(self, e):
        """Fill the propagation fields, e.g. with one of the parameter sets in lorasim.config."""
        preset = next((name for name, p in ENVIRONMENTS.items() if p.env == e), None)
        if preset is not None:
            self.preset_note.setText(ENVIRONMENTS[preset].source + " Hover over a field to see its source.")
        self.model.setCurrentText(e.path_loss_model)
        self.ground.setCurrentText(e.ground)
        self._model_changed(e.path_loss_model)
        self.d0.setValue(e.reference_distance)
        self.pl0.setValue(e.reference_loss)
        self.n.setValue(e.path_loss_exponent)
        self.shadow.setValue(e.shadowing_sigma)
        self.corr.setValue(e.correlation_distance)
        self.kf.setValue(e.k_factor_db)
        self.noise.setValue(e.frame_noise_sigma)
        self.dynamic.setValue(e.dynamic_share * 100)
        self.station_corr.setValue(e.station_correlation)

    def _changed(self, *_):
        self._update_info()
        if self.result is not None:
            self.pending.setText("Settings changed - press Run simulation to apply.")

    def _update_info(self):
        try:
            cfg = self.read_config()
        except ValueError:
            return
        points = grid_size(cfg)
        style = "color:#b00020" if points > MAX_GRID_POINTS else "color:#555"
        self.grid_info.setStyleSheet(style)
        text = f"{points:,} grid points (limit {MAX_GRID_POINTS:,})"
        if points <= MAX_GRID_POINTS:
            sx, sy = actual_spacing(cfg)
            if abs(sx - cfg.grid_spacing) > 1e-6 or abs(sy - cfg.grid_spacing) > 1e-6:
                text += f"<br>actual spacing {sx:.2f} × {sy:.2f} m (grid covers the whole area)"
        self.grid_info.setText(text)
        r = cfg.radio
        airtime = radio.time_on_air_ms(r.spreading_factor, r.bandwidth_khz, r.payload_bytes)
        sens = radio.sensitivity_dbm(r.spreading_factor, r.bandwidth_khz, r.noise_figure_db)
        over = r.erp_dbm > r.erp_limit_dbm + 1e-9
        self.radio_info.setStyleSheet("color:#b00020" if over else "color:#555")
        self.radio_info.setText(
            f"Channel {r.frequency_mhz:g} MHz · frame time {airtime:.1f} ms · sensitivity {sens:.1f} dBm<br>"
            f"e.r.p. {r.erp_dbm:.2f} dBm (limit {r.erp_limit_dbm:g} dBm)"
            + (" - <b>above the legal limit</b>" if over else ""))

    def read_config(self):
        stations = []
        for i in range(4):
            try:
                stations.append((float(self.table.item(i, 0).text()), float(self.table.item(i, 1).text())))
            except (AttributeError, ValueError):
                raise ValueError(f"Station S{i + 1}: coordinates must be numbers") from None
        env = Environment(self.d0.value(), self.pl0.value(), self.n.value(), self.shadow.value(),
                          self.corr.value(), self.kf.value(), self.noise.value(),
                          dynamic_share=self.dynamic.value() / 100,
                          station_correlation=self.station_corr.value(),
                          path_loss_model=self.model.currentText(), ground=self.ground.currentText())
        rad = Radio(sub_band=self.sub_band.currentText(), bandwidth_khz=float(self.bw.currentText()),
                    spreading_factor=int(self.sf.currentText()), tx_power_dbm=self.tx.value(),
                    antenna_gain_dbi=self.gain.value(), noise_figure_db=self.nf.value(),
                    payload_bytes=self.payload.value(), guard_ms=self.guard.value())
        if not rad.band.fits(rad.bandwidth_khz):
            raise ValueError(f"A {rad.bandwidth_khz:g} kHz channel does not fit into the sub-band {rad.sub_band}")
        return Config(width=self.width_.value(), height=self.height_.value(),
                      grid_spacing=self.spacing.value(), station_layout=self.layout_.currentText(),
                      manual_stations=tuple(stations), radio=rad, env=env,
                      station_height=self.station_height.value(),
                      receiver_height=self.receiver_height.value(),
                      scan_frames=self.scan_frames.value(), locate_frames=self.locate_frames.value(),
                      scan_positions=self.scan_positions.value(),
                      locate_positions=self.locate_positions.value(),
                      test_points=self.test_points.value(), algorithm=self.algorithm.currentText(),
                      k=self.k.value(), seed=self.seed.value())

    def _checked_config(self):
        try:
            cfg = self.read_config()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid settings", str(exc))
            return None
        points = grid_size(cfg)
        if points > MAX_GRID_POINTS:
            QtWidgets.QMessageBox.warning(
                self, "Grid too dense",
                f"The grid would have {points:,} points (limit {MAX_GRID_POINTS:,}).\n"
                "Increase the grid spacing or reduce the area.")
            return None
        return cfg

    # ---------- running ----------
    def _busy(self, busy, text=""):
        self.run_button.setEnabled(not busy)
        self.sweep_button.setEnabled(not busy)
        if busy:
            self.pending.setText(text)

    def _start(self, fn, on_done, text):
        self._busy(True, text)
        self.job = Job(fn)
        self.job.done.connect(on_done)
        self.job.failed.connect(self._failed)
        self.job.progress.connect(lambda i, n: (self.sweep_progress.setMaximum(n),
                                                self.sweep_progress.setValue(i)))
        self.job.finished.connect(lambda: self._busy(False))
        self.job.start()

    def _failed(self, message):
        self.pending.setText("")
        QtWidgets.QMessageBox.critical(self, "Simulation failed", message)

    def run_simulation(self):
        cfg = self._checked_config()
        if cfg is not None:
            self._start(lambda _p: simulate(cfg, algorithms=ALGORITHMS), self._show_result, "Running...")

    def run_sweep(self):
        cfg = self._checked_config()
        if cfg is None:
            return
        param = self.sweep_param.currentText()
        self.sweep_progress.setValue(0)
        self._start(lambda p: (cfg, param, run_sweep(cfg, param, p)), self._show_sweep,
                    f"Sweeping {param.lower()}...")

    # ---------- results ----------
    def _show_result(self, result):
        self.result = result
        self.pending.setText("")
        cfg = result.cfg
        s = result.stats()
        grid = result.scenario.grid
        notes = []
        if any(getattr(f, "coarse", False) for f in result.scenario.shadow):
            notes.append("shadowing raster is coarser than 1/10 of the correlation distance")
        if cfg.radio.erp_dbm > cfg.radio.erp_limit_dbm + 1e-9:
            notes.append("e.r.p. above the legal limit of the sub-band")
        warn = "<br><span style='color:#b00020'>Note: " + "; ".join(notes) + ".</span>" if notes else ""
        self.summary.setText(
            f"<b>{cfg.algorithm}</b> &nbsp; mean error <b>{fmt_len(s['mean'])}</b> · median {fmt_len(s['median'])}"
            f" · 90th percentile <b>{fmt_len(s['p90'])}</b> · max {fmt_len(s['max'])}"
            f" · frames lost {result.frame_loss * 100:.1f} %<br>"
            f"Scan: {len(grid):,} points × 4 stations × {cfg.scan_frames} frames = "
            f"<b>{fmt_time(result.scan_time_s)}</b> with the {cfg.radio.duty_cycle * 100:g} % duty cycle "
            f"({fmt_time(result.scan_radio_time_s)} on air) · one localization "
            f"<b>{fmt_time(result.locate_time_s)}</b>, at most {result.locates_per_hour:.0f} per hour"
            f" · {cfg.width:g} × {cfg.height:g} m, grid {self._spacing_text(cfg)}" + warn)
        self.draw_map()
        self.draw_rssi()
        self.draw_error()

    def _stations(self, ax, scale, highlight=None):
        st = self.result.scenario.stations / scale
        ax.scatter(st[:, 0], st[:, 1], marker="^", s=150, c="#d62728", edgecolors="black",
                   zorder=6, label="station", clip_on=False)
        for i, (x, y) in enumerate(st):
            weight = "bold" if i == highlight else "normal"
            ax.annotate(f"S{i + 1}", (x, y), xytext=(7, 7), textcoords="offset points",
                        fontsize=10, fontweight=weight, zorder=7)

    def _frame(self, ax, cfg, scale, unit):
        w, h = cfg.width / scale, cfg.height / scale
        ax.add_patch(Rectangle((0, 0), w, h, fill=False, ec="0.35", lw=1.2))
        pad = 0.06 * max(w, h)
        ax.set_xlim(-pad, w + pad)
        ax.set_ylim(-pad, h + pad)
        ax.set_aspect("equal")
        ax.set_xlabel(f"x [{unit}]")
        ax.set_ylabel(f"y [{unit}]")

    def draw_map(self):
        r, fig = self.result, self.map_tab.figure
        cfg = r.cfg
        scale, unit = length_unit(cfg)
        fig.clear()
        ax = fig.add_subplot()
        self._frame(ax, cfg, scale, unit)
        grid = r.scenario.grid / scale
        if len(grid) <= 5000:
            ax.scatter(grid[:, 0], grid[:, 1], s=5, c="#b8bec7", label=f"grid node ({len(grid):,})")
        shown = min(MAP_POINTS, len(r.test_points))          # all points are used for the statistics
        true = r.test_points[:shown] / scale
        est = r.estimates[cfg.algorithm][:shown] / scale
        ax.add_collection(LineCollection(np.stack([true, est], axis=1), colors="#ff7f0e",
                                         linewidths=0.8, alpha=0.7))
        ax.scatter(true[:, 0], true[:, 1], s=14, c="#1f77b4", label="true position", zorder=4)
        ax.scatter(est[:, 0], est[:, 1], s=20, marker="x", c="#ff7f0e", linewidths=1.2,
                   label=f"estimate ({cfg.algorithm})", zorder=5)
        self._stations(ax, scale)
        ax.set_title(f"{shown} of {len(r.test_points)} test points - {cfg.algorithm}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        self.map_tab.canvas.draw_idle()

    def draw_rssi(self):
        if self.result is None:
            return
        r, fig = self.result, self.rssi_tab.figure
        cfg = r.cfg
        scale, unit = length_unit(cfg)
        s = self.rssi_station.currentIndex()
        q = self.rssi_quantity.currentText()
        received = r.scan.received[:, s]
        if q == "Mean RSSI":
            values, label, cmap = r.scan.rssi[:, s], "mean reported RSSI [dBm]", "viridis"
        else:
            values, label, cmap = received / r.scan.frames * 100, "frames received [%]", "RdYlGn"
        sc = r.scenario
        z = np.ma.masked_where((received == 0) & (q != "Frames received"),
                               values).reshape(len(sc.grid_y), len(sc.grid_x))
        fig.clear()
        ax = fig.add_subplot()
        ax.set_facecolor("#e6e6e6")
        mesh = ax.pcolormesh(sc.grid_x / scale, sc.grid_y / scale, z, shading="nearest", cmap=cmap)
        if q == "Frames received":
            mesh.set_clim(0, 100)
        elif z.count():
            mesh.set_clim(*np.percentile(z.compressed(), [1, 99]))   # one cell next to a station would dominate
        fig.colorbar(mesh, ax=ax, label=label, shrink=0.85)
        self._frame(ax, cfg, scale, unit)
        self._stations(ax, scale, highlight=s)
        note = " (grey: no frame received)" if q != "Frames received" and (received == 0).any() else ""
        ax.set_title(f"Scan from station S{s + 1}: {label}{note}")
        self.rssi_tab.canvas.draw_idle()

    def draw_error(self):
        r, fig = self.result, self.error_tab.figure
        cfg = r.cfg
        scale, unit = length_unit(cfg)
        fig.clear()
        ax, cdf = fig.subplots(1, 2, width_ratios=[1.15, 1])
        self._frame(ax, cfg, scale, unit)
        err = r.errors[cfg.algorithm]
        pts = ax.scatter(r.test_points[:, 0] / scale, r.test_points[:, 1] / scale, c=err / scale,
                         s=22, cmap="magma_r", vmin=0, vmax=np.percentile(err, 95) / scale)
        fig.colorbar(pts, ax=ax, label=f"error [{unit}]", shrink=0.8)
        self._stations(ax, scale)
        ax.set_title(f"Error at each test point - {cfg.algorithm}")
        for alg in ALGORITHMS:
            e = np.sort(r.errors[alg]) / scale
            frac = np.arange(1, len(e) + 1) / len(e)
            bold = alg == cfg.algorithm
            cdf.plot(e, frac, color=COLORS[alg], lw=2.4 if bold else 1.3,
                     label=f"{alg}: mean {fmt_len(r.errors[alg].mean())}")
        for level in (0.5, 0.9):
            cdf.axhline(level, color="0.6", ls=":", lw=1)
        cdf.set_xlim(left=0)
        cdf.set_ylim(0, 1)
        cdf.set_xlabel(f"localization error [{unit}]")
        cdf.set_ylabel("fraction of test points")
        cdf.set_title("Error distribution (CDF)")
        cdf.grid(alpha=0.3)
        cdf.legend(loc="lower right")
        self.error_tab.canvas.draw_idle()

    def _show_sweep(self, payload):
        cfg, param, (xs, mean, p90) = payload
        self.pending.setText("")
        fig = self.sweep_tab.figure
        fig.clear()
        ax = fig.add_subplot()
        for alg in ALGORITHMS:
            ax.plot(xs, mean[alg], "o-", color=COLORS[alg], lw=2, label=f"{alg} - mean")
            ax.plot(xs, p90[alg], "--", color=COLORS[alg], lw=1.2, alpha=0.8, label=f"{alg} - 90th pct")
        if param in LOG_AXIS:
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" for x in xs])
            ax.minorticks_off()
        if param == "Zone size":
            ax.set_yscale("log")
        ax.set_xlabel(f"{param} [{UNITS[param]}]")
        ax.set_ylabel("localization error [m]")
        e = cfg.env
        model = "two-ray" if e.path_loss_model == "Two-ray" else f"n {e.path_loss_exponent:g}"
        env = f"{model}, σ {e.shadowing_sigma:g} dB"
        if param == "Zone size":
            nodes = f"{len(grid_axis(cfg.width, cfg.grid_spacing))} × {len(grid_axis(cfg.height, cfg.grid_spacing))}"
            setup = f"{nodes} grid nodes scaled with the zone"
        else:
            setup = f"{cfg.width:g} × {cfg.height:g} m, grid {self._spacing_text(cfg)}"
        ax.set_title(f"Error vs {param.lower()} - {env}, {setup}, SF{cfg.radio.spreading_factor}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(ncols=2, fontsize=8, loc="best")
        self.sweep_tab.canvas.draw_idle()
        self.tabs.setCurrentWidget(self.sweep_tab)

    # ---------- export ----------
    def export_png(self):
        tab = self.tabs.currentWidget()
        name = self.tabs.tabText(self.tabs.currentIndex()).split(" ")[0].lower()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save figure", f"{name}.png", "PNG (*.png)")
        if path:
            tab.figure.savefig(path, dpi=150)

    def export_csv(self):
        if self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export fingerprint map",
                                                        "fingerprint_map.csv", "CSV (*.csv)")
        if not path:
            return
        r = self.result
        cfg = r.cfg
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            fh.write(f"# area {cfg.width:g} x {cfg.height:g} m, grid {self._spacing_text(cfg)}, "
                     f"SF{cfg.radio.spreading_factor}, {cfg.scan_frames} frames per station, "
                     f"{cfg.env.path_loss_model} model, station antennas {cfg.station_height:g} m, "
                     f"receiver {cfg.receiver_height:g} m\n")
            for i, (x, y) in enumerate(r.scenario.stations):
                fh.write(f"# station S{i + 1}: x={x:.2f} m, y={y:.2f} m\n")
            n = len(r.scenario.stations)
            w.writerow(["x_m", "y_m"] + [f"rssi_s{i + 1}" for i in range(n)]
                       + [f"frames_s{i + 1}" for i in range(n)])
            for p, rs, rc in zip(r.scenario.grid, r.scan.rssi, r.scan.received):
                w.writerow([f"{p[0]:.2f}", f"{p[1]:.2f}"] + [f"{v:.2f}" for v in rs] + [int(v) for v in rc])


def main():
    app = QtWidgets.QApplication(sys.argv)
    # The colours of the panel and the plots are designed for a light theme; keep it in Windows dark mode too.
    try:
        app.styleHints().setColorScheme(QtCore.Qt.ColorScheme.Light)
    except AttributeError:      # Qt older than 6.8
        pass
    app.setStyle("Fusion")
    app.setPalette(app.style().standardPalette())
    app.setFont(QtGui.QFont("Segoe UI", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
