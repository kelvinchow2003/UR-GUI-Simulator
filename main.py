#!/usr/bin/env python3
"""
main.py — UR GUI Simulator entry point
==================================================================
Bootstraps the PySide6 application:

    * High-DPI + platform setup
    * Root logging wired to a Qt-friendly handler
    * Global exception hook (so a crash in a slot doesn't vanish)
    * Loads the full :class:`ui.main_window.MainWindow` when the UI
      modules are present; otherwise falls back to a small diagnostic
      window that exercises :class:`robot.ur_bridge.URBridge` in
      SIMULATED mode — letting you validate the core framework before
      the remaining panels are built.

Run:
    python main.py [--ip 192.168.1.10] [--model UR10e] [--log DEBUG]
==================================================================
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import traceback
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPlainTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QMessageBox,
)

from robot.ur_bridge import URBridge, ConnectionConfig, RobotState
from robot.ur_models import MODEL_NAMES, DEFAULT_MODEL

APP_NAME = "UR GUI Simulator"
APP_VERSION = "0.1.0"
log = logging.getLogger("ur_gui")


# ===========================================================================
#  CLI + logging
# ===========================================================================
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=APP_NAME)
    p.add_argument("--ip", default="192.168.1.100", help="Robot IP address")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=MODEL_NAMES,
                   help="UR model")
    p.add_argument("--log", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Console log level")
    p.add_argument("--connect", action="store_true",
                   help="Attempt to connect on startup")
    return p.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def install_excepthook() -> None:
    """Log uncaught exceptions instead of silently dropping them."""
    def _hook(exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log.critical("Uncaught exception:\n%s", text)
        try:
            QMessageBox.critical(None, "Unexpected error", text)
        except Exception:                                  # noqa: BLE001
            pass
    sys.excepthook = _hook


# ===========================================================================
#  Fallback diagnostic window (used until ui/main_window.py exists)
# ===========================================================================
class DiagnosticWindow(QMainWindow):
    """
    Minimal window proving the bridge + threading model work end-to-end.
    Streams simulated joint feedback and lets you fire a couple of commands.
    """

    def __init__(self, bridge: URBridge):
        super().__init__()
        self.bridge = bridge
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION} — core diagnostic")
        self.resize(720, 480)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        rtde = "available" if URBridge.rtde_available() else "NOT installed (socket/sim fallback)"
        root.addWidget(QLabel(
            f"<b>Core framework online.</b> ur_rtde: {rtde}.<br>"
            "The full UI (ui/main_window.py) will replace this window."
        ))

        row = QHBoxLayout()
        self.model_box = QComboBox()
        self.model_box.addItems(MODEL_NAMES)
        self.model_box.setCurrentText(bridge.config.model_name)
        self.model_box.currentTextChanged.connect(bridge.set_model)
        row.addWidget(QLabel("Model:"))
        row.addWidget(self.model_box)

        btn_connect = QPushButton("Connect")
        btn_connect.clicked.connect(bridge.connect_robot)
        btn_home = QPushButton("Move Home (MoveJ)")
        btn_home.clicked.connect(self._move_home)
        btn_estop = QPushButton("E-STOP")
        btn_estop.setStyleSheet("background:#c0392b;color:white;font-weight:bold")
        btn_estop.clicked.connect(bridge.emergency_stop)
        btn_free = QPushButton("Freedrive")
        btn_free.setCheckable(True)
        btn_free.toggled.connect(bridge.set_freedrive)
        for w in (btn_connect, btn_home, btn_free, btn_estop):
            row.addWidget(w)
        root.addLayout(row)

        self.state_label = QLabel("waiting for state…")
        self.state_label.setStyleSheet("font-family:Consolas,monospace")
        root.addWidget(self.state_label)

        self.console = QPlainTextEdit(readOnly=True)
        self.console.setMaximumBlockCount(500)
        root.addWidget(self.console, 1)

        bridge.state_updated.connect(self._on_state)
        bridge.log.connect(self._on_log)

    def _move_home(self) -> None:
        import numpy as np
        self.bridge.move_j([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])

    def _on_state(self, st: RobotState) -> None:
        q = "  ".join(f"{v:+.3f}" for v in st.q_actual)
        self.state_label.setText(
            f"[{st.transport.value:9s}] mode={st.mode.value:11s} "
            f"safety={st.safety.value:14s} ping={st.ping_ms:5.1f}ms\n"
            f"q(rad) = {q}"
        )

    def _on_log(self, level: str, msg: str) -> None:
        self.console.appendPlainText(f"{level:5s} | {msg}")


# ===========================================================================
#  Window loader
# ===========================================================================
def build_main_window(bridge: URBridge, args) -> QMainWindow:
    """Load the full UI if available, else the diagnostic fallback."""
    try:
        from ui.main_window import MainWindow          # noqa: WPS433 (lazy)
        log.info("Loading full MainWindow UI.")
        return MainWindow(bridge)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith(("ui", "gui", "cad")):
            log.warning("Full UI not present yet (%s) — using diagnostic window.", exc)
            return DiagnosticWindow(bridge)
        raise                                           # a real missing dep
    except Exception:                                   # noqa: BLE001
        log.exception("MainWindow failed to load — using diagnostic window.")
        return DiagnosticWindow(bridge)


# ===========================================================================
#  main
# ===========================================================================
def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log)
    log.info("%s %s starting…", APP_NAME, APP_VERSION)

    # Qt6 enables high-DPI scaling by default; set attributes that remain useful.
    QApplication.setApplicationName(APP_NAME)
    QApplication.setApplicationVersion(APP_VERSION)
    QApplication.setOrganizationName("UR GUI Simulator")

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setStyle("Fusion")

    install_excepthook()

    # Let Ctrl-C in a terminal actually quit the Qt loop.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    config = ConnectionConfig(ip=args.ip, model_name=args.model)
    bridge = URBridge(config)
    bridge.start()

    window = build_main_window(bridge, args)
    window.show()

    if args.connect:
        QTimer.singleShot(300, bridge.connect_robot)

    # ensure the worker thread is stopped on quit
    app.aboutToQuit.connect(bridge.shutdown)

    log.info("Event loop running.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
