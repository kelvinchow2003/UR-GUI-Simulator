"""
ui/panels/editor_panel.py
==================================================================
Dual-mode Code Editor dock.

    Tab 1 — URScript editor (native .script).
    Tab 2 — Python (ur_rtde) editor.

Both are generated live from the visual Program via
:class:`robot.program.URScriptGenerator` / :class:`PythonGenerator`,
and either can be hand-edited, exported, or executed.

Syntax highlighting is a lightweight :class:`QSyntaxHighlighter` (no
QScintilla dependency required — it is used automatically if present).
"execute" runs the URScript on the robot through the bridge, guarded by
a confirmation prompt.
==================================================================
"""
from __future__ import annotations

import re

from PySide6.QtCore import Qt, QRegularExpression
from PySide6.QtGui import (
    QSyntaxHighlighter, QTextCharFormat, QColor, QFont, QTextCursor,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QPlainTextEdit, QPushButton,
    QFileDialog, QMessageBox, QLabel,
)

from robot.program import Program, URScriptGenerator, PythonGenerator
from robot.ur_bridge import URBridge


# ===========================================================================
#  Syntax highlighter
# ===========================================================================
class _Highlighter(QSyntaxHighlighter):
    URSCRIPT_KW = [
        "def", "end", "if", "else", "elif", "while", "for", "return", "global",
        "movej", "movel", "movep", "movec", "servoj", "speedj", "stopj", "stopl",
        "set_tcp", "set_digital_out", "set_standard_digital_out", "sleep",
        "get_standard_digital_in", "freedrive_mode", "end_freedrive_mode",
        "sync", "halt", "thread", "socket_open", "popup",
    ]
    PYTHON_KW = [
        "import", "from", "def", "class", "return", "if", "else", "elif",
        "while", "for", "in", "try", "except", "finally", "with", "as",
        "True", "False", "None", "and", "or", "not", "lambda", "pass",
    ]

    def __init__(self, document, language: str):
        super().__init__(document)
        self.rules = []
        kw_fmt = self._fmt("#5e81ac", bold=True)
        num_fmt = self._fmt("#b48ead")
        str_fmt = self._fmt("#a3be8c")
        com_fmt = self._fmt("#6c7480", italic=True)
        fn_fmt = self._fmt("#88c0d0")

        keywords = self.URSCRIPT_KW if language == "urscript" else self.PYTHON_KW
        for kw in keywords:
            self.rules.append((QRegularExpression(rf"\b{kw}\b"), kw_fmt))
        self.rules.append((QRegularExpression(r"\b[0-9]+\.?[0-9]*\b"), num_fmt))
        self.rules.append((QRegularExpression(r"\b[A-Za-z_]\w*(?=\s*\()"), fn_fmt))
        self.rules.append((QRegularExpression(r"\"[^\"]*\"|'[^']*'"), str_fmt))
        self.rules.append((QRegularExpression(r"#[^\n]*"), com_fmt))

    @staticmethod
    def _fmt(color: str, bold=False, italic=False) -> QTextCharFormat:
        f = QTextCharFormat()
        f.setForeground(QColor(color))
        if bold:
            f.setFontWeight(QFont.Weight.Bold)
        if italic:
            f.setFontItalic(True)
        return f

    def highlightBlock(self, text: str) -> None:      # noqa: N802 (Qt override)
        for pattern, fmt in self.rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt)


def _make_editor() -> QPlainTextEdit:
    ed = QPlainTextEdit()
    font = QFont("Consolas")
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSize(10)
    ed.setFont(font)
    ed.setTabStopDistance(4 * ed.fontMetrics().horizontalAdvance(" "))
    ed.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    return ed


# ===========================================================================
#  Editor panel
# ===========================================================================
class EditorPanel(QWidget):
    def __init__(self, bridge: URBridge, program: Program, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.program = program
        self._urgen = URScriptGenerator()
        self._pygen = PythonGenerator()
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)

        # toolbar
        bar = QHBoxLayout()
        gen = QPushButton("⟳ Generate from Program")
        gen.clicked.connect(self.regenerate)
        self.export_btn = QPushButton("Export current tab…")
        self.export_btn.clicked.connect(self._export)
        self.exec_btn = QPushButton("Execute on Robot")
        self.exec_btn.clicked.connect(self._execute)
        bar.addWidget(gen)
        bar.addWidget(self.export_btn)
        bar.addStretch(1)
        bar.addWidget(self.exec_btn)
        root.addLayout(bar)

        self.bridge.connected_changed.connect(self._on_connected)
        self._on_connected(self.bridge.state.connected)

        # tabs
        self.tabs = QTabWidget()
        self.ur_edit = _make_editor()
        self.py_edit = _make_editor()
        self._ur_hl = _Highlighter(self.ur_edit.document(), "urscript")
        self._py_hl = _Highlighter(self.py_edit.document(), "python")
        self.tabs.addTab(self.ur_edit, "URScript (.script)")
        self.tabs.addTab(self.py_edit, "Python (ur_rtde)")
        root.addWidget(self.tabs, 1)

        self.regenerate()

    def _on_connected(self, connected: bool) -> None:
        if connected:
            self.exec_btn.setText("▶ Execute on Robot")
            self.exec_btn.setStyleSheet("background:#a3be8c;font-weight:bold")
            self.exec_btn.setToolTip("Push the URScript tab to the connected robot.")
        else:
            self.exec_btn.setText("▶ Execute (offline — logged)")
            self.exec_btn.setStyleSheet("background:#5a5f6a;color:#e8e8e8")
            self.exec_btn.setToolTip(
                "Not connected — the URScript is logged, not run. "
                "Connect a robot to execute for real.")

    # ---- generation -------------------------------------------------------
    def regenerate(self) -> None:
        ip = self.bridge.config.ip
        self.ur_edit.setPlainText(self._urgen.generate(self.program))
        self.py_edit.setPlainText(self._pygen.generate(self.program, ip))

    # ---- export -----------------------------------------------------------
    def _export(self) -> None:
        is_ur = self.tabs.currentIndex() == 0
        text = (self.ur_edit if is_ur else self.py_edit).toPlainText()
        flt = "URScript (*.script)" if is_ur else "Python (*.py)"
        default = "program.script" if is_ur else "program.py"
        path, _ = QFileDialog.getSaveFileName(self, "Export", default, flt)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            QMessageBox.information(self, "Export", f"Written to:\n{path}")

    # ---- execute ----------------------------------------------------------
    def _execute(self) -> None:
        if self.tabs.currentIndex() != 0:
            QMessageBox.information(
                self, "Execute",
                "Only the URScript tab can be pushed to the controller "
                "directly.\nRun the Python tab from a terminal with ur_rtde "
                "installed.")
            return
        script = self.ur_edit.toPlainText().strip()
        if not script:
            return
        # Only a live connection needs the safety gate; offline it's logged.
        if self.bridge.state.connected:
            reply = QMessageBox.warning(
                self, "Execute URScript",
                "Send this URScript to the PHYSICAL robot now?\n"
                "Confirm the workspace is clear.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.bridge.run_script(script)
