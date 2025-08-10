# main.py
import os
import sys
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from effects import cvimg_to_qimage, fit_to_label, Effects, EffectParams
from video_io import VideoReader, ExportWorker

class VideoFXStudio(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VideoFX Studio — Real‑time Effects")
        self.setAcceptDrops(True)
        self.resize(1200, 700)
        self.setStyleSheet(self._style())

        self.reader = VideoReader()
        self.reader.frame_signal.connect(self.on_frame)

        self.params = EffectParams()
        self.current_fps = 30.0
        self.current_frame: np.ndarray | None = None
        self.current_processed: np.ndarray | None = None
        self.loaded_path: str | None = None

        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main = QtWidgets.QVBoxLayout(central)
        topbar = QtWidgets.QHBoxLayout()
        main.addLayout(topbar)

        self.btn_open = QtWidgets.QPushButton("Open…")
        self.btn_save = QtWidgets.QPushButton("Save Processed…")
        self.btn_play = QtWidgets.QPushButton("Pause")
        self.lbl_status = QtWidgets.QLabel("Drop a video to start")
        self.lbl_fps = QtWidgets.QLabel("FPS: —")

        topbar.addWidget(self.btn_open)
        topbar.addWidget(self.btn_save)
        topbar.addWidget(self.btn_play)
        topbar.addStretch(1)
        topbar.addWidget(self.lbl_fps)
        topbar.addWidget(self.lbl_status)

        self.btn_open.clicked.connect(self.open_file)
        self.btn_save.clicked.connect(self.save_processed)
        self.btn_play.clicked.connect(self.toggle_play)

        views = QtWidgets.QHBoxLayout()
        main.addLayout(views, 1)

        self.view_orig = QtWidgets.QLabel("Original")
        self.view_proc = QtWidgets.QLabel("Processed")
        for v in (self.view_orig, self.view_proc):
            v.setAlignment(QtCore.Qt.AlignCenter)
            v.setMinimumSize(400, 300)
            v.setFrameShape(QtWidgets.QFrame.StyledPanel)
        views.addWidget(self.view_orig, 1)
        views.addWidget(self.view_proc, 1)

        panel = QtWidgets.QGroupBox("Effects")
        panel.setObjectName("Effects")
        main.addWidget(panel)
        grid = QtWidgets.QGridLayout(panel)

        self.cmb_effect = QtWidgets.QComboBox()
        self.cmb_effect.addItems([
            "None", "Grayscale", "Canny Edge", "Gaussian Blur", "Sharpen",
            "Sepia", "Cartoon", "Pixelate", "HSV Adjust", "Glitch", "Motion Blur"
        ])
        self.cmb_effect.currentTextChanged.connect(self.on_effect_change)

        self.sld_intensity = self._labeled_slider("Intensity", 0, 100, int(self.params.intensity * 100), grid, 1)
        self.sld_strength  = self._labeled_slider("Strength", 1, 31, self.params.strength, grid, 2)
        self.sld_pixel     = self._labeled_slider("Pixel Size", 2, 40, self.params.pixel_size, grid, 3)
        self.sld_sharpen   = self._labeled_slider("Sharpen Amt", 0, 300, int(self.params.sharpen * 100), grid, 4)
        self.sld_hue       = self._labeled_slider("Hue", -180, 180, self.params.hue, grid, 5)
        self.sld_sat       = self._labeled_slider("Saturation", 0, 300, int(self.params.saturation * 100), grid, 6)
        self.sld_val       = self._labeled_slider("Value", 0, 300, int(self.params.value * 100), grid, 7)

        grid.addWidget(QtWidgets.QLabel("Effect"), 0, 0)
        grid.addWidget(self.cmb_effect, 0, 1, 1, 3)

        self._update_control_visibility()

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        main.addWidget(self.progress)

    def _labeled_slider(self, name: str, minv: int, maxv: int, val: int,
                        layout: QtWidgets.QGridLayout, row: int) -> QtWidgets.QSlider:
        label = QtWidgets.QLabel(name)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(minv, maxv)
        slider.setValue(val)
        value_lbl = QtWidgets.QLabel(str(val))
        def on_change(v: int):
            value_lbl.setText(str(v))
            self._sync_params()
        slider.valueChanged.connect(on_change)
        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_lbl, row, 2)
        return slider

    def _update_control_visibility(self):
        name = self.cmb_effect.currentText()
        show = {
            "Intensity": name in {"Canny Edge", "Glitch"},
            "Strength": name in {"Gaussian Blur", "Motion Blur"},
            "Pixel Size": name == "Pixelate",
            "Sharpen Amt": name == "Sharpen",
            "Hue": name == "HSV Adjust",
            "Saturation": name == "HSV Adjust",
            "Value": name == "HSV Adjust",
        }
        grid = self.centralWidget().findChild(QtWidgets.QGroupBox, "Effects").layout()
        for i in range(grid.rowCount()):
            for j in range(3):
                item = grid.itemAtPosition(i, j)
                if not item:
                    continue
                w = item.widget()
                if isinstance(w, QtWidgets.QLabel) and w.text() in show:
                    vis = show[w.text()]
                    w.setVisible(vis)
                    sld_item = grid.itemAtPosition(i, 1)
                    val_item = grid.itemAtPosition(i, 2)
                    if sld_item: sld_item.widget().setVisible(vis)
                    if val_item: val_item.widget().setVisible(vis)

    def _sync_params(self):
        self.params.name = self.cmb_effect.currentText()
        self.params.intensity = self.sld_intensity.value() / 100.0
        self.params.strength = self.sld_strength.value()
        self.params.pixel_size = self.sld_pixel.value()
        self.params.sharpen = self.sld_sharpen.value() / 100.0
        self.params.hue = self.sld_hue.value()
        self.params.saturation = self.sld_sat.value() / 100.0
        self.params.value = self.sld_val.value() / 100.0
        self._update_control_visibility()

    def on_effect_change(self, _):
        self._sync_params()

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                if url.isLocalFile() and self._is_video_file(url.toLocalFile()):
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dropEvent(self, e: QtGui.QDropEvent) -> None:
        for url in e.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if self._is_video_file(path):
                    self.load_video(path)
                    break

    def _is_video_file(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.mov *.avi *.mkv *.webm)")
        if path:
            self.load_video(path)

    def load_video(self, path: str):
        ok, msg = self.reader.open(path)
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Error", msg)
            return
        self.loaded_path = path
        self.lbl_status.setText(f"Loaded: {os.path.basename(path)}")
        self.btn_play.setText("Pause")

    def toggle_play(self):
        if not self.reader.cap:
            return
        self.reader.paused = not self.reader.paused
        self.btn_play.setText("Play" if self.reader.paused else "Pause")

    @QtCore.Slot(np.ndarray, float)
    def on_frame(self, frame: np.ndarray, fps: float):
        self.current_fps = fps
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.current_frame = frame
        self.current_processed = Effects.apply(frame, self.params)
        q1 = cvimg_to_qimage(self.current_frame)
        q2 = cvimg_to_qimage(self.current_processed)
        self.view_orig.setPixmap(fit_to_label(q1, self.view_orig))
        self.view_proc.setPixmap(fit_to_label(q2, self.view_proc))

    def save_processed(self):
        if not self.loaded_path:
            QtWidgets.QMessageBox.information(self, "No video", "Load a video first.")
            return
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Processed Video", self._suggest_out_path(self.loaded_path),
            "MP4 Video (*.mp4)")
        if not out_path:
            return
        export_params = EffectParams(**self.params.__dict__)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.exporter = ExportWorker(self.loaded_path, out_path, export_params)
        self.exporter.progress.connect(self.progress.setValue)
        self.exporter.finished_ok.connect(self.on_export_done)
        self.exporter.failed.connect(self.on_export_failed)
        self.exporter.start()

    def on_export_done(self, path: str):
        self.progress.setVisible(False)
        QtWidgets.QMessageBox.information(self, "Done", f"Saved to:\n{path}")

    def on_export_failed(self, msg: str):
        self.progress.setVisible(False)
        QtWidgets.QMessageBox.critical(self, "Export failed", msg)

    def _suggest_out_path(self, inpath: str) -> str:
        root, _ = os.path.splitext(inpath)
        return root + "_processed.mp4"

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self.reader.stop()
        super().closeEvent(e)

    def _style(self) -> str:
        return """
        QWidget { font-family: Inter, Segoe UI, Arial; font-size: 14px; }
        QGroupBox { font-weight: 600; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px; margin-top: 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
        QPushButton { padding: 8px 14px; border-radius: 10px; border: 1px solid #cbd5e1; }
        QPushButton:hover { background: #f1f5f9; }
        QLabel#hint { color: #64748b; }
        QProgressBar { border: 1px solid #cbd5e1; border-radius: 10px; height: 16px; }
        QProgressBar::chunk { border-radius: 10px; }
        """

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = VideoFXStudio()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
