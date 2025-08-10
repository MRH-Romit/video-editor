
import os
import sys
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

def cvimg_to_qimage(img_bgr: np.ndarray) -> QtGui.QImage:
    if img_bgr is None:
        return QtGui.QImage()
    if len(img_bgr.shape) == 2:
        h, w = img_bgr.shape
        bytes_per_line = w
        qimg = QtGui.QImage(img_bgr.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        return qimg.copy()
    h, w, ch = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bytes_per_line = ch * w
    qimg = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    return qimg.copy()


def fit_to_label(image: QtGui.QImage, label: QtWidgets.QLabel) -> QtGui.QPixmap:
    if image.isNull():
        return QtGui.QPixmap()
    target = label.size() * label.devicePixelRatioF()
    pm = QtGui.QPixmap.fromImage(image)
    return pm.scaled(int(target.width()), int(target.height()), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)


@dataclass
class EffectParams:
    name: str = "None"
    intensity: float = 0.5
    strength: int = 3
    hue: int = 0
    saturation: float = 1.0
    value: float = 1.0
    pixel_size: int = 8
    sharpen: float = 1.0


class Effects:
    @staticmethod
    def apply(frame: np.ndarray, p: EffectParams) -> np.ndarray:
        if frame is None:
            return frame
        name = p.name
        if name == "None":
            return frame
        if name == "Grayscale":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if name == "Canny Edge":
            low = int(50 + 200 * (1 - p.intensity))
            high = int(150 + 400 * p.intensity)
            edges = cv2.Canny(frame, low, high)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        if name == "Gaussian Blur":
            k = max(1, p.strength)
            if k % 2 == 0:
                k += 1
            return cv2.GaussianBlur(frame, (k, k), 0)
        if name == "Sharpen":
            amt = max(0.0, min(3.0, p.sharpen))
            blur = cv2.GaussianBlur(frame, (0, 0), 2.0)
            out = cv2.addWeighted(frame, 1 + amt, blur, -amt, 0)
            return out
        if name == "Sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]], dtype=np.float32)
            sep = cv2.transform(frame, kernel)
            return np.clip(sep, 0, 255).astype(np.uint8)
        if name == "Cartoon":
            color = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 2)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return cv2.bitwise_and(color, edges)
        if name == "Pixelate":
            step = max(2, p.pixel_size)
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (w // step, h // step), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        if name == "HSV Adjust":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            h = (h + (p.hue / 2)) % 180
            s = np.clip(s * p.saturation, 0, 255)
            v = np.clip(v * p.value, 0, 255)
            hsv = cv2.merge([h, s, v]).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if name == "Glitch":
            h, w = frame.shape[:2]
            shift = int(max(1, w * 0.01 + p.intensity * w * 0.02))
            ch = list(cv2.split(frame))
            ch[2] = np.roll(ch[2], shift, axis=1)
            ch[0] = np.roll(ch[0], -shift // 2, axis=0)
            out = cv2.merge(ch)
            for r in range(0, h, 4):
                out[r:r+1, :, :] = out[r:r+1, :, :] * (0.8 + 0.2 * np.random.rand())
            return out
        if name == "Motion Blur":
            k = max(3, p.strength)
            if k % 2 == 0:
                k += 1
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k//2, :] = 1.0 / k
            return cv2.filter2D(frame, -1, kernel)
        return frame


class VideoReader(QtCore.QThread):
    frame_signal = QtCore.Signal(np.ndarray, float)

    def __init__(self):
        super().__init__()
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.paused = False
        self.filepath: Optional[str] = None
        self._lock = threading.Lock()

    def open(self, path: str) -> Tuple[bool, str]:
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        self.filepath = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            return False, "Failed to open video."
        self.running = True
        self.paused = False
        if not self.isRunning():
            self.start()
        return True, "OK"

    def run(self):
        while True:
            if not self.running or self.cap is None:
                time.sleep(0.03)
                continue
            if self.paused:
                time.sleep(0.02)
                continue
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.frame_signal.emit(frame, fps)
            delay = 1.0 / max(1.0, fps)
            time.sleep(min(0.05, delay))

    def stop(self):
        self.running = False
        self.paused = True
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None


class ExportWorker(QtCore.QThread):
    progress = QtCore.Signal(int)
    finished_ok = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, path_in: str, path_out: str, params: EffectParams):
        super().__init__()
        self.path_in = path_in
        self.path_out = path_out
        self.params = params

    def run(self):
        cap = cv2.VideoCapture(self.path_in)
        if not cap.isOpened():
            self.failed.emit("Could not open input video")
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.path_out, fourcc, fps, (w, h))
        if not out.isOpened():
            self.failed.emit("Could not open output file for writing")
            cap.release()
            return
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                processed = Effects.apply(frame, self.params)
                out.write(processed)
                i += 1
                if total > 0:
                    self.progress.emit(int(100 * i / total))
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            cap.release()
            out.release()
        self.finished_ok.emit(self.path_out)


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
        self.current_frame: Optional[np.ndarray] = None
        self.current_processed: Optional[np.ndarray] = None
        self.loaded_path: Optional[str] = None

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

    def _labeled_slider(self, name: str, minv: int, maxv: int, val: int, layout: QtWidgets.QGridLayout, row: int) -> QtWidgets.QSlider:
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
                w = grid.itemAtPosition(i, j)
                if not w:
                    continue
                item = w.widget()
                if isinstance(item, QtWidgets.QLabel) and item.text() in show:
                    visible = show[item.text()]
                    item.setVisible(visible)
                    sld_item = grid.itemAtPosition(i, 1)
                    val_item = grid.itemAtPosition(i, 2)
                    if sld_item: sld_item.widget().setVisible(visible)
                    if val_item: val_item.widget().setVisible(visible)

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
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.mov *.avi *.mkv *.webm)")
        if path:
            self.load_video(path)

    def load_video(self, path: str):
        ok, msg = self.reader.open(path)
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Error", msg)
            return
        self.loaded_path = path
        self.lbl_status.setText(f"Loaded: {os.path.basename(path)}")

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
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Processed Video", self._suggest_out_path(self.loaded_path), "MP4 Video (*.mp4)")
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
