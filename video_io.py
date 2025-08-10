import time
import threading
import cv2
import numpy as np
from typing import Optional, Tuple
from PySide6 import QtCore
from effects import Effects, EffectParams

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
            time.sleep(min(0.05, 1.0 / max(1.0, fps)))

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
