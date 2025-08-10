import cv2
import numpy as np
from dataclasses import dataclass
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
            return cv2.addWeighted(frame, 1 + amt, blur, -amt, 0)
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
                out[r:r+1, :, :] *= (0.8 + 0.2 * np.random.rand())
            return out
        if name == "Motion Blur":
            k = max(3, p.strength)
            if k % 2 == 0:
                k += 1
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k//2, :] = 1.0 / k
            return cv2.filter2D(frame, -1, kernel)
        return frame
