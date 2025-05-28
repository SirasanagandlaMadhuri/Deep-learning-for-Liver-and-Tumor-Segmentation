import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt5.QtGui import QFont, QDesktopServices
from fastai.vision.all import *
from fastai.learner import load_learner
from PIL import Image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pathlib
from pathlib import Path
import warnings
import nibabel as nib  

# ---------------------- MODEL LOADING ----------------------
def label_func(x): return x
def cust_foreground_acc(inp, targ): return 1.0
def compute_dice_score(preds, targets, smooth=1e-5, axis=1): return 1.0
def get_x(x): return x
def _to_float(x): return 0.0
def _to_safe(x): return 0.0
def _fmt(x): return "0.0000"

class ExtendedMetrics:
    def __init__(self, *args, **kwargs):  
        pass
    def __call__(self, event_name):      
        pass
    def before_fit(self):
        pass
    def after_batch(self):
        pass
    def after_epoch(self):
        pass
    def after_fit(self):
        pass

  
def windowed(img, w, l):
    px_min = l - w // 2
    px_max = l + w // 2
    img = np.clip(img, px_min, px_max)
    return (img - px_min) / (px_max - px_min)

def freqhist_bins(img, n_bins=100):
    sorted_vals = np.sort(img.flatten())
    t = np.concatenate([
        [0.001],
        np.arange(n_bins) / n_bins + 1 / (2 * n_bins),
        [0.999]
    ])
    t = (len(sorted_vals) * t).astype(int)
    return np.unique(sorted_vals[np.clip(t, 0, len(sorted_vals)-1)])

def hist_scaled(img, brks=None):
    if brks is None:
        brks = freqhist_bins(img)
    ys = np.linspace(0., 1., len(brks))
    flat = img.flatten()
    interp = np.interp(flat, brks, ys)
    return interp.reshape(img.shape).clip(0., 1.)

def to_nchan(img):
    img = img.astype(np.float32)
    win1 = windowed(img, 150, 30)
    win2 = windowed(img, 200, 60)
    hist = hist_scaled(img)
    stacked = np.stack([win1, win2, hist], axis=-1)
    return (stacked * 255).astype(np.uint8)


def load_model():
    MODEL_PATH = r"C:\\Users\\ramya\\Downloads\\sdp_best_seg_model.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learn = load_learner(MODEL_PATH, cpu=True)
    learn.model.eval()
    return learn

def predict_single_slice(learn, slice_rgb):
    img = PILImage.create(slice_rgb)
    test_dl = learn.dls.test_dl([img])
    preds, _ = learn.get_preds(dl=test_dl)
    pred_mask = torch.argmax(preds, dim=1)[0].numpy()
    return pred_mask

class PredictionThread(QThread):
    finished = pyqtSignal(np.ndarray)

    def __init__(self, learn, image_data):
        super().__init__()
        self.learn = learn
        self.image_data = image_data

    def run(self):
        pred = predict_single_slice(self.learn, self.image_data)
        self.finished.emit(pred)

# ---------------------- GUI CLASS ----------------------

class LiverSegmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Liver + Tumor Segmentation")
        if isinstance(pathlib.Path(), pathlib.WindowsPath):
            pathlib.PosixPath = pathlib.WindowsPath

        self.resize(900, 700)
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        title = QLabel("DEEP LEARNING FOR LIVER SEGMENTATION")
        title.setFont(QFont('Arial', 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Senior Design Project")
        subtitle.setFont(QFont('Arial', 14))
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        spacer = QLabel("")
        spacer.setFixedHeight(150)
        layout.addWidget(spacer)

        self.label = QLabel("Select an image:")
        self.label.setFont(QFont('Arial', 12, QFont.Bold))
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        spacer2 = QLabel("")
        spacer2.setFixedHeight(45)
        layout.addWidget(spacer2)

        button_style = """
            QPushButton {
                background-color: #87CEEB;
                color: black;
                font-weight: bold;
                padding: 10px;
                border-radius: 10px;
            }
            QPushButton:pressed {
                background-color: #00BFFF;
            }
        """

        self.select_button = QPushButton("Browse Image")
        self.select_button.setStyleSheet(button_style)
        self.select_button.clicked.connect(self.load_image)
        layout.addWidget(self.select_button)

        self.predict_button = QPushButton("Predict")
        self.predict_button.setStyleSheet(button_style)
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button)

                # Bottom container for contact info
        self.bottom_container = QVBoxLayout()
        self.bottom_container.setAlignment(Qt.AlignCenter)

        self.contact_button = QPushButton("Contact Us")
        self.contact_button.setStyleSheet(button_style)
        self.contact_button.clicked.connect(self.toggle_contacts)
        self.bottom_container.addWidget(self.contact_button)

        self.contact_buttons = []
        # Example usage
        self.create_contact_button("Madhuri Sirasanagandla", "7416759083", "https://www.linkedin.com/in/madhuri-sirasanagandla-988957232/", self.bottom_container)
        self.create_contact_button("Amarnath Chigurupati", "7995016856", "https://www.linkedin.com/in/amarnath-chigurupati-3a1b5b238/", self.bottom_container)
        self.create_contact_button("Ankit Kommalapati", "9885081606", "https://www.linkedin.com/in/ankitkommalapati/", self.bottom_container)

        for btn_group in self.contact_buttons:
            for widget in btn_group:
                widget.hide()


        # Add stretch and then the bottom contact container to main layout
        layout.addStretch(1)
        layout.addLayout(self.bottom_container)


        self.setLayout(layout)
        self.image_data = None
        self.learn = load_model()
        self.plot_canvas = None
        self.axes = None
        self.go_back_button = None

    def create_contact_button(self, name, phone, linkedin, layout):
        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignCenter)

        contact_row = QWidget()
        row_layout = QHBoxLayout(contact_row)
        row_layout.setAlignment(Qt.AlignCenter)

        name_label = QLabel(f'<a href="{linkedin}">{name.strip()}</a>')
        name_label.setFont(QFont('Arial', 11))
        name_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        name_label.setOpenExternalLinks(True)
        name_label.setStyleSheet("color: black;")

        phone_label = QLabel(phone.strip())
        phone_label.setFont(QFont('Arial', 11))
        phone_label.setStyleSheet("color: black;")

        link_icon = QLabel("🔗")
        link_icon.setFont(QFont('Arial', 11))

        # Add widgets centered in horizontal layout
        row_layout.addWidget(name_label)
        row_layout.addSpacing(15)
        row_layout.addWidget(phone_label)
        row_layout.addSpacing(10)
        row_layout.addWidget(link_icon)

        vbox.addWidget(contact_row)
        layout.addLayout(vbox)

        self.contact_buttons.append((name_label, phone_label, link_icon))



    def toggle_contacts(self):
        for btn_group in self.contact_buttons:
            for widget in btn_group:
                widget.setVisible(not widget.isVisible())

    
    def load_image(self):
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "JPEG Files (*.jpg *.jpeg)")
            if not file_path:
                return

            image = Image.open(file_path)
            rgb_array = np.array(image)

            if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
                self.label.setText("Invalid RGB image. Please use .jpg ")
                return

            self.image_data = rgb_array
            self.label.setText(f"Loaded: {file_path.split('/')[-1]}")


    def predict(self):
        if self.image_data is None:
            self.label.setText("Please load an image first!")
            return

        self.label.setText("Predicting... Please wait.")
        self.thread = PredictionThread(self.learn, self.image_data)
        self.thread.finished.connect(self.display_result)
        self.thread.start()

        # Hide contact section
        self.contact_button.hide()
        for btn_group in self.contact_buttons:
            for widget in btn_group:
                widget.hide()


    def display_result(self, prediction_mask):
        self.label.setText("Prediction complete.")
        self.select_button.hide()
        self.predict_button.hide()
        QTimer.singleShot(1000, lambda: self.label.hide())

        if self.plot_canvas is None:
            self.plot_canvas = FigureCanvas(Figure(figsize=(9, 3)))
            self.axes = self.plot_canvas.figure.subplots(1, 3)
            self.plot_container = QWidget()
            h_layout = QVBoxLayout()
            h_layout.setAlignment(Qt.AlignCenter)
            h_layout.addWidget(self.plot_canvas)
            self.plot_container.setLayout(h_layout)
            self.layout().addWidget(self.plot_container)
        else:
            for ax in self.axes:
                ax.clear()
            self.plot_canvas.show()
            self.plot_container.show()

        self.axes[0].imshow(self.image_data)
        self.axes[0].set_title("RGB INPUT CT SCAN")
        self.axes[0].axis('off')

        gray_img = np.mean(self.image_data, axis=2).astype(np.uint8)
        self.axes[1].imshow(gray_img, cmap='gray')
        self.axes[1].set_title("Grayscale Structure View")
        self.axes[1].axis('off')

        self.axes[2].imshow(prediction_mask, cmap='viridis', interpolation='nearest')
        self.axes[2].set_title("PREDICTED OUTPUT")
        self.axes[2].axis('off')

        self.plot_canvas.draw()

        if not self.go_back_button:
            self.go_back_button = QPushButton("Go Back to Predict Again")
            self.go_back_button.setStyleSheet("""
                QPushButton {
                    background-color: #90EE90;
                    color: black;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 10px;
                }
                QPushButton:pressed {
                    background-color: #32CD32;
                }
            """)
            self.go_back_button.clicked.connect(self.reset_ui)
            self.layout().addWidget(self.go_back_button)
        else:
            self.go_back_button.show()

    


    def reset_ui(self):
        self.label.setText("Select an image:")
        self.image_data = None
        if self.plot_canvas:
            self.plot_canvas.hide()
        if hasattr(self, 'plot_container'):
            self.plot_container.hide()
        self.select_button.show()
        self.predict_button.show()
        self.label.show()
        if self.go_back_button:
            self.go_back_button.hide()

        # Show contact section again
        self.contact_button.show()
        for btn_group in self.contact_buttons:
            for widget in btn_group:
                widget.hide()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LiverSegmentationApp()
    window.show()
    sys.exit(app.exec_())