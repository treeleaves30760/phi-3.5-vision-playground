import sys
import requests
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QLabel)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

class ImageLoader(QThread):
    image_loaded = pyqtSignal(QPixmap)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            pixmap = QPixmap()
            pixmap.loadFromData(response.content)
            self.image_loaded.emit(pixmap)
        except Exception as e:
            print(f"Error loading image: {str(e)}")

class Phi35VisionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phi-3.5 Vision Interaction GUI")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.create_widgets()

    def create_widgets(self):
        # Prompt input
        self.prompt_label = QLabel("Enter your prompt:")
        self.prompt_input = QTextEdit()
        self.prompt_input.setMaximumHeight(100)
        self.layout.addWidget(self.prompt_label)
        self.layout.addWidget(self.prompt_input)

        # Image URL input
        self.url_label = QLabel("Enter image URLs (one per line):")
        self.url_input = QTextEdit()
        self.url_input.setMaximumHeight(100)
        self.layout.addWidget(self.url_label)
        self.layout.addWidget(self.url_input)

        # Submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit_request)
        self.layout.addWidget(self.submit_button)

        # Response display
        self.response_label = QLabel("Response:")
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.layout.addWidget(self.response_label)
        self.layout.addWidget(self.response_text)

        # Image preview
        self.image_preview_label = QLabel("Image Preview:")
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMaximumHeight(200)
        self.layout.addWidget(self.image_preview_label)
        self.layout.addWidget(self.image_preview)

    def submit_request(self):
        prompt = self.prompt_input.toPlainText().strip()
        image_urls = [url.strip() for url in self.url_input.toPlainText().split("\n") if url.strip()]

        if not prompt:
            self.show_response("Please enter a prompt.")
            return

        if not image_urls:
            self.show_response("Please enter at least one image URL.")
            return

        self.show_response("Sending request to server...")

        try:
            response = requests.post("http://localhost:5000/submit", json={
                "prompt": prompt,
                "image_urls": image_urls
            })
            response.raise_for_status()
            self.show_response(response.json()["response"])
            self.load_image_preview(image_urls[0])
        except requests.exceptions.RequestException as e:
            self.show_response(f"Error communicating with the server: {str(e)}")

    def show_response(self, text):
        self.response_text.setPlainText(text)

    def load_image_preview(self, url):
        self.image_loader = ImageLoader(url)
        self.image_loader.image_loaded.connect(self.update_image_preview)
        self.image_loader.start()

    def update_image_preview(self, pixmap):
        scaled_pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_preview.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = Phi35VisionGUI()
    gui.show()
    sys.exit(app.exec())