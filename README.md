# Phi-3.5 Vision

This project provides a graphical user interface (GUI) for interacting with the Phi-3.5 Vision model. It allows users to input prompts and image URLs, send requests to a Flask server running the model, and view the responses along with image previews.

## Features

- Text input for prompts
- Text input for image URLs (multiple URLs supported)
- Submit button to send requests to the server
- Response display area
- Image preview of the first entered URL

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/phi-3.5-vision-gui.git
   cd phi-3.5-vision-gui
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask server (ensure you have the `server.py` file set up and running):
   ```
   python server.py
   ```

2. In a new terminal, run the GUI:
   ```
   python gui_pyqt6.py
   ```

3. Use the GUI to enter your prompt and image URLs, then click "Submit" to send the request to the server.

## Project Structure

- `gui_pyqt6.py`: The main GUI application using PyQt6
- `server.py`: Flask server that interfaces with the Phi-3.5 Vision model
- `requirements.txt`: List of Python dependencies

## Configuration

By default, the GUI connects to a Flask server running on `http://localhost:5000`. If your server is running on a different address or port, modify the `submit_request` method in the `Phi35VisionGUI` class in `gui_pyqt6.py`.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses the Phi-3.5 Vision model developed by Microsoft.
- GUI built using PyQt6.

## Contact

If you have any questions or feedback, please open an issue in the GitHub repository.

## Install

Flash-attn Install

```bash
pip install flash-attn --no-build-isolation
```

Torch Install, use Cuda12.2

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For 8bit

```bash
pip install -U bitsandbytes
```
