# Handwritten Digit Recognition Framework (CNN, BNN, LSVM) + ATM FSM

A small research/teaching repository that contains:
- A PyQt5 desktop app for real-time handwritten digit recognition from a webcam using three interchangeable models: CNN, BNN, and LSVM
- Training/testing scripts to reproduce the models on MNIST
- A separate C++ Finite State Machine (FSM) example for an ATM

## Repository Structure
- `Framework_source/APP_1/`: PyQt5 GUI application for live recognition
  - `main.py`: App entry point
  - `modelAPP.py`: Generated UI code
  - `CheckCamrera.py`: Camera enumeration via GStreamer
  - `CheckResource.py`: CPU/RAM monitor helpers
  - `model_CNN.py`, `model_BNN.py`, `model_LSVM.py`: Inference wrappers
  - `Model_CNN.h5`, `Model_BNN.npy`, `Model_LSVM.pkl`: Pretrained weights used by the app
- `Model_Source/`: Model training and testing scripts (MNIST)
  - `train_model_BNN.py`, `test_model_BNN.py`
  - `train_model_BNN_K-Means.py`, `test_model_BNN_K-Means.py`
  - `train_model_LSVM.py`, `test_model_LSVM.py`
- `Model/`: Example pretrained artifacts (for reference)
- `Framework_source/FSM_ATM/`: C++ ATM FSM example (separate from the GUI app)
- `Image/`: Figures and diagrams used in documentation/presentation

## Requirements
- Python 3.8–3.11 recommended
- OS notes
  - The GUI uses POSIX message queues (`sysv_ipc`) and GStreamer. Linux is recommended.
  - On Windows, `sysv_ipc` is not supported; consider WSL/Linux, or refactor the app to avoid message queues.
  - GStreamer is required for camera enumeration (PyGObject). OpenCV is used for capture.

### Python dependencies (minimum)
Install into a virtual environment.

```bash
pip install numpy opencv-python pillow psutil PyQt5 tensorflow dill bitarray scikit-learn pandas matplotlib scikit-image tqdm
```

- For Linux GUI features:
```bash
pip install sysv-ipc pygobject
```

- PyGObject and GStreamer also require system packages. For Ubuntu/Debian:
```bash
sudo apt update && sudo apt install -y \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  python3-gi python3-gi-cairo gir1.2-gst-plugins-base-1.0 gir1.2-gstreamer-1.0
```

## Quick Start (GUI App)
1. Create and activate a Python virtual environment
2. Install dependencies (see above)
3. Ensure GStreamer is installed and working (Linux recommended)
4. Run the app:

```bash
cd Framework_source/APP_1
python main.py
```

In the UI:
- Select a camera from the dropdown and click "Start"
- Choose a model (CNN/BNN/LSVM) from the dropdown
- Click "Start Predict" to see live predictions

The app expects model files in `Framework_source/APP_1`:
- `Model_CNN.h5` (Keras/TensorFlow)
- `Model_BNN.npy` (binary neural network weights)
- `Model_LSVM.pkl` (dill-serialized representative images and thresholds)

These files are already included. To use your own weights, replace them with the same filenames.

## Training and Testing (MNIST)
All scripts live under `Model_Source/`. They are research scripts and may need minor path edits before running.

- BNN (standard):
  - Train: `python Model_Source/train_model_BNN.py` → saves `BNN_relu.npy` and plots
  - Test: `python Model_Source/test_model_BNN.py`
  - Note: expects a helper module named `resources` (providing `FeedForward`/`AccTest`). If missing, you will need to implement or adapt it.

- BNN (with K-Means clustering):
  - Train: `python Model_Source/train_model_BNN_K-Means.py` → saves `BNN_model_cluster_29.npy`
  - Test: `python Model_Source/test_model_BNN_K-Means.py`

- LSVM-style classifier:
  - Train: `python Model_Source/train_model_LSVM.py` → saves a `best_model.pkl`
  - Test: `python Model_Source/test_model_LSVM.py`

Dataset paths:
- Several scripts use hard-coded MNIST image folder and CSV label paths, e.g.
  - `test_folder_path = 'C:/Users/Admin/Desktop/Fianl Project Document/MNIST/test'`
  - `test_csv_path = 'C:/Users/Admin/Desktop/Fianl Project Document/MNIST/test/_classes.csv'`
- Update these paths to your local dataset before running. The GUI app itself does not use these paths (it works from the camera).

Tip: You can copy artifacts from `Model/` to `Framework_source/APP_1/` and rename them to the expected filenames if you want to try different pre-trained weights.

## ATM FSM (C++ example)
The ATM FSM example is independent from the Python app.

- Location: `Framework_source/FSM_ATM/FSM_atm`
- To build on Linux:
```bash
cd Framework_source/FSM_ATM/FSM_atm
/usr/bin/g++ -std=c++17 ATM_FSM.cpp FSMHelper.cpp main.cpp mainhelper.cpp -o testmain
./testmain
```
Adjust the compiler path as needed.

## Troubleshooting
- No cameras appear in the dropdown:
  - Verify GStreamer is installed and discoverable
  - Try a different plugin pack (e.g., `gstreamer1.0-plugins-good`)
- OpenCV cannot open the camera:
  - Ensure the selected device index matches an actual OS camera
- `sysv_ipc` import error on Windows:
  - Run on Linux, or remove/refactor the message-queue code and pass the model selection directly within the Qt app
- TensorFlow errors:
  - Use a CPU-only TensorFlow build if you don’t have GPU/CUDA

## Notes
- This project is intended for educational purposes; performance and robustness are secondary
- The GUI and research scripts are loosely coupled; the GUI only needs the three model files
- See `Image/` for figures used in the accompanying report/presentation
