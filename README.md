# Kokoro-TTS

**Note:** This is not the official repository. This tutorial explains how to run [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on Windows and Google Colab. You may encounter some bugs while running the Gradio app. For a smoother experience, visit the [Hugging Face discussions](https://huggingface.co/hexgrad/Kokoro-82M/discussions) for an easy installation guide, including ONNX Runtime.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/Kokoro-82M-WebUI/blob/main/Kokoro_82M_Colab.ipynb) <br>
[![HuggingFace Space Demo](https://img.shields.io/badge/🤗-Space%20demo-yellow)](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)

---

### Installation Tutorial

My Python Version is 3.10.9.

#### 1. Clone the GitHub Repository:
```bash
git clone https://github.com/NeuralFalconYT/Kokoro-82M-WebUI.git
cd Kokoro-82M-WebUI
```

#### 2. Create a Python Virtual Environment:
```bash
python -m venv myenv
```
This command creates a new Python virtual environment named `myenv` for isolating dependencies.

#### 3. Activate the Virtual Environment:
- **For Windows:**
  ```bash
  myenv\Scripts\activate
  ```
- **For Linux:**
  ```bash
  source myenv/bin/activate
  ```
This activates the virtual environment, enabling you to install and run dependencies in an isolated environment.

#### 4. Check CUDA Version:
```bash
nvcc --version
```
This checks the installed version of CUDA to ensure compatibility with PyTorch.

#### 5. Install PyTorch:
Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and install the version compatible with your CUDA setup. For example:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 6. Install Required Dependencies:
```bash
pip install -r requirements.txt
```
This installs all the required Python libraries listed in the `requirements.txt` file.

#### 7. Download Model and Get Latest VoicePack:
```bash
python download_model.py
```

---

### Install eSpeak NG

#### **For Windows:**
1. Download the latest eSpeak NG release from the [eSpeak NG GitHub Releases](https://github.com/espeak-ng/espeak-ng/releases/tag/1.51).
2. Locate and download the file named **`espeak-ng-X64.msi`**.
3. Run the installer and follow the installation steps. Ensure that you install eSpeak NG in the default directory:
   ```
   C:\Program Files\eSpeak NG
   ```
   > **Note:** This default path is required for the application to locate eSpeak NG properly.

#### **For Linux:**
1. Open your terminal.
2. Install eSpeak NG using the following command:
   ```bash
   sudo apt-get -qq -y install espeak-ng > /dev/null 2>&1
   ```
   > **Note:** This command suppresses unnecessary output for a cleaner installation process.

---

### Run Gradio App

To run the Gradio app, follow these steps:

1. **Activate the Virtual Environment:**
   ```bash
   myenv\Scripts\activate
   ```

2. **Run the Application:**
   ```bash
   python app.py
   ```

   Alternatively,
   on Windows, double-click on `run_app.bat` to start the application.

---
![app](https://github.com/user-attachments/assets/4db74922-7047-40cf-add1-b48274d27732)


### Credits
[Kokoro HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)

