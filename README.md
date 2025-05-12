# 🛰️ EuroSAT Classifier + Grad-CAM

An interactive Gradio app that classifies satellite images using a fine-tuned ResNet50 model on the [EuroSAT dataset](https://github.com/phelber/eurosat). 
The app visualises model predictions with Grad-CAM and even explains them using a language model via OpenRouter.

---

## 🌍 Features

- 🔍 **Image Classification** for 10 EuroSAT land cover classes
- 🧠 **Grad-CAM Visualisation** to understand model focus
- 🤖 **LLM-powered Explanations** via OpenRouter's Mistral model
- 🎛️ Gradio UI with random category sampling and live feedback

---

## 🚀 Getting Started

### 🐳 Docker (Recommended)

Build and run with Docker:

```bash
docker build -t eurosat-app .
docker run -p 7860:7860 eurosat-app
```

### 🧑‍💻 Manual (Python)
Install dependencies:

```bash
pip install -r requirements.txt
python app.py
```
