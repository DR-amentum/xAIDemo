import os
import random
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms
import requests

# --- Configuration ---
OPENROUTER_API_KEY = "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"
MODEL_URL = "https://huggingface.co/cm93/resnet50-eurosat/resolve/main/pytorch_model.bin"
EUROSAT_PATH = "./eurosat"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model ---
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
state_dict = torch.hub.load_state_dict_from_url(MODEL_URL, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device).eval()

LAND_COVER_CLASSES = [
    "Forest",
    "River",
    "Highway",
    "AnnualCrop",
    "SeaLake",
    "HerbaceousVegetation",
    "Industrial",
    "Residential",
    "PermanentCrop",
    "Pasture"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Grad-CAM + Explanation ---
def get_target_layer(model):
    return model.layer4[-1]

def classify_and_cache(image):
    input_tensor = transform(image).unsqueeze(0).to(device)
    cache = {}
    target_layer = get_target_layer(model)

    def forward_hook(_, __, output): cache["activations"] = output
    def backward_hook(_, __, grad_output): cache["gradients"] = grad_output[0]

    fwd = target_layer.register_forward_hook(forward_hook)
    bwd = target_layer.register_full_backward_hook(backward_hook)

    logits = model(input_tensor)[0]
    probs = torch.softmax(logits, dim=0)
    pred_idx = int(probs.argmax().item())
    confidence = float(probs.max().item())

    model.zero_grad()
    logits[pred_idx].backward()

    fwd.remove()
    bwd.remove()

    return pred_idx, confidence, probs.detach().cpu().numpy(), {"pixel_values": input_tensor}, cache

def compute_gradcam(cache, input_tensor):
    activations = cache.get("activations")
    gradients = cache.get("gradients")
    if activations is None or gradients is None:
        return None
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=input_tensor["pixel_values"].shape[2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()  # <-- fixed here
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def explain_prediction(class_name, confidence, probs):
    summary = "\n".join(f"{LAND_COVER_CLASSES[i]}: {probs[i]*100:.2f}%" for i in range(len(probs)))
    prompt = (
        f"The model predicted: '{class_name}' with {confidence*100:.2f}% confidence.\n"
        f"Here are the class probabilities:\n{summary}\n"
        f"Why might the model have predicted '{class_name}'?"
    )
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "mistralai/mistral-7b-instruct",
                "messages": [{"role": "system", "content": "Explain remote sensing results simply."},
                             {"role": "user", "content": prompt}],
                "max_tokens": 400
            },
            timeout=30
        )
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM error] {e}"

# --- Dataset Loading ---
def get_gallery_images_by_category(root=EUROSAT_PATH):
    result = {}
    if os.path.exists(root):
        for cat in os.listdir(root):
            full_path = os.path.join(root, cat)
            if os.path.isdir(full_path):
                images = [os.path.join(full_path, f) for f in os.listdir(full_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                if images:
                    result[cat] = images
    return result

gallery_dict = get_gallery_images_by_category()
categories = list(gallery_dict)

# --- Gradio Logic ---
def process_image(image):
    pred_idx, confidence, probs, inputs, cache = classify_and_cache(image)
    class_name = LAND_COVER_CLASSES[pred_idx]
    explanation = explain_prediction(class_name, confidence, probs)
    cam = compute_gradcam(cache, inputs)
    image_resized = image.resize((224, 224))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Original image
    axs[0].imshow(image_resized)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    # Improved horizontal bar chart
    sorted_idx = np.argsort(probs)[::-1]
    axs[1].barh(range(len(probs)), probs[sorted_idx], color='skyblue')
    axs[1].set_yticks(range(len(probs)))
    axs[1].set_yticklabels([LAND_COVER_CLASSES[i] for i in sorted_idx])
    axs[1].invert_yaxis()  # Highest probability at top
    axs[1].set_xlabel("Probability")
    axs[1].set_title("Class Probabilities")

    # Grad-CAM overlay
    if cam is not None:
        axs[2].imshow(image_resized)
        axs[2].imshow(cam, cmap="jet", alpha=0.5)
        axs[2].set_title("Grad-CAM")
    else:
        axs[2].text(0.5, 0.5, "No Grad-CAM", ha='center', va='center')
    axs[2].axis("off")

    plt.tight_layout()
    return fig, explanation

def classify_random_image(category):
    if category not in gallery_dict:
        return plt.Figure(), "Invalid category"
    path = random.choice(gallery_dict[category])
    img = Image.open(path).convert("RGB")
    return process_image(img)

def handle_chat(user_msg, chat_history):
    if not user_msg:
        return chat_history, ""
    reply = f"Echo: {user_msg}"  # Replace with actual LLM reply if needed
    updated = chat_history + f"\nUser: {user_msg}\nAssistant: {reply}"
    return updated, ""

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🛰️ EuroSAT Classifier + Grad-CAM")

    with gr.Row():
        category = gr.Dropdown(label="Category", choices=categories, value=categories[0])
        classify_btn = gr.Button("Classify Random Image")

    plot = gr.Plot(label="Visualisation")
    explanation_box = gr.Textbox(label="Model Explanation", lines=5)

    chat_history = gr.Textbox(label="Chat History", lines=10, interactive=False)
    user_input = gr.Textbox(label="Ask a question...", lines=1)

    classify_btn.click(fn=classify_random_image, inputs=category, outputs=[plot, explanation_box])
    user_input.submit(fn=handle_chat, inputs=[user_input, chat_history], outputs=[chat_history, user_input])

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
