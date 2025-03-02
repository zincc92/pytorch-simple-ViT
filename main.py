import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import requests
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from io import BytesIO

# charging the huggingface model
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# quick print of all the class labels (for reference)
labels = model.config.id2label
for idx, label in labels.items():
    print(f"{idx}: {label}")

def predict_image():
    url = url_entry.get()
    # check if the url is valid
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, stream=True, headers=headers)
        if response.status_code != 200:
            raise ValueError("Invalid URL !")
        
        # if yes, download the image and display it
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))  # Redimensionner pour l'affichage
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        
        # predict the image 
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
        
        label_text.set(f"Predicted class: {model.config.id2label[predicted_class]}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# tkinter interface
root = tk.Tk()
root.title("ViT Image Classifier")
root.geometry("500x400")

tk.Label(root, text="Enter URL of the image :").pack(pady=5)
url_entry = tk.Entry(root, width=50)
url_entry.pack(pady=5)

predict_button = tk.Button(root, text="Predict", command=predict_image)
predict_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

label_text = tk.StringVar()
prediction_label = tk.Label(root, textvariable=label_text, font=("Arial", 14))
prediction_label.pack(pady=10)

root.mainloop()