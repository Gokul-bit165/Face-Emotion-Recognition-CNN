import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont

model = tf.keras.models.load_model("artifacts/face_emotion_model.h5")

class_labels = ["angry","disgust","fear","happy","neutral","sad","surprise"]

def predict_emotion(img):
    img_gray = img.convert("L")
    img_resized = img_gray.resize((48, 48))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    pred_class = class_labels[np.argmax(preds)]
    confidences = {class_labels[i]: float(preds[i]) for i in range(len(class_labels))}

    img_with_text = img.copy()
    draw = ImageDraw.Draw(img_with_text)

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    text = f"{pred_class} ({max(preds)*100:.1f}%)"
    draw.text((10, 10), text, fill=(255, 0, 0), font=font)

    return img_with_text, pred_class, confidences

title = "😃 Emotion Recognition App"
description = "Upload an image or use your webcam to detect emotions using a trained CNN model."
article = "<p style='text-align: center'>Built with ❤️ using TensorFlow + Gradio</p>"

demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil", label="Upload Image or Use Webcam", sources=["upload", "webcam"]),
    outputs=[
        gr.Image(type="pil", label="Image with Prediction"),
        gr.Label(label="Predicted Emotion"),
        gr.Label(label="Confidence Scores")
    ],
    title=title,
    description=description,
    article=article,
    live=False
)

if __name__ == "__main__":
    demo.launch()
