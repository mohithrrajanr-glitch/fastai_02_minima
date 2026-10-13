from pathlib import Path
import gradio as gr
from fastai.vision.all import *

learn_inf = load_learner(
    Path(__file__).parent / "export.pkl",
    cpu=True
)

def classify(image):
    if image is None:
        return "Please upload a photo first."

    img = PILImage.create(image)
    pred, pred_idx, probs = learn_inf.predict(img)

    return (
        f"Prediction: {pred}; "
        f"Probability: {float(probs[pred_idx]):.04f}"
    )

gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a bear photo"),
    outputs=gr.Textbox(label="Result"),
    title="Bear Classifier",
    submit_btn="Classify"
).launch(share=True, debug=True, show_error=True)