import gradio as gr
from safetensors.torch import load_file
from Model.VisionEncoder import SiglipEecoder
import numpy as np
from Utils.FontData import draw_char

model = None
image_size = (64,64)

def load_model(path):
    global model
    weights = load_file(path)
    model = SiglipEecoder()
    model.load_state_dict(weights, strict=False)

def to_grey(img):
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    return gray

def compare(img1, img2):
    if img1.shape[-1] == 3:
        img1 = to_grey(img1)
    if img2.shape[-1] == 3:
        img2 = to_grey(img2)
    return model.get_distance(img1/255.0, img2/255.0).cpu().detach().numpy()

def compare_fonts(font1_path, font2_path, character):
    if model is None:
        return "err load model"
    distance = []
    for c in character:
        img1 = np.array(draw_char(font1_path, c, image_size))
        img2 = np.array(draw_char(font2_path, c, image_size))
        distance.append(compare(img1, img2))
    d = float(sum(distance) / len(distance))
    if d < 1.0:
        return f"差距: {d:.4f}, Font1 和 Font2 相同"
    else:
        return f"差距: {d:.4f}, Font1 和 Font2 不同"
    

def render(font1_path, font2_path, character):
    imgs1 = []
    imgs2 = []
    for c in character:
        img1 = np.array(draw_char(font1_path, c, image_size))
        img2 = np.array(draw_char(font2_path, c, image_size))
        imgs1.append(img1)
        imgs2.append(img2)
    img1 = np.concatenate(imgs1, axis=1)
    img2 = np.concatenate(imgs2, axis=1)
    return img1, img2

load_model("Model/model_data/model.safetensors")

with gr.Blocks() as demo:
    font1_path = gr.File(label="Font1", file_count="single", type="filepath")
    font2_path = gr.File(label="Font2", file_count="single", type="filepath")
    character = gr.Textbox(label="Character", value="在这里输入需要比较的字符")
    img1 = gr.Image(label="Font1 Image", type='numpy')
    img2 = gr.Image(label="Font2 Image", type='numpy')
    result = gr.Textbox(label="Result", type='text')
    rander_btn = gr.Button("Render")
    compare_btn = gr.Button("Compare")
    rander_btn.click(render, inputs=[font1_path, font2_path, character], outputs=[img1, img2])
    compare_btn.click(compare_fonts, inputs=[font1_path, font2_path, character], outputs=result)
demo.launch()