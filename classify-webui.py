import gradio as gr
from safetensors.torch import load_file
from Model.VisionEncoder import SiglipEecoder
import numpy as np
from Utils.FontData import draw_char

# 全局模型和图片尺寸
model = None
image_size = (64, 64)


# 加载模型
def load_model(path):
    global model
    try:
        weights = load_file(path)
        model = SiglipEecoder()
        model.load_state_dict(weights, strict=False)
        return "模型加载成功！"
    except Exception as e:
        return f"加载模型失败：{str(e)}"


# 转换为灰度图
def to_grey(img):
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    return gray


# 比较图片
def compare(img1, img2):
    if img1.shape[-1] == 3:
        img1 = to_grey(img1)
    if img2.shape[-1] == 3:
        img2 = to_grey(img2)
    return model.get_distance(img1 / 255.0, img2 / 255.0).cpu().detach().numpy()


# 比较字体
def compare_fonts(font1_path, font2_path, character1, character2):
    if model is None:
        return "错误：请先加载模型。"
    if not font1_path or not font2_path:
        return "错误：请上传两个字体文件。"
    if not character1 or not character2:
        return "错误：请输入需要比较的字符。"

    distance = []
    for i in range(min(len(character1), len(character2))):
        c1 = character1[i]
        c2 = character2[i]
        img1 = np.array(draw_char(font1_path, c1, image_size))
        img2 = np.array(draw_char(font2_path, c2, image_size))
        distance.append(compare(img1, img2))

    d = float(sum(distance) / len(distance))
    if d < 0.5:
        return f"差距：{d:.4f} - 字体1与字体2相似"
    else:
        return f"差距：{d:.4f} - 字体1与字体2不同"


# 渲染字体图片
def render(font1_path, font2_path, character1, character2):
    if not font1_path or not font2_path:
        return None, None, "错误：请上传两个字体文件。"
    if not character1 or not character2:
        return None, None, "错误：请输入需要比较的字符。"

    imgs1 = []
    imgs2 = []
    for i in range(min(len(character1), len(character2))):
        c1 = character1[i]
        c2 = character2[i]
        img1 = np.array(draw_char(font1_path, c1, image_size))
        img2 = np.array(draw_char(font2_path, c2, image_size))
        imgs1.append(img1)
        imgs2.append(img2)

    img1 = np.concatenate(imgs1, axis=1)
    img2 = np.concatenate(imgs2, axis=1)
    return img1, img2, "图片渲染成功！"


# 启动时加载模型
load_model("Model/model_data/model.safetensors")

# 自定义 CSS 样式
custom_css = """
body {
    background-color: #f0f4f8;
}
.gr-button {
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
}
.gr-button-primary {
    background: #4a90e2 !important;
    color: white !important;
}
.gr-button-primary:hover {
    background: #357abd !important;
}
.gr-textbox, .gr-file {
    border-radius: 6px !important;
    border: 1px solid #d1d5db !important;
}
.gr-image {
    border: 2px solid #e5e7eb !important;
    border-radius: 8px !important;
    background: white !important;
}
.container {
    max-width: 1200px !important;
    margin: auto !important;
}
.header {
    text-align: center;
    font-size: 2em;
    color: #1f2937;
    margin-bottom: 20px;
}
.subheader {
    text-align: center;
    font-size: 1.2em;
    color: #6b7280;
    margin-bottom: 30px;
}
"""

# Gradio 界面
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("<div class='header'>字体比较工具</div>")
    gr.Markdown("<div class='subheader'>上传两个字体文件，比较字符的视觉和计算差异</div>")

    with gr.Row():
        with gr.Column(scale=1):
            font1_path = gr.File(label="字体 1 (TTF/OTF)", file_count="single", type="filepath", elem_classes="gr-file")
            font2_path = gr.File(label="字体 2 (TTF/OTF)", file_count="single", type="filepath", elem_classes="gr-file")

        with gr.Column(scale=1):
            character1 = gr.Textbox(
                label="字体 1 的字符",
                placeholder="输入字符（如：汉字123）",
                value="在这里输入需要比较的字符",
                elem_classes="gr-textbox"
            )
            character2 = gr.Textbox(
                label="字体 2 的字符",
                placeholder="输入字符（如：汉字123）",
                value="在这里输入需要比较的字符",
                elem_classes="gr-textbox"
            )

    with gr.Row():
        render_btn = gr.Button("渲染图片", variant="secondary", elem_classes="gr-button")
        compare_btn = gr.Button("比较字体", variant="primary", elem_classes="gr-button")

    with gr.Row():
        with gr.Column(scale=1):
            img1 = gr.Image(label="字体 1 渲染结果", type="numpy", elem_classes="gr-image")
        with gr.Column(scale=1):
            img2 = gr.Image(label="字体 2 渲染结果", type="numpy", elem_classes="gr-image")

    result = gr.Textbox(label="比较结果", interactive=False, elem_classes="gr-textbox")
    status = gr.Textbox(label="状态", interactive=False, visible=False)

    # 事件处理
    render_btn.click(
        fn=render,
        inputs=[font1_path, font2_path, character1, character2],
        outputs=[img1, img2, status],
        show_progress=True
    )
    compare_btn.click(
        fn=compare_fonts,
        inputs=[font1_path, font2_path, character1, character2],
        outputs=[result],
        show_progress=True
    )

demo.launch()