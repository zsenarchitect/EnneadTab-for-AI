import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PIL import Image
import numpy as np
import cv2
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler


class ImageProcessor:
    def __init__(self, script_dir):
        self.script_dir = script_dir

    def convert2canny(self, image_path):
        image = Image.open(image_path)
        image = np.array(image)

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        out_path = os.path.join(self.script_dir, 'imgs', 'OUT', 'canny.jpg')
        canny_image.save(out_path)

        print("Image saved")
        return canny_image

    def initiate_pipeline(self, canny_image):
        torch.cuda.manual_seed(12345)
        generator = torch.Generator()
        generator.manual_seed(12345)
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()

        return pipe, generator

    def text2image(self, pipe, generator, canny_image, text, negative_text):
        image = pipe(
            text,
            negative_prompt=negative_text,
            num_inference_steps=20,
            generator=generator,
            image=canny_image,
            controlnet_conditioning_scale=0.5
        ).images[0]

        out_path = os.path.join(self.script_dir, 'imgs', 'OUT', 'AI.jpg')
        image.save(out_path)

        print("AI out")


class GUI(QtWidgets.QMainWindow):
    def __init__(self, image_processor):
        super(GUI, self).__init__()

        self.image_processor = image_processor

        self.setWindowTitle('AI Image Processor')

        self.input_path_line = QtWidgets.QLineEdit(self)
        self.input_path_line.setPlaceholderText("Enter image path here...")
        self.input_path_line.setGeometry(50, 50, 300, 40)

        self.browse_button = QtWidgets.QPushButton("Browse", self)
        self.browse_button.setGeometry(360, 50, 100, 40)
        self.browse_button.clicked.connect(self.browse_image)

        self.text_line = QtWidgets.QLineEdit(self)
        self.text_line.setPlaceholderText("Enter text here...")
        self.text_line.setGeometry(50, 100, 300, 40)

        self.negative_text_line = QtWidgets.QLineEdit(self)
        self.negative_text_line.setPlaceholderText(
            "Enter negative prompt here...")
        self.negative_text_line.setGeometry(50, 150, 300, 40)

        self.submit_button = QtWidgets.QPushButton("Process", self)
        self.submit_button.setGeometry(50, 200, 100, 40)
        self.submit_button.clicked.connect(self.process_image)

    def browse_image(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                             "All Files (*);;JPEG (*.jpg);;PNG (*.png)", options=options)
        if file_name:
            self.input_path_line.setText(file_name)

    def process_image(self):
        img_path = self.input_path_line.text()
        text = self.text_line.text()
        negative_text = self.negative_text_line.text()

        canny_image = self.image_processor.convert2canny(img_path)
        pipe, generator = self.image_processor.initiate_pipeline(canny_image)
        self.image_processor.text2image(
            pipe, generator, canny_image, text, negative_text)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_processor = ImageProcessor(script_dir)
    gui = GUI(image_processor)
    gui.show()

    app.exec_()
