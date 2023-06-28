import os

from PIL import Image
import numpy as np
import cv2
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler
import streamlit as st

class ImageProcessor:
    def __init__(self, input_image):
        self.input_image = input_image

    @property
    def script_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    def convert2canny(self):
        image = Image.open(self.input_image)
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
        self.canny_image = canny_image

    def initiate_pipeline(self):
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

        self.pipe, self.generator = pipe, generator

    def text2image(self, text, negative_text):
        image = self.pipe(
            text,
            negative_prompt=negative_text,
            num_inference_steps=20,
            generator=self.generator,
            image=self.canny_image,
            controlnet_conditioning_scale=0.5
        ).images[0]

        out_path = os.path.join(self.script_dir, 'imgs', 'OUT', 'AI.jpg')
        image.save(out_path)

        print("AI out")
        return image


    def setup(self):
        self.convert2canny()
        self.initiate_pipeline()


def main():
    st.title('EnneadTab-for-Web')
    st.header("Text2Image 💬")
    st.markdown("""---""")

    input_image = st.file_uploader(
        "Upload your control image.", type='jpg, png')
    if input_image is None:
        return
    st.image(input_image, use_column_width=True,
             caption="Uploaded Image as refernce.")
    image_processor = ImageProcessor(input_image)
    image_processor.setup()

    
    st.markdown("""---""")
    positive_prompt = st.text_input(
        "Descript your desired images.")
    negative_prompt = st.text_input(
        "Descript your desired negative images.")
    
    out_image = image_processor.text2image(positive_prompt, negative_prompt)


   
    
    
    
    st.markdown("""---""")
    st.image(out_image, use_column_width=True,
             caption="This App is created by Sen Zhang")
 



if __name__ == '__main__':
    main()

