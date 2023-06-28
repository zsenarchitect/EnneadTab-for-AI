import cv2
import numpy as np
from PIL import Image
import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
import time
import tracebac

# begin_time = time.time()
# import_time = time.time() - begin_time
# print("import_time: {}".format(import_time))
print("import module finished")


def convert2canny():

    # Get the absolute path to the active script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the image
    image_path = os.path.join(script_dir, 'imgs', 'IN', 'input.jpg')
    image = Image.open(image_path)
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    image_path = os.path.join(script_dir, 'imgs', 'OUT', 'canny.jpg')
    canny_image.save(image_path)

    print("Image saved")
    return canny_image


def initiate_pipeline(canny_image):

    # for deterministic generation
    # Set the seed for CUDA-enabled devices
    torch.cuda.manual_seed(12345)
    # Create a random generator
    generator = torch.Generator()

    # Set the generator seed
    generator.manual_seed(12345)
    # generator = torch.Generator(device='cuda').manual_seed(12345)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    # change the scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config)
    
    
    # check availibity
    print (torch.cuda.is_available() )
    
    # enable xformers (optional), requires xformers installation
    pipe.enable_xformers_memory_efficient_attention()

    # cpu offload for memory saving, requires accelerate>=0.17.0
    pipe.enable_model_cpu_offload()

    return pipe, generator


def text2image(pipe, generator):
    image = pipe(
        "wooden computer, high quality",
        negative_prompt="cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, blurry, bad anatomy, bad proportions",
        num_inference_steps=20,
        generator=generator,
        image=canny_image,
        controlnet_conditioning_scale=0.5
    ).images[0]

    # Get the absolute path to the active script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'imgs', 'OUT', 'AI.jpg')
    image.save(image_path)

    print("AI out")


if __name__ == '__main__':


    canny_image = convert2canny()

    pipe, generator = initiate_pipeline(canny_image)

    text2image(pipe, generator)