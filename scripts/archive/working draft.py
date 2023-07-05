import traceback

try:
    import clear_memory
    import traceback
    clear_memory.clear()
    import os
    # try:
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=64'
    # except:
    #     print (traceback.format_exc())

    import torch

    import cv2
    import numpy as np
    import PIL # this is to force include PIL in the pyinstaller process. The import itself does nothing.
    print (PIL.__version__)
    from PIL import Image

    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
    import time
    from playsound import playsound
except:

    error = traceback.format_exc()
    print(error)
    with open("{}\error.txt".format(r"C:\Users\szhang\github\EnneadTab-for-AI\output"), "w") as f:
        f.write(error)
    import sys
    sys.exit()


import subprocess
# play audio when script reach end


def play_audio(audio=None):
    if audio is None:
        audio_path = os.path.join(os.path.dirname(__file__), "audio", "default.wav")
    else:
        audio_path = os.path.join(os.path.dirname(__file__), "audio", audio)
    print (audio_path)
    if os.path.isfile(audio_path):
        print("Playing audio: " + audio_path)
    else:
        print("Audio file not found")
        return

    playsound(audio_path)
    return
    try:
        import clr

        clr.AddReference('System')
        from System.Media import SoundPlayer
        sp = SoundPlayer()
        sp.SoundLocation = audio_path
        sp.Play()
    except:
        pass


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
    play_audio()
    return canny_image


def initiate_pipeline(canny_image):

    # for deterministic generation
    # Set the seed for CUDA-enabled devices
    """
    torch.cuda.manual_seed(12345)
    # Create a random generator
    generator = torch.Generator()

    # Set the generator seed
    generator.manual_seed(12345)
    """

    # Making the code device-agnostic
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    generator = torch.Generator(device=device).manual_seed(12345)
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
    print(torch.cuda.is_available())

    # enable xformers (optional), requires xformers installation
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        print("xformers optimzation not available")

    # cpu offload for memory saving, requires accelerate>=0.17.0
    pipe.enable_model_cpu_offload()
    play_audio()

    print("pipeline and generator initiated")

    return pipe, generator


def text2image(canny_image, pipe, generator):

    images = pipe(
        "architecture exterior rendering, professional, archdaily.com, japanese architects, peaceful, few people, city center, after rain, dusk, vivid texture and reflection, high resolution, european modern architects, very detailed, natural lighting, award-winning, highest quality, sci-fi, natural material.",
        negative_prompt="cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, blurry, bad anatomy, bad proportions",
        num_inference_steps=20,
        generator=generator,
        image=canny_image,
        num_images_per_prompt=10,
        controlnet_conditioning_scale=0.5
    ).images

    # Get the absolute path to the active script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    session = time.strftime("%Y%m%d-%H%M%S")
    for i, image in enumerate(images):
        # make sure this folder exists:
        os.makedirs(os.path.join(script_dir, 'imgs', 'OUT', 'Session_{}'.format(session)),exist_ok=True)
        image_path = os.path.join(script_dir, 'imgs', 'OUT', 'Session_{}'.format(
            session), 'AI_{}.jpg'.format(i+1))
        image.save(image_path)

        clear_memory.clear()

    print("AI out")
    play_audio("AI_img_finish.wav")


def main():
    canny_image = convert2canny()

    pipe, generator = initiate_pipeline(canny_image)

    for i in range(1):
        text2image(canny_image, pipe, generator)


if __name__ == '__main__':

    try:
        play_audio()
        play_audio("import_finish.wav")
        main()
    except Exception as e:

        error = traceback.format_exc()

        print(error)
        with open("{}\error.txt".format(r"C:\Users\szhang\github\EnneadTab-for-AI\output"), "w") as f:
            f.write(error)
