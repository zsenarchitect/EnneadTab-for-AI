"""
for nexr version
see how to hook diffusion 2-1
 that has bigger resolution for base model
maybe can work with x4 upscaler better.
"""

import pyautogui

EXE_NAME = u"Ennead_IMAGE_AI_SCALER"


def is_another_app_running():

    # print [x.title for x in pyautogui.getAllWindows()]
    for window in pyautogui.getAllWindows():
        # print window.title
        if window.title == EXE_NAME:
            
            return True
        # if window.title == "EA_AI_CONVERTER":
        #     return True
    return False



if is_another_app_running():
    import sys
    sys.exit()

print ("EnneadTab UpScaler Ai is starting, please DO NOT CLOSE ME!!!!!!!!")
try:
    import traceback
    import os
    import utils
    import tkinter as tk
    import json
    import shutil
    import pprint
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w',
                        filename=utils.get_EA_dump_folder_file("{}_scaler_log.log".format(EXE_NAME)))

    print (utils.random_joke())
    import clear_memory

    clear_memory.clear()
    print (utils.random_joke())
    # try:
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=64'
    # except:
    #     print (traceback.format_exc())

    import torch
    print (utils.random_joke())

    import cv2
    import numpy as np
    # this is to force include PIL in the pyinstaller process. The import itself does nothing.
    import PIL
    print(PIL.__version__)
    from PIL import Image

    from diffusers import StableDiffusionUpscalePipeline
    print (utils.random_joke())
    import time
    from playsound import playsound
    import pyautogui
    print("Loading Finish.")
except:

    error = traceback.format_exc()
    print(error)
    with open(utils.get_EA_dump_folder_file("scaler_error.txt"), "w") as f:
        f.write(error)
    import sys
    sys.exit()


class AiScaler:
    @utils.try_catch_error
    def __init__(self):
        self.initiate_pipeline()

    def play_audio(self, file=None, force_play=False):
        # check if the username is "szhang"
        if os.getlogin() != "szhang":
            if not force_play:
                return

        if file is None:
            file = os.path.join(os.path.dirname(
                __file__), "audio", "default.wav")
        else:
            file = os.path.join(os.path.dirname(__file__), "audio", file)
        playsound(file)

    def has_new_job(self):
        # if self.is_thinking:
        #     return False

        # get all files in the folder that has "AI_RENDER_SCALER" in file name
        files = [f for f in os.listdir(
            utils.get_EA_local_dump_folder()) if "AI_RENDER_SCALER" in f]
        # if there is any file, return true
        if len(files) == 0:
            return False

        for file in files:
            copy_file = shutil.copyfile(
                utils.get_EA_dump_folder_file(file), utils.get_EA_dump_folder_file("temp_scaler_data_LISTENER.json"))
            with open(copy_file, 'r') as f:
                # get dictionary from json file
                data = json.load(f)

            if data["direction"] == "IN":
                self.data_file = utils.get_EA_dump_folder_file(file)
                return True
            else:
                os.remove(utils.get_EA_dump_folder_file(file))

        return False

    def initiate_pipeline(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # check availibity
        logging.info("Checking availibity: Device Name = {}".format(device))

        # Making the code device-agnostic
        generator = torch.Generator(device=device).manual_seed(22345)

        scaler_model_id = "stabilityai/stable-diffusion-x4-upscaler"
        # also can try "stabilityai/sd-x2-latent-upscaler"
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            scaler_model_id,  torch_dtype=torch.float16
        )
        pipeline = pipeline.to("cuda")

        # comment out below two line if want to do default:  load the LoRA weights from the Hub on top of the regular model weights, move the pipeline to the cuda device and run inference:
        # pipeline.unet.load_attn_procs(pipeline_model, local_files_only = True)
        """
        - A path to a *directory* containing model weights saved using [`~ModelMixin.save_config`], e.g.,
                      `./my_model_directory/`."""
        # pipeline.to("cuda")

        # enable xformers (optional), requires xformers installation
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except:
            logging.info("xformers optimzation not available")

        # cpu offload for memory saving, requires accelerate>=0.17.0
        pipeline.enable_model_cpu_offload()

        self.play_audio()

        logging.info("pipeline and generator initiated")

        self.scaler_pipeline = pipeline
        self.generator = generator

    def text2image(self, user_data):
        # print (self.canny_image)
        user_image = Image.open(user_data.get("input_image"))

        positive_prompt = user_data.get("positive_prompt")
        negative_prompt = user_data.get("negative_prompt")
        number_of_output = user_data.get("number_of_output")
        comment = ""
        print("Expecting {} upscale images".format(number_of_output))
        while True:
            try:
                upscale_images = self.scaler_pipeline(prompt=positive_prompt,
                                                      negative_prompt=negative_prompt,

                                                      image=user_image,
                                                      num_images_per_prompt=number_of_output
                                                      ).images
                print(
                    "Internal Resolution = {} -> {}".format(user_image.size, upscale_images[0].size))
                break
            except Exception as e:
                logging.info("Error in pipeline: {}".format(e))
                print(e)
                width, height = user_image.size

                # Calculate the new size
                old_size = (width, height)
                new_size = (int(width*0.75), int(height*0.75))
                comment = "\nThe image is too large, scaling it down to 75%. {} -> {}".format(
                    old_size, new_size)
                logging.info(comment)
                print(comment)
                comment += comment
                clear_memory.clear()
                # Resize the image
                user_image = user_image.resize(new_size)

        # Get the absolute path to the active script

        output_folder = os.path.join(utils.get_EA_local_dump_folder(
        ), 'EnneadTab_Ai_Rendering', 'Session_{}_Upscale'.format(self.session))
        os.makedirs(output_folder, exist_ok=True)
        print(output_folder)

        image_path = os.path.join(output_folder, 'Original.jpg')
        user_image.save(image_path)

        for i, raw_image in enumerate(upscale_images):

            # make sure this folder exists:
            image_path = os.path.join(
                output_folder, 'AI_Upscale_{}.jpg'.format(i+1))

            raw_image = raw_image.resize(user_data["desired_resolution"])
            raw_image.save(image_path)

        print("AI out! All images save in folder: {}".format(output_folder))
        self.play_audio("AI_img_finish.wav", force_play=True)

        # print ("TO DO: save the human readable meta data of input in this folder. Include P-promt, N prompt, style_tags, session time and name, number of output.")
        meta_data_json = {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "comments": comment,
            "desired_resolution": user_data.get("desired_resolution", [0, 0]),
            "session_time": self.session,
            "number_of_output": number_of_output}

        # save the meta data in the same folder as the images
        with open(os.path.join(output_folder, "EnneadTab AI Meta Data.json"), "w") as f:

            json.dump(meta_data_json, f, indent=4)

        # del self.scaler_pipeline
        # del self.generator
        del upscale_images
        del user_image
        clear_memory.clear()


        logging.info("meta_data = {}".format(pprint.pprint(meta_data_json)))
        print("AI out! All images save in folder: {}".format(output_folder))

        return meta_data_json

    @utils.try_catch_error
    def main(self):
        print("\n\n\nStarting a new job:")
        begin_time = time.time()
        with open(self.data_file, mode='r') as f:
            # get dictionary from json file
            data = json.load(f)

        self.session = time.strftime("%Y%m%d-%H%M%S")

        meta_data = self.text2image(data)

        data["meta_data"] = meta_data
        data["direction"] = "OUT"
        # data["compute_time"] = float(time.time() - begin_time)
        
        # logging.info("time = {}s".format(data['compute_time']))
        used_time_note = "{}s".format(time.time() - begin_time)
        print("Job finished! Time elapsed: {}".format(used_time_note))
        utils.toast(main_text="Rendering Upscale Job Done!",
                    sub_text="Job Time = {}".format(used_time_note))
        with open(self.data_file, mode='w') as f:
            # get dictionary from json file
            json.dump(data, f)


class App:
    def __init__(self):

        print("\n\nWelcome to EnneadTab AI Render Farm!!! This is a work-in-progress product.")
        print("Feedbacks are highly appreciated!")
        print("Please report any bugs or issues to: Sen Zhang.")

        self.AI = AiScaler()
        self.window = tk.Tk()
        self.window.iconify()
        self.window.title(EXE_NAME)
        self.is_thinking = False
        self.x = 900
        self.y = 300

        self.begining_time = time.time()

        self.window_width = 650
        self.window_height = 300
        # 100x100 size window, location 700, 500. No space between + and numbers
        self.window.geometry("{}x{}+{}+{}".format(self.window_width,
                                                  self.window_height,
                                                  self.x,
                                                  self.y))

        self.talk_bubble = tk.Label(self.window, text="EnneadTab Scaler AI is happy to help!", font=(
            "Comic Sans MS", 18), borderwidth=3, relief="solid")
        # pady ====> pad in Y direction
        self.talk_bubble.pack(pady=15)

        self.window.after(1, self.update)

    def update(self):
        # kill the app if running for more than 30 mins.
        if time.time() - self.begining_time > 60*30:
            self.window.destroy()
            return

        self.window.after(1000, self.check_job)

    def check_job(self):
        if not self.is_thinking and self.AI.has_new_job():
            # reset timer
            self.begining_time = time.time()

            self.is_thinking = True
            self.talk_bubble.configure(text="Porcessing...")
            self.AI.main()

            self.is_thinking = False
            self.talk_bubble.configure(text="Standby.")

            logging.info("DONE!")
        self.window.after(1, self.update)

    def run(self):
        print("\n\nThe app is up and running!")
        print("Enjoy!")
        self.window.mainloop()




@utils.try_catch_error
def main():
    if is_another_app_running():
        return

    app = App()
    app.run()


########################################
if __name__ == "__main__":
    main()
