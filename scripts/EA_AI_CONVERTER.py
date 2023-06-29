

try:
    import traceback
    import os
    import utils

    import clear_memory
 
    clear_memory.clear()
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
    import pyautogui
except:

    error = traceback.format_exc()
    print(error)
    with open(utils.get_EA_dump_folder_file("error.txt"), "w") as f:
        f.write(error)
    import sys
    sys.exit()


class AiConverter:

    @utils.try_catch_error
    def __init__(self):
        pass

EXE_NAME = u"Ennead_QAQC_AI"


def is_another_app_running():

    # print [x.title for x in pyautogui.getAllWindows()]
    for window in pyautogui.getAllWindows():
        # print window.title
        if window.title == EXE_NAME:
            return True
    return False


@utils.try_catch_error
def main():
    if is_another_app_running():
        return
    app = App()
    app.run()

########################################
if __name__ == "__main__":
    main()
