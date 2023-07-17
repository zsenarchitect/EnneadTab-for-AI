import os
import traceback
import subprocess
import tkinter as tk
import time
import math
from tkinter import ttk


def get_EA_local_dump_folder():
    return "{}\Documents\EnneadTab Settings\Local Copy Dump".format(os.environ["USERPROFILE"])


def get_EA_dump_folder_file(file_name):
    """include extension"""
    return "{}\{}".format(get_EA_local_dump_folder(), file_name)


def toast(main_text="", sub_text=""):
    pop_message(main_text, sub_text)
    return
    """
    :param main_text:
    :param sub_text:
    :return:
    """

    icon = os.path.dirname(os.path.realpath(__file__)) + "\\imgs\\ai_brain.png"

    def get_toaster():
        """Return full file path of the toast binary utility."""
        return r"L:\4b_Applied Computing\03_Rhino\12_EnneadTab for Rhino\Source Codes\lib\EnneadTab\EXE\Ennead_Toaster.exe"

    app_name = "EnneadTab For AI"

    # build the toast
    toast_args = r'"{}"'.format(get_toaster())
    toast_args += r' --app-id "{}"'.format(app_name)
    toast_args += r' --title "{}"'.format(main_text)
    toast_args += r' --message "{}"'.format(sub_text)
    toast_args += r' --icon "{}"'.format(icon)
    toast_args += r' --audio "default"'

    # send the toast now
    subprocess.Popen(toast_args, shell=True)


def try_catch_error(func):

    def wrapper(*args, **kwargs):

        # print_note ("Wrapper func for EA Log -- Begin: {}".format(func.__name__))
        try:
            # print "main in wrapper"
            out = func(*args, **kwargs)
            # print_note ( "Wrapper func for EA Log -- Finish:")
            return out
        except Exception as e:
            print(str(e))
            print("Wrapper func for EA Log -- Error: " + str(e))
            error = traceback.format_exc()

            error += "\n\n######If you have EnneadTab UI window open, just close the window. Do no more action, otherwise the program might crash.##########\n#########Not sure what to do? Msg Sen Zhang, you have dicovered a important bug and we need to fix it ASAP!!!!!########"
            error_file = get_EA_dump_folder_file("error.txt")
            with open(error_file, "w") as f:
                f.write(error)
            os.startfile(error_file)

            import sys
            sys.exit()

    return wrapper


def random_joke():
    import random
    with open('L:\\4b_Applied Computing\\03_Rhino\\12_EnneadTab for Rhino\\Source Codes\\lib\\EnneadTab\\FUN\_loading_screen_message.txt', "r") as f:
        lines = f.readlines()
    random.shuffle(lines)
    return lines[0].replace("\n", "")


class MessageApp:
    @try_catch_error
    def __init__(self, main_text, sub_text,
                 animation_in_duration=0.5,
                 animation_stay_duration=5,
                 animation_fade_duration=2):

        self.window = tk.Tk()
        self.window.iconify()
        # self.window.title("EnneadTab Messager")

        # self.window.attributes("-topmost", True)
        self.window.deiconify()
        self.begining_time = time.time()

        self.window_width = 800
        self.window_height = 150
        self.x = self.get_screen_width() // 2 - self.window_width//2
        self.y_final = self.get_screen_height() - self.window_height
        self.y_initial = self.get_screen_height()

        # 100x100 size window, location 700, 500. No space between + and numbers
        self.window.geometry("{}x{}+{}+{}".format(self.window_width,
                                                  self.window_height,
                                                  self.x,
                                                  self.y_initial))

        self.style = ttk.Style()
        self.style.configure(
            "Rounded.TLabel",
            background="dark grey",
            borderwidth=6,
            # relief="solid",
            foreground="white",
            font=("Comic Sans MS", 20),
            outline="white",
            bordercolor="orange",
            padding=20,
            anchor="center",
        )

        self.talk_bubble = ttk.Label(
            self.window,
            text=main_text + "\n" + sub_text,
            style="Rounded.TLabel"
        )
        # pady ====> pad in Y direction
        self.talk_bubble.pack(pady=5)

        # set the window to have transparent background, only show the label
        self.window.config(background="green")
        self.window.wm_attributes('-transparentcolor', 'green')
        self.window.wm_attributes('-topmost', True)
        self.window.overrideredirect(True)

        self.animation_in_duration = animation_in_duration  # Animation duration in seconds
        # Time to stay visible in seconds
        self.animation_stay_duration = animation_stay_duration
        self.animation_fade_duration = animation_fade_duration  # Fade duration in seconds

        self.window.after(1, self.update)

    @try_catch_error
    def update(self):
        # kill the app if running for more than total s .
        time_passed = time.time() - self.begining_time
        # print(time_passed)
        if time_passed > (self.animation_in_duration + self.animation_stay_duration + self.animation_fade_duration + 2):
            self.window.destroy()
            return

        if time_passed < self.animation_in_duration:
            progress = time_passed / self.animation_in_duration
            eased_progress = 1 - math.pow(1 - progress, 4)  # Ease-out function
            y = int(self.y_initial - eased_progress *
                    (self.y_initial - self.y_final))

            self.window.geometry("{}x{}+{}+{}".format(self.window_width,
                                                      self.window_height,
                                                      self.x,
                                                      y))

        elif time_passed > self.animation_in_duration + self.animation_stay_duration:
            progress = (time_passed - self.animation_in_duration -
                        self.animation_stay_duration) / self.animation_fade_duration
            opacity = 1.0 - progress
            self.window.attributes("-alpha", opacity)
            # print(opacity)

        self.window.after(1, self.update)

    def run(self):
        self.window.mainloop()

    def get_screen_width(self):
        return self.window.winfo_screenwidth()

    def get_screen_height(self):
        return self.window.winfo_screenheight()


def pop_message(main_text, sub_text):
    MessageApp(main_text, sub_text).run()
###########################################


if __name__ == "__main__":
    toast(main_text="test", sub_text="test")
    pop_message(main_text="test", sub_text="test")
