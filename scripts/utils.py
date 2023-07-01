import os
import traceback
import subprocess

def get_EA_local_dump_folder():
    return "{}\Documents\EnneadTab Settings\Local Copy Dump".format(os.environ["USERPROFILE"])

def get_EA_dump_folder_file(file_name):
    """include extension"""
    return "{}\{}".format(get_EA_local_dump_folder(), file_name)

def toast(main_text = "", sub_text = ""):
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
    subprocess.Popen(toast_args, shell = True)

def try_catch_error(func):

    def wrapper(*args, **kwargs):

        #print_note ("Wrapper func for EA Log -- Begin: {}".format(func.__name__))
        try:
            # print "main in wrapper"
            out = func(*args, **kwargs)
            #print_note ( "Wrapper func for EA Log -- Finish:")
            return out
        except Exception as e:
            print ( str(e))
            print (  "Wrapper func for EA Log -- Error: " + str(e)  )
            error = traceback.format_exc()

            error += "\n\n######If you have EnneadTab UI window open, just close the window. Do no more action, otherwise the program might crash.##########\n#########Not sure what to do? Msg Sen Zhang, you have dicovered a important bug and we need to fix it ASAP!!!!!########"
            error_file = get_EA_dump_folder_file("error.txt")
            with open(error_file, "w") as f:
                f.write(error)
            os.startfile(error_file)

            import sys
            sys.exit()


    return wrapper

if __name__ == "__main__":
    toast(main_text="test", sub_text="test")