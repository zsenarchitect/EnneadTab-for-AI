import os
import traceback

def get_EA_local_dump_folder():
    return "{}\Documents\EnneadTab Settings\Local Copy Dump".format(os.environ["USERPROFILE"])

def get_EA_dump_folder_file(file_name):
    """include extension"""
    return "{}\{}".format(get_EA_local_dump_folder(), file_name)


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