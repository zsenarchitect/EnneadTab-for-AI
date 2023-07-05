
import traceback
def main():

    ace > 2



if __name__ == '__main__':

    try:
        main()
    except Exception as e:
        error = traceback.format_exc()
        with open("{}\error.txt".format(r"C:\Users\szhang\github\EnneadTab-for-AI\output"), "w") as f:
            f.write(error)