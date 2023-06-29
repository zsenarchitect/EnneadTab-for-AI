import torch
import traceback
def main():
    pass
    print('Pytorch CUDA Version is ', torch.version.cuda)

    print('Whether CUDA is supported by our system:', torch.cuda.is_available())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device: ' + device)

    cuda_id = torch.cuda.current_device()
    print('CUDA Device ID: ', torch.cuda.current_device())
    print('Name of the current CUDA Device: ', torch.cuda.get_device_name(cuda_id))

    import playsound
    playsound.playsound("C:\\Users\\szhang\\github\\EnneadTab-for-AI\\scripts\\audio\\AI_img_finish.wav", True)
if __name__ == '__main__':

    try:
        main()
    except Exception as e:
        error = traceback.format_exc()
        with open("{}\error.txt".format(r"C:\Users\szhang\github\EnneadTab-for-AI\output"), "w") as f:
            f.write(error)