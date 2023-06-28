import torch
print('Pytorch CUDA Version is ', torch.version.cuda)

print('Whether CUDA is supported by our system:', torch.cuda.is_available())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device: ' + device)

cuda_id = torch.cuda.current_device()
print('CUDA Device ID: ', torch.cuda.current_device())
print('Name of the current CUDA Device: ', torch.cuda.get_device_name(cuda_id))
