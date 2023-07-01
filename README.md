# EnneadTab-for-AI

This is a working-in-progress project that will evetually intergrated to the EnneadTab for Rhino and Revit.



- How to activate:
when inside the terminal, type <.venv/Scripts/activate> and enter(no <>)
it will show as below:
PS C:\Users\szhang\github\EnneadTab-for-AI> .venv/Scripts/activate
(.venv) PS C:\Users\szhang\github\EnneadTab-for-AI> 



- A good reference to the workflow, include what to pip install frist:
https://ngwaifoong92.medium.com/introduction-to-controlnet-for-stable-diffusion-ea83e77f086e


- special note for pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



- Documentation on stable diffusion.
https://huggingface.co/docs/diffusers/v0.16.0/en/api/pipelines/stable_diffusion/img2img


- Guide to train my own SD model
https://betterprogramming.pub/train-your-own-stable-diffusion-model-locally-no-code-needed-36f943825d23


- python will work fine, but during py2exe some meta data will be not curried over, making exe fail. See this help to include additional meta data copy.
https://stackoverflow.com/questions/75393856/tqdm-4-27-distribution-was-not-found-error-while-executing-a-exe-file-create
