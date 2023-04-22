# %%
from x3d_revised import x3d_with_regression
import torch

MODEL_PATH = '../model_pts/x3d_m_no_reg2.pt'
device = 'cuda'

model = x3d_with_regression(pretrained = True, hidden_dim= 9408, model_num_class = 5, head_activation = None, in_between= 256).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# %%
import torch
import numpy as np
import torchvision.transforms._functional_video as F

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)


model_name = "x3d_m"
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
model_transform_params  = {
    "x3d_xs": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 4,
        "sampling_rate": 12,
    },
    "x3d_s": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 13,
        "sampling_rate": 6,
    },
    "x3d_m": {
        "side_size": 224,
        "crop_size": 224,
        "num_frames": 16,
        "sampling_rate": 5,
    }
}
transform_params = model_transform_params[model_name]
sss = ShortSideScale(size=transform_params["side_size"])
crop_size = 224

# def process_img(img):
#     # tmp = F.center_crop(sss(F.normalize(torch.tensor(np.moveaxis(frame, -1, 0), device=device).unsqueeze(1) / 255, mean, std, inplace= True)), (crop_size, crop_size))
#     return sss(F.normalize(torch.tensor(np.moveaxis(img, -1, 0), device=device).unsqueeze(1) / 255, mean, std, inplace= True)).permute((1, 0, 2, 3))


# %%
import gc

num_frame = 16
vid_tensor = torch.zeros(size= (3, num_frame, crop_size, crop_size), device = device)
img_tensor = None

async def inference_once(frame):
    global vid_tensor
    global img_tensor

    tmp1 = torch.tensor(frame.transpose((2,0,1)), dtype=torch.float32).unsqueeze(1) / 255
    img_tensor = F.center_crop(sss(F.normalize(tmp1, mean, std, inplace= True)), (crop_size, crop_size)).to(device)
    new_tensor = torch.concat((vid_tensor[:,1:], img_tensor), dim=1)
    del img_tensor
    del vid_tensor
    torch.cuda.empty_cache()
    gc.collect()
    
    with torch.no_grad():
        outputs = model(new_tensor.unsqueeze(0))[0]
    vid_tensor = new_tensor
    print(torch.argmax(outputs, dim = -1))
    print(outputs)


# %%
import cv2
import asyncio

# define a video capture object
vid = cv2.VideoCapture(0)


vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
vid.set(cv2.CAP_PROP_FPS, 30)


INTERVAL = 5
cur_frame = 0

while(True):
	
	# Capture the video frame
	# by frame
	if cur_frame == 0:
		ret, frame = vid.read()
		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		asyncio.run(inference_once(frame))
	else:
		ret, _ = vid.read()
	
	# frame = putIterationsPerSec(frame, cps.countsPerSec())
	cv2.imshow('frame', frame)
	# cps.increment()

	cur_frame = (cur_frame + 1) % INTERVAL
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()



