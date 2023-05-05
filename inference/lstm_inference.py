# %%
import torch
import torch
from torch import nn
import torchvision

embed_size = 96



class lstm(nn.Module):
    def __init__(self, backbone,embed_size = embed_size, in_between = 64, out_features = 4, num_layers = 2, bidirectional = False):
        super(lstm, self).__init__()
        self.embed_size = embed_size
        self.backbone = backbone

        self.rnn = nn.LSTM(input_size = embed_size, hidden_size = 300, batch_first = True, bidirectional = bidirectional, dropout = 0.1, num_layers = num_layers)
        self.dropout = nn.Dropout(p = 0.1)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(300, 200)
        self.dropout2 = nn.Dropout(p = 0.1)
        self.fc2 = nn.Linear(200, 5)

    def forward(self, x):
        a, b, c, d, e = x.shape
        features = self.backbone(x.view((a * b, c, d, e))).view((a, b, self.embed_size))
        return self.fc2(self.dropout2(self.relu( \
            self.fc1(self.dropout(self.rnn(features)[0][:, -1, :])))))


# %%
device = 'cuda'

# # %%
MODEL_PATH = '../model_pts/resnet_with_regression_and_cls_4_21_no_crop.pt'

# %%

class fast_inference(nn.Module):
    def __init__(self, backbone,embed_size = embed_size, in_between = 64, out_features = 4, num_layers = 2, num_frame = 16, bidirectional = False):
        super(fast_inference, self).__init__()
        self.embed_size = embed_size
        self.backbone = backbone
        # self.embedding = nn.Embedding(x_grid_size * y_grid_size + 1, embed_size)
        self.rnn = nn.LSTM(input_size = embed_size, hidden_size = 300, batch_first = True, bidirectional = bidirectional, dropout = 0.1, num_layers = num_layers)
        # self.rnn2 = nn.LSTM(input_size = embed_size, hidden_size = 100, batch_first = True, bidirectional = True, dropout = 0.1, num_layers = num_layers)
        self.dropout = nn.Dropout(p = 0.1)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(300, 200)
        self.dropout2 = nn.Dropout(p = 0.1)
        self.fc2 = nn.Linear(200, 5)

        self.features = torch.zeros(size=(num_frame, embed_size), device = device)
        self.num_frame = num_frame



    def forward(self, x):
        o = self.backbone(x)
        
        self.features = torch.concat((self.features[1:,:], o))
        
        # print(self.features.shape)
        
        return self.fc2(self.dropout2(self.relu( \
            self.fc1(self.dropout(self.rnn(self.features)[0][-1, :])))))

# %%
fast_model = fast_inference(backbone=torchvision.models.resnet18(num_classes = embed_size)).to(device)
fast_model.load_state_dict(torch.load(MODEL_PATH), strict= False)
fast_model.eval()

# %%
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
frames_per_second = 30
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

# Get transform parameters based on model
transform_params = model_transform_params[model_name]


# %%
import torch
import numpy as np
import torchvision.transforms._functional_video as F

sss = ShortSideScale(size=transform_params["side_size"])
crop_size = 224


# %%
import cv2
def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

# %%
from datetime import datetime

class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time if elapsed_time > 0 else 0

# %%
import gc
async def inference_once(frame):
    tmp1 = torch.tensor(np.moveaxis(frame, -1, 0)).unsqueeze(1) / 255
    img_tensor = sss(F.normalize(tmp1, mean, std, inplace= True)).permute((1, 0, 2, 3)).to(device)

    # img_tensor = process_img(frame)
    with torch.no_grad():
        outputs = fast_model(img_tensor)
    print(torch.argmax(outputs, dim = -1))
    print(outputs)

    del outputs
    del img_tensor
    torch.cuda.empty_cache()
    gc.collect()



# %%
# import the opencv library
import cv2
import asyncio

# define a video capture object
vid = cv2.VideoCapture(0)


vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
vid.set(cv2.CAP_PROP_FPS, 30)

cps = CountsPerSec().start()

INTERVAL = 6
cur_frame = 0

while(True):
	
	# Capture the video frame
	# by frame
	if cur_frame == 0:
		ret, frame = vid.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
