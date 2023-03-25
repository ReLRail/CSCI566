import os
import datetime
import time
from pathlib import Path

output_dir = './data/v' # path to save
count = 4 # of video

directory = Path(output_dir).parent.__str__()
print(directory)
assert os.path.isdir(directory), f"no directory {directory}, please create manually"
assert not os.path.exists(output_dir), f"{output_dir} already exists"
os.mkdir(output_dir)

for _ in range(count):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    file_name = output_dir + now if output_dir.endswith('/') else output_dir + '/' + now
    time.sleep(3)
    os.system(f'cmd /c "ffmpeg -y -f dshow -video_size 1280x720 -framerate 30 -t 00:00:03 -i video="Integrated Webcam" "{file_name}.mp4""')
