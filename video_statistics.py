'''
Video Statistics

1. FPS - this captures the frames per second per video to establish a distribution
and other descriptive statistics eg. mean, mode, median, std
'''
import cv2
import glob
import re
import pandas as pd
import numpy as np
import subprocess

video_names = []
frame_count = []
duration = []

for video in glob.glob('data/*'):
    vidcap = cv2.VideoCapture(video)
    video_names.append(video)
    frame_count.append(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))
    # result = subprocess.Popen(["ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1", video],
    #                           stdout=subprocess.PIPE,
    #                           stderr=subprocess.STDOUT)
    # import ipdb; ipdb.set_trace()
    # duration_str = [re.sub("b'  Duration: ", "", str(x).split(",")[0]) for x in result.stdout.readlines() if b"Duration" in x]
    # if "00:00" not in duration_str[0]:
    #     import ipdb; ipdb.set_trace()
    # duration.append(duration_str[0])

df = pd.DataFrame({"name": video_names, "frame_count": frame_count})
import ipdb; ipdb.set_trace()
# df["frames_per_second"] = df["frame_count"] / df["duration"]
