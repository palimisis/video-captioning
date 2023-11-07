import os
import pathlib as plb
import shutil
import subprocess
import warnings

import numpy as np
import pretrainedmodels
import torch
import torchvision.models as models
from pretrainedmodels import utils

from video_feature_extractor import VideoDataset, VideoFeatureExtractor

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_video_frames(video, tmp_path):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        os.makedirs(tmp_path)
        video_to_frames_command = [
            "ffmpeg",
            # (optional) overwrite output file if it exists
            "-y",
            "-i",
            video,  # input file
            "-vf",
            "scale=400:300",  # input file
            "-qscale:v",
            "2",  # quality for JPEG
            "{0}/%06d.jpg".format(tmp_path),
        ]  # %06d 6-digit
        subprocess.call(video_to_frames_command, stdout=ffmpeg_log, stderr=ffmpeg_log)


def extract_features(frames_path, features_path, interval, video_id):
    C, H, W = 3, 224, 224
    # Load the pre-trained 3D CNN model
    model = pretrainedmodels.resnet152(pretrained="imagenet")
    model.last_linear = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    load_image_fn = utils.LoadTransformImage(model)
    # load data
    img_list = sorted(frames_path.glob("*.jpg"))
    # get index
    samples_ix = np.arange(0, len(img_list), interval)
    img_list = [img_list[int(i)] for i in samples_ix]
    # build tensor
    imgs = torch.zeros([len(img_list), C, H, W])
    for i in range(len(img_list)):
        img = load_image_fn(img_list[i])
        imgs[i] = img
    imgs = imgs.to(device)
    with torch.no_grad():
        feats = model(imgs)
    feats = feats.cpu().numpy()
    # save
    np.save(os.path.join(features_path, video_id + ".npy"), feats)


# Specify the paths to your data and label files
data_dir = "./msvd/YouTubeClips"
label_file = "./msvd/AllVideoDescriptions.txt"

# Create an instance of the VideoDataset
dataset = VideoDataset(data_dir, label_file)

# Instantiate the VideoFeatureExtractor
feature_extractor = VideoFeatureExtractor(input_channels=3, output_size=128)

tmp_path = plb.Path(r"./_frames_out")

# Iterate through the dataset and extract features
for video in dataset.iter_without_loading_all():
    print(video.stem)
    extract_video_frames(str(video), tmp_path)
    extract_features(tmp_path, "./msvd/video_features", 10, video.stem)
    pass
    # video_features = feature_extractor(video_data)
    # Save the extracted features to a file or database
    # perform other necessary operations

# Further data processing or analysis can be performed here
