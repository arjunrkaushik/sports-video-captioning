import argparse
import csv
import os
from collections.abc import Callable
import json
import numpy as np
import pandas as pd
import torch
from pytorchvideo.transforms import UniformTemporalSubsample
from pytorchvideo.data.clip_sampling import ClipInfo
from pytorchvideo.data.video import VideoPathHandler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
from transformers import Blip2Processor
from video_blip.data.myUtils import generate_input_ids_and_labels_as_list

# from video_blip.data.ego4d import Ego4dFHOMainDataset
# from video_blip.data.myData import SoccerCaptioning

# parser = argparse.ArgumentParser()
# parser.add_argument("--split_path", required=True)
# parser.add_argument("--video_dir", required=True)
# parser.add_argument("--frames_dir", required=True)
# parser.add_argument("--model_name_or_path", required=True)
# parser.add_argument("--num_subsample_frames", type=int, required=True)
# parser.add_argument("--num_workers", type=int, default=0)
# parser.add_argument("--max_num_narrated_actions", type=int, default=0)
# args = parser.parse_args()
PROMPT = "Question: Generate soccer commentary Answer:"

def extractTimeInfo(time_str):
    parts = time_str.split('-')
    minutes, seconds = map(int, parts[1].split(':'))
    total_seconds = minutes * 60 + seconds

    return int(parts[0]), int(total_seconds)

def extractFrames(csv_path, clipTime, video1, video2, video_transform, json_path):
    infoDict = {}
    csv = pd.read_csv(csv_path)
    
    for index, row in csv.iterrows():
        print(f"Extracting {row['gameTime']} ")
        decoder_only_lm = False
        cleaned_text = row['anonymized'].strip()
        preprocessed = generate_input_ids_and_labels_as_list(
            processor.tokenizer, PROMPT, cleaned_text, decoder_only_lm
        )
        half, timeStamp = extractTimeInfo(row['gameTime'])
        if half == 1:
            if timeStamp > video1.duration:
                clip = video1.get_clip(video1.duration - clipTime, video1.duration)
            elif timeStamp - clipTime < 0:
                clip = video1.get_clip(0, clipTime)
            else:
                clip = video1.get_clip(timeStamp - clipTime, timeStamp)
        elif half == 2:
            if timeStamp > video2.duration:
                clip = video2.get_clip(video2.duration - clipTime, video2.duration)
            elif timeStamp - clipTime < 0:
                clip = video2.get_clip(0, clipTime)
            else:
                clip = video2.get_clip(timeStamp - clipTime, timeStamp)
        else:
            print(f"Half = {half} unknown. Defaulting to last 30s of 2nd Half")
            clip = video2.get_clip(video2.duration - clipTime, video2.duration)

        pixel_values = clip["video"]
        if video_transform is not None:
            # preprocessed["pixel_values"] = video_transform(preprocessed["pixel_values"])
            pixel_values = video_transform(pixel_values)

            # run pixel_values through the image processor
            pixel_values = processor.image_processor(
                pixel_values.permute(1, 0, 2, 3), return_tensors="pt"
            )["pixel_values"].permute(1, 0, 2, 3)
        preprocessed["pixel_values"] = pixel_values.numpy().tolist()
        # print("input ids type = ", type(preprocessed["input_ids"]))
        # print("labels type = ", type(preprocessed["labels"]))
        # print("pixels type = ", type(preprocessed["pixel_values"]))
        infoDict[row['gameTime']] = preprocessed

    with open(json_path, 'w') as file:
        json.dump(infoDict, file)

if __name__ == '__main__':
    clipTime = 15
    num_subsample_frames = 15
    video_dir = '/home/csgrad/kaushik3/VideoBLIP/Data'
    video_handler = VideoPathHandler()
    video1 = video_handler.video_from_path(
        os.path.join(video_dir, '1_224p.mkv')
    )
    video2 = video_handler.video_from_path(
        os.path.join(video_dir, '2_224p.mkv')
    )
    processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-opt-2.7b"
    )
    video_transform = Compose([UniformTemporalSubsample(num_subsample_frames)])
    train_json_path = "/data/kaushik3/DVC_Data/train_new.json"
    test_json_path = "/data/kaushik3/DVC_Data/val_new.json"
    print("Working on Train")
    extractFrames(csv_path="/home/csgrad/kaushik3/VideoBLIP/Data/train.csv", clipTime=clipTime, video1=video1, video2=video2, video_transform=video_transform, json_path=train_json_path)
    print("Working on Test")
    extractFrames(csv_path="/home/csgrad/kaushik3/VideoBLIP/Data/val.csv", clipTime=clipTime, video1=video1, video2=video2, video_transform=video_transform, json_path=test_json_path)
   
