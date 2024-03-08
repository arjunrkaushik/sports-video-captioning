import json
import os
import random
import re
from collections.abc import Callable
from csv import DictReader
from fractions import Fraction
import json
import torch
from typing import Any
import pandas as pd
from pytorchvideo.data import ClipSampler, LabeledVideoDataset
from pytorchvideo.data.clip_sampling import ClipInfo
from pytorchvideo.data.video import VideoPathHandler
from torch.utils.data import Dataset
from transformers import Blip2Processor

PROMPT = "Question: Generate soccer commentary Answer:"

def extractTimeInfo(time_str):
    parts = time_str.split('-')
    minutes, seconds = map(int, parts[1].split(':'))
    total_seconds = minutes * 60 + seconds

    return int(parts[0]), int(total_seconds)

class SoccerCaptioningEmbeddings(Dataset):
    def __init__(
        self,
        data_dir: str,
        train: bool 
        # processor: Blip2Processor
    ) -> None:
        """
        :param json_path: path to json with extracted embeddings and captions
        """
        
        if train:
            csv_path = os.path.join(data_dir, "train.csv")
            json_path = os.path.join(data_dir, "train_new.json")
        else:
            csv_path = os.path.join(data_dir, "val.csv")
            json_path = os.path.join(data_dir, "val_new.json")
        
        with open(json_path, 'r') as json_file:
            self.data_dict = json.load(json_file)
        self.csv_data = pd.read_csv(csv_path)
        # self.processor = processor
        

    def __getitem__(self, index: int) -> dict[str, Any]:
        # datapoint = self.csv_data[index]
        # print(self.data_dict[self.csv_data.iloc[index]["gameTime"]][0])
        # print(torch.tensor(self.data_dict[self.csv_data.iloc[index]["gameTime"]][0]).shape)

        item = {}
        item["pixel_values"] = torch.tensor(self.data_dict[self.csv_data.iloc[index]["gameTime"]]["pixel_values"], dtype = torch.float32) 
        item["labels"] = self.data_dict[self.csv_data.iloc[index]["gameTime"]]["labels"]
        item["input_ids"] = self.data_dict[self.csv_data.iloc[index]["gameTime"]]["input_ids"]

        print("Labels = ", item["labels"])
        print("True sentence = ", self.csv_data.iloc[index]["anonymized"])
        # print("Embeddings = ", item["pixel_values"].shape)
        # print("Narration = ", item["narration_text"].shape)
        # print("Input ids = ", item["input_ids"].shape)
        # decoder_only_lm = True
        # preprocessed = generate_input_ids_and_labels(
        #     self.processor.tokenizer, PROMPT, self.csv_data.iloc[index]["anonymized"], decoder_only_lm
        # )
        # preprocessed["pixel_values"] = torch.tensor(self.data_dict[self.csv_data.iloc[index]["gameTime"]]["pixel_values"], dtype = torch.float32) 
        return item

    def __len__(self) -> int:
        return len(self.csv_data)

class SoccerCaptioningFrame(Dataset[dict[str, Any]]):
    def __init__(
        self,
        narrated_actions_dir: str,
        train: bool,
        video_dir: str,
        transform: Callable[[dict], Any] | None = None,
    ) -> None:
        """
        :param narrated_actions_dir: path to dir that contains narrated_actions.csv
            and extracted frames
        """
        self.narrated_actions_dir = narrated_actions_dir
        self.data: list[dict] = []
        if train:
            csv_path = os.path.join(self.narrated_actions_dir, "train.csv")
        else:
            csv_path = os.path.join(self.narrated_actions_dir, "val.csv")
        self.csv_data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self._video_path_handler = VideoPathHandler()
        self._transform = transform

    def __getitem__(self, index: int) -> dict[str, Any]:
        # datapoint = self.csv_data[index]
        clipTime = 15
        video1 = self._video_path_handler.video_from_path(
            os.path.join(self.video_dir, '1_224p.mkv')
        )
        video2 = self._video_path_handler.video_from_path(
            os.path.join(self.video_dir, '2_224p.mkv')
        )
        # just get the whole video since the clip is already extracted
        half, timeStamp = extractTimeInfo(self.csv_data.iloc[index]['gameTime'])
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

        item = {"video": clip["video"], "narration_text": self.csv_data.iloc[index]["anonymized"]}

        if self._transform is not None:
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        return len(self.csv_data)