import json
import os
import random
import re
from collections.abc import Callable
from csv import DictReader
from fractions import Fraction
from typing import Any
import pandas as pd
from pytorchvideo.data import ClipSampler, LabeledVideoDataset
from pytorchvideo.data.clip_sampling import ClipInfo
from pytorchvideo.data.video import VideoPathHandler
from torch.utils.data import Dataset


def extractTimeInfo(time_str):
    parts = time_str.split('-')
    minutes, seconds = map(int, parts[1].split(':'))
    total_seconds = minutes * 60 + seconds

    return int(parts[0]), int(total_seconds)

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
                clip = video1.get_clip(video1.duration - 30, video1.duration)
            elif timeStamp - 30 < 0:
                clip = video1.get_clip(0, 30)
            else:
                clip = video1.get_clip(timeStamp - 30, timeStamp)
        elif half == 2:
            if timeStamp > video2.duration:
                clip = video2.get_clip(video2.duration - 30, video2.duration)
            elif timeStamp - 30 < 0:
                clip = video2.get_clip(0, 30)
            else:
                clip = video2.get_clip(timeStamp - 30, timeStamp)
        else:
            print(f"ERROR!!! Half = {half} unknown.")
            return {}

        item = {"video": clip["video"], "narration_text": self.csv_data.iloc[index]["anonymized"]}

        if self._transform is not None:
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        return len(self.csv_data)