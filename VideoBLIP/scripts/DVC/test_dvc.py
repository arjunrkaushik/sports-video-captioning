import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
import json
from functools import partial
from typing import Any
import torch
import transformers
from pytorchvideo.transforms import UniformTemporalSubsample
from torchvision.transforms import Compose
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers import Blip2Processor
from video_blip.data.myData import SoccerCaptioningEmbeddings
from video_blip.data.myUtils import (
    DataCollatorForVideoSeq2Seq,
    generate_input_ids_and_labels,
)
from video_blip.model import VideoBlipForConditionalGeneration


PROMPT = "Question: Generate soccer commentary Answer:"
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--test_data_dir", required=True)
parser.add_argument("--num_subsample_frames", type=int, required=True)
parser.add_argument("--num_workers", type=int, default=0)
args = parser.parse_args()

def test() -> None:

    processor = transformers.Blip2Processor.from_pretrained(
        args.model_dir
    )
    model = VideoBlipForConditionalGeneration.from_pretrained(
        args.model_dir,
        low_cpu_mem_usage=False if is_deepspeed_zero3_enabled() else True,
    )
    # freeze everything except for qformer
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False
    # we need to enable input require grads since the vision model (the first layer) is
    # frozen.
    model.enable_input_require_grads()
    print("Preparing test dataset")
    test_data = SoccerCaptioningEmbeddings(
        args.test_data_dir,
        train=False
    )

    training_args = transformers.TrainingArguments(
        output_dir="/home/csgrad/kaushik3/VideoBLIP/Data/Test/",
        do_predict = True
        # Add other necessary training arguments
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_data,
        data_collator=DataCollatorForVideoSeq2Seq(
            processor.tokenizer,
            pad_to_multiple_of = 100
        )
    )

    output = []

    for item in trainer.get_eval_dataloader():
        generated_ids = model.generate(
            **item,
            max_new_tokens=16,
            do_sample = True,
            num_beams = 5
        )
        generated = processor.batch_decode(generated_ids, skip_special_tokens=True)
        # print(generated)
        generated_text = generated
        # print(generated_text)
        output.extend(generated_text)
    # results = trainer.predict(test_data)
    json_path = "/home/csgrad/kaushik3/VideoBLIP/Data/out2.json"
    with open(json_path, 'w') as file:
        json.dump(output, file)
    # generated_text = processor.batch_decode(results.predictions, skip_special_tokens=True)[0].strip()
    # print(generated_text)
if __name__ == "__main__":
    test()
