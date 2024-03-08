import torch
from transformers import BatchEncoding, DataCollatorForSeq2Seq, PreTrainedTokenizer

class DataCollatorForVideoSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        # print(type(features))
        # print(features[0]["pixel_values"])
        # print(features[0]["labels"].shape)
        # print(features[0]["input_ids"].shape)

        pixel_values = torch.stack(
            [feature.pop("pixel_values") for feature in features]
        )
        collated = super().__call__(features, return_tensors=return_tensors)
        
        # labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # # same length to return tensors.
        # if labels is not None:
        #     # max_label_length = max(len(l) for l in labels)
        #     # max_label_length = 100
        #     padding_side = self.tokenizer.padding_side
        #     for feature in features:
        #         remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
        #         feature["labels"] = (
        #             feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
        #         )

        # features = self.tokenizer.pad(
        #     features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors="pt",
        # )

        # # prepare decoder_input_ids
        # if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
        #     decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
        #     features["decoder_input_ids"] = decoder_input_ids

        collated["pixel_values"] = pixel_values
        
        # print("After padding")
        # print(features[0]["pixel_values"].shape)
        # print(features[0]["labels"].shape)
        # print(features[0]["input_ids"].shape)
        return collated

# def clean_narration_text(narration_text: str) -> str:
#     # strip it first
#     cleaned = narration_text.strip()
#
#     # replace "#C C" with "The camera wearer"
#     return re.sub(Ego4dFHOMainDataset.C_REGEX, "The camera wearer", cleaned)


def generate_input_ids_and_labels(
    tokenizer: PreTrainedTokenizer, prompt: str, text: str, decoder_only_lm: bool
) -> BatchEncoding:
    """Generate input ids and labels from the given prompt and text. If
    decoder_only_lm is True, the input and label texts are the same, but label
    tokens that correspond to the prompt are masked with -100. If
    decoder_only_lm is False, the input corresponds to the prompt and the label
    to the text.

    :param tokenizer: tokenizer for tokenizing inputs and label
    :param prompt: prompt for the LLM
    :param text: text for the LLM to generate based on the prompt
    :param decoder_only_lm: whether the LLM is decoder only or not
    :returns: preprocessed results
    """
    if decoder_only_lm:
        # tokenize prompt first
        prompt_tokens = tokenizer(prompt, return_attention_mask=False).input_ids

        # tokenize the narration and append eos
        preprocessed = tokenizer(
            " " + text,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        preprocessed["input_ids"].append(tokenizer.eos_token_id)

        # join tokenized prompt and narration text
        preprocessed["input_ids"] = prompt_tokens + preprocessed["input_ids"]
        preprocessed["input_ids"] = torch.tensor(preprocessed.input_ids)

        # for decoder only LMs, labels are same as input_ids, but we mask
        # tokens for the prompt
        preprocessed["labels"] = preprocessed["input_ids"].clone()
        preprocessed["labels"][: len(prompt_tokens)] = -100
    else:
        # eos is automatically appended by the tokenizer
        # we don't use return_tensors='pt' here b/c it automatically batchifies things
        # which we don't want
        preprocessed = tokenizer(prompt, return_attention_mask=False)
        preprocessed["input_ids"] = torch.tensor(preprocessed["input_ids"])
        preprocessed["labels"] = torch.tensor(
            tokenizer(text, return_attention_mask=False).input_ids
        )

    return preprocessed

def generate_input_ids_and_labels_as_list(
    tokenizer: PreTrainedTokenizer, prompt: str, text: str, decoder_only_lm: bool
) -> dict:
    """Generate input ids and labels from the given prompt and text. If
    decoder_only_lm is True, the input and label texts are the same, but label
    tokens that correspond to the prompt are masked with -100. If
    decoder_only_lm is False, the input corresponds to the prompt and the label
    to the text.

    :param tokenizer: tokenizer for tokenizing inputs and label
    :param prompt: prompt for the LLM
    :param text: text for the LLM to generate based on the prompt
    :param decoder_only_lm: whether the LLM is decoder only or not
    :returns: preprocessed results
    """
    print(decoder_only_lm)
    if decoder_only_lm:
        # tokenize prompt first
        prompt_tokens = tokenizer(prompt, return_attention_mask=False).input_ids

        # tokenize the narration and append eos
        preprocessed = tokenizer(
            " " + text,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        preprocessed["input_ids"].append(tokenizer.eos_token_id)

        # join tokenized prompt and narration text
        preprocessed["input_ids"] = prompt_tokens + preprocessed["input_ids"]
        preprocessed["input_ids"] = torch.tensor(preprocessed.input_ids)

        # for decoder only LMs, labels are same as input_ids, but we mask
        # tokens for the prompt
        preprocessed["labels"] = preprocessed["input_ids"].clone()
        preprocessed["labels"][: len(prompt_tokens)] = -100
    else:
        # eos is automatically appended by the tokenizer
        # we don't use return_tensors='pt' here b/c it automatically batchifies things
        # which we don't want
        preprocessed = tokenizer(prompt, return_attention_mask=False)
        preprocessed["input_ids"] = torch.tensor(preprocessed["input_ids"])
        preprocessed["labels"] = torch.tensor(
            tokenizer(text, return_attention_mask=False).input_ids
        )

    item = {}
    input_text = preprocessed.pop("input_ids")
    item["input_ids"] = input_text.numpy().tolist()
    labels = preprocessed.pop("labels")
    item["labels"] = labels.numpy().tolist()
    return item