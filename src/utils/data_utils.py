
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset


# Speakers dialogues dataset loader
class DialogLoader(Dataset):

    IGNORE_INDEX = -100

    def __init__(self, data: List[Tuple[str, ...]], tokenizer, config):
        self.tokenizer = tokenizer
        self.max_len = config.max_len

        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id must not be None")

        self.turn_sep_id = tokenizer.eos_token_id
        self.ctx_token_id = tokenizer.cls_token_id

        self.data = []
        for dialog in data:
            self.data.extend([dialog, dialog])

        self.length = len(self.data)


    def __len__(self):
        return self.length


    def make_data(self, multi_turn_sentences: Tuple[str, ...], predict_parity: int):
        #for i, sent in enumerate(multi_turn_sentences):
        #   print(f"Tokens[{i}]: {self.tokenizer.tokenize(sent)}")

        input_ids = []
        labels = []

        for i, sentence in enumerate(multi_turn_sentences):
            sentence_ids = self.tokenizer(
                sentence,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"].squeeze(0)

            token_ids = sentence_ids.tolist()

            # Attach from beginning the role-marker only for context utterance
            if i % 2 != predict_parity:
                token_ids = [self.ctx_token_id] + token_ids

            # Append turn separator for both roles
            token_ids = token_ids + [self.turn_sep_id]

            remaining = self.max_len - len(input_ids)
            if remaining <= 0:
                break

            if len(token_ids) > remaining:
                token_ids = token_ids[:remaining]

            input_ids.extend(token_ids)

            if i % 2 == predict_parity:
                labels.extend(token_ids)
            else:
                labels.extend([self.IGNORE_INDEX] * len(token_ids))

            if len(input_ids) >= self.max_len:
                break

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        return input_ids, labels, attention_mask


    def __getitem__(self, idx: int):
        multi_turn_sentences = self.data[idx]
        predict_parity = idx % 2

        input_ids, labels, attention_mask = self.make_data(
            multi_turn_sentences=multi_turn_sentences,
            predict_parity=predict_parity
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


    @staticmethod
    def collate_fn_batch(
        batch: List[Dict[str, torch.Tensor]],
        padding_id: int,
        label_padding_id: int = -100
    ) -> Dict[str, torch.Tensor]:

        max_len = max(item["input_ids"].size(0) for item in batch)

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for item in batch:
            input_ids = item["input_ids"]
            labels = item["labels"]
            attention_mask = item["attention_mask"]

            pad_len = max_len - input_ids.size(0)

            if pad_len > 0:
                input_ids = torch.cat(
                    [input_ids, torch.full((pad_len,), padding_id, dtype=torch.long)],
                    dim=0
                )
                labels = torch.cat(
                    [labels, torch.full((pad_len,), label_padding_id, dtype=torch.long)],
                    dim=0
                )
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros(pad_len, dtype=torch.long)],
                    dim=0
                )

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)

        return {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "labels": torch.stack(batch_labels, dim=0),
            "attention_mask": torch.stack(batch_attention_mask, dim=0),
        }
