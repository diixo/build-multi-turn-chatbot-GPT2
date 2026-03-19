
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from utils import colorstr


# Speakers dialogues dataset loader
'''
all_turns_tokens = [CLS] + turn0 + [SEP] + turn1 + [SEP] + turn2 + [SEP] + ...
label_tokens = [PAD] + PAD(turn0) + turn1 + PAD(turn2) + turn3 + ...

CrossEntropyLoss(ignore_index=pad_token_id), instead of IGNORE_INDEX = -100:

* ignores CLS
* ignores user-turns
* ignores padding
* calculates loss only for assistant-turns
'''
class DialogLoader(Dataset):
    def __init__(self, data: List[Tuple[str, ...]], tokenizer, config):
        self.tokenizer = tokenizer
        self.max_len = config.max_len

        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id must not be None")
        if self.tokenizer.cls_token_id is None:
            raise ValueError("tokenizer.cls_token_id must not be None")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must not be None")

        self.turn_sep_id = tokenizer.eos_token_id
        self.ctx_token_id = tokenizer.cls_token_id
        self.pad_token_id = tokenizer.pad_token_id

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
        seed_sentence_ids = None
        seed_sentence_len = 0

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

            # save first context turn for inference/eval
            if seed_sentence_ids is None and i % 2 != predict_parity:
                seed_sentence_len = min(len(token_ids), self.max_len)
                seed_sentence_ids = token_ids[:seed_sentence_len].copy()

            remaining = self.max_len - len(input_ids)
            if remaining <= 0:
                colorstr("red", f"max_len remaining={remaining}: over_size(<0) id=[{i}]")
                break

            # truncate the current utterance to fit the remaining space
            was_truncated = len(token_ids) > remaining
            if was_truncated:
                token_ids = token_ids[:remaining]

            input_ids.extend(token_ids)

            if i % 2 == predict_parity:
                labels.extend(token_ids)
            else:
                labels.extend([self.pad_token_id] * len(token_ids))

            # break the loop if already truncated, cause there is no more space
            if was_truncated:
                colorstr("red", f"turn_truncated_to_remaining={remaining} id=[{i}]")
                break

            if len(input_ids) >= self.max_len:
                colorstr("red", f"max_len: over_size={len(input_ids)} id=[{i}]")
                break

        assert seed_sentence_ids is not None, colorstr("red", "seed_sentence_ids is NONE")

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        seed_sentence_ids = torch.tensor(seed_sentence_ids, dtype=torch.long)

        return input_ids, labels, seed_sentence_ids, seed_sentence_len


    def __getitem__(self, idx: int):

        multi_turn_sentences = self.data[idx]
        predict_parity = idx % 2

        input_ids, labels, seed_sentence_ids, seed_sentence_len = self.make_data(
            multi_turn_sentences=multi_turn_sentences,
            predict_parity=predict_parity
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "seed_sentence_ids": seed_sentence_ids,
            "seed_sentence_len": seed_sentence_len,
        }


    @staticmethod
    def collate_fn_batch(
        batch: List[Dict[str, torch.Tensor]],
        padding_id: int,
        label_padding_id: int
    ) -> Dict[str, torch.Tensor]:

        max_len = max(item["input_ids"].size(0) for item in batch)
        max_seed_len = max(item["seed_sentence_ids"].size(0) for item in batch)

        batch_input_ids = []
        batch_labels = []
        batch_seed_sentence_ids = []
        batch_seed_sentence_len = []

        for item in batch:
            input_ids = item["input_ids"]
            labels = item["labels"]
            seed_sentence_ids = item["seed_sentence_ids"]
            seed_sentence_len = item["seed_sentence_len"]

            pad_len = max_len - input_ids.size(0)
            seed_pad_len = max_seed_len - seed_sentence_ids.size(0)

            if pad_len > 0:
                input_ids = torch.cat(
                    [input_ids, torch.full((pad_len,), padding_id, dtype=torch.long)],
                    dim=0
                )
                labels = torch.cat(
                    [labels, torch.full((pad_len,), label_padding_id, dtype=torch.long)],
                    dim=0
                )

            if seed_pad_len > 0:
                seed_sentence_ids = torch.cat(
                    [seed_sentence_ids, torch.full((seed_pad_len,), padding_id, dtype=torch.long)],
                    dim=0
                )

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_seed_sentence_ids.append(seed_sentence_ids)
            batch_seed_sentence_len.append(seed_sentence_len)

        return {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "labels": torch.stack(batch_labels, dim=0),
            "seed_sentence_ids": torch.stack(batch_seed_sentence_ids, dim=0),
            "seed_sentence_len": torch.tensor(batch_seed_sentence_len, dtype=torch.long),
        }
