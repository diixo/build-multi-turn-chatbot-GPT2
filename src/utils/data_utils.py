
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset



class DialogLoader(Dataset):

    IGNORE_INDEX = -100

    def __init__(self, data: List[Tuple[str, ...]], tokenizer, config):
        self.tokenizer = tokenizer
        self.max_len = config.max_len

        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id must not be None")

        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "tokenizer.pad_token_id must not be None. "
                "For GPT-2 usually set tokenizer.pad_token = tokenizer.eos_token"
            )

        self.turn_sep_id = self.tokenizer.eos_token_id

        self.data = []
        for dialog in data:
            self.data.extend([dialog, dialog])

        self.length = len(self.data)


    @staticmethod
    def _pad(x: list, length: int, pad_value: int) -> list:
        if len(x) >= length:
            return x[:length]
        return x + [pad_value] * (length - len(x))


    def make_data(self, multi_turn_sentences: Tuple[str, ...], predict_parity: int):
        input_ids = []
        labels = []

        for i, sentence in enumerate(multi_turn_sentences):
            sentence_tokens = self.tokenizer.encode(sentence)
            sentence_tokens.append(self.turn_sep_id)

            remaining = self.max_len - len(input_ids)
            if remaining <= 0:
                break

            if len(sentence_tokens) > remaining:
                sentence_tokens = sentence_tokens[:remaining]

            input_ids.extend(sentence_tokens)

            if i % 2 == predict_parity:
                labels.extend(sentence_tokens)
            else:
                labels.extend([self.IGNORE_INDEX] * len(sentence_tokens))

            if len(input_ids) >= self.max_len:
                break

        real_len = len(input_ids)

        input_ids = self._pad(input_ids, self.max_len, self.tokenizer.pad_token_id)
        labels = self._pad(labels, self.max_len, self.IGNORE_INDEX)
        attention_mask = [1] * real_len + [0] * (self.max_len - real_len)

        assert len(input_ids) == self.max_len
        assert len(labels) == self.max_len
        assert len(attention_mask) == self.max_len

        return input_ids, labels, attention_mask


    def __getitem__(self, idx: int):
        multi_turn_sentences = self.data[idx]
        predict_parity = idx % 2

        for i, sent in enumerate(multi_turn_sentences):
            print(f"Tokens[{i}]: {self.tokenizer.tokenize(sent)}")

        input_ids, labels, attention_mask = self.make_data(
            multi_turn_sentences=multi_turn_sentences,
            predict_parity=predict_parity
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


    def __len__(self):
        return self.length
