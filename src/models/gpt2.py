from transformers import GPT2LMHeadModel

import torch
import torch.nn as nn

from utils import LOGGER, colorstr



class GPT2(nn.Module):

    def __init__(self, config, tokenizer):
        super(GPT2, self).__init__()
        self.pretrained_model = config.pretrained_model
        self.model = GPT2LMHeadModel.from_pretrained(self.pretrained_model, attn_implementation="eager")
        self.model.resize_token_embeddings(config.vocab_size, mean_resizing=False)
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id


    def make_mask(self, x):
        pad_mask = torch.where(x==self.pad_token_id, 0, 1)
        return pad_mask


    def forward(self, x):
        pad_mask = self.make_mask(x)
        output = self.model(input_ids=x, attention_mask=pad_mask)
        return output.logits
    

    def batch_inference(self, src, start_tokens, max_len, tokenizer, loss_func=None, target=None):
        loss = None
        if loss_func:
            assert target != None, LOGGER(colorstr('red', 'Target must be required if you want to return loss values..'))
            output = self.forward(src)
            loss = loss_func(output[:, :-1, :].reshape(-1, output.size(-1)), target[:, 1:].reshape(-1))
        
        if isinstance(start_tokens, tuple):
            st, stl = start_tokens
            start_tokens = [single_s[:single_sl].unsqueeze(0) for single_s, single_sl in zip(st, stl)]
        else:
            start_tokens = [start_tokens.unsqueeze(1)]
        
        # Due to decoder-only architecture, token length of every single batch is different
        preds = []
        for start_token in start_tokens:
            while start_token.size(1) < max_len:
                output = self.forward(start_token)
                start_token = torch.cat((start_token, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
            preds.append(start_token[0])

        predictions = [tokenizer.decode(pred.detach().cpu().tolist()) for pred in preds]

        return predictions, loss


    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=64,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
        num_beams=1,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        return_only_new_tokens=False,
    ):
        """
        input_ids: LongTensor [batch, seq_len] or [seq_len]
        """

        self.model.eval()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        attention_mask = self.make_mask(input_ids)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }

        if do_sample:
            if temperature is not None:
                generate_kwargs["temperature"] = temperature
            if top_k is not None:
                generate_kwargs["top_k"] = top_k
            if top_p is not None:
                generate_kwargs["top_p"] = top_p

        output_ids = self.model.generate(**generate_kwargs)

        if return_only_new_tokens:
            output_ids = output_ids[:, input_ids.size(1):]

        return output_ids
