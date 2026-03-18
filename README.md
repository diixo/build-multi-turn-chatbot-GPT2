
# Build multi-turn Chatbot GPT-2
The following code trains a multi-turn chatbot model based on the **DailyDialog** dataset and pre-trained GPT-2 model.
<br><br><br>

### Supported Models
* Pre-trained GPT-2 from Hugging Face: ["gpt2"](https://huggingface.co/docs/transformers/model_doc/gpt2) from Hugging Face.
<br><br><br>


### Supported Tokenizer
* Pre-trained GPT-2 tokenizer from Hugging Face.
<br><br><br>

## Base Dataset
* [DailyDialog](http://yanran.li/dailydialog) multi-turn dataset.
* If you want to use your custom data, you have to set train/validation/test data paths in the `config/config.yaml`. Also, you have to implement your custom tokenizer, data loading parts in the `src/trainer/build.py`.
<br><br><br>


### Multi-turn GPT-2 Dialogue Results
"Q" represents user (your inputs) message, and "A" represents the chatbot's response.
```
Q: Have you ever been to France
A: Oh, yes! I went there once last summer.
Q: How was there?
A: It was amazing.
Q: Which spot was your best?
A: That of the Louvre.

Q: Did you see her yesterday?
A: Yes, I did. I saw her with a pretty girl.
Q: Oh where did you see her?                       
A: She was in a cafe. She told me she was going to a concert.
Q: I see. I've tried to contact her by phone, but she never received my call.
A: I'm sure she was in New York. She said she'd be in New York at the latest.
Q: Really? Thank you for your information. I will contact her today. Thanks.
```

<br><br><br>


## Supported Devices
* CPU, GPU, multi-GPU (DDP), MPS (for Mac and torch>=1.12.0)
<br><br>

## Quick Start
```bash
python3 src/run/train.py --config config/config.yaml --mode train
```
<br><br>


## Tutorials & Documentations
Please follow the steps below to train a multi-turn chatbot model.
1. [Getting Started](./docs/1_getting_started.md)
2. [Data Preparation](./docs/2_data_preparation.md)
3. [Training](./docs/3_trainig.md)
4. ETC
   * [Evaluation](./docs/4_model_evaluation.md)
   * [Live Demo](./docs/5_live_demo.md)

<br><br><br>


## Training Results
### Results of GPT-2-based Multi-turn Chatbot Model
* BLEU Score History<br>
<img src="docs/figs/bleu.jpg" width="80%"><br><br>

* NIST Score History<br>
<img src="docs/figs/nist.jpg" width="80%"><br><br>

* Test Set Scores<br>
The results of applying the model, which achieved the highest metric on the validation set, to the test set.
    * BLEU-2: 0.4052
    * BLEU-4: 0.2268
    * NIST-2: 5.1797
    * NIST-4: 5.5162
    <br><br>
