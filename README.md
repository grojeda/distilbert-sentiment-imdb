---
language: en
license: apache-2.0
datasets:
- imdb
metrics:
- accuracy
base_model: distilbert-base-uncased
pipeline_tag: text-classification
tags:
- sentiment-analysis
- imdb
- text-classification
- distilbert
library_name: transformers
---

## Model Description
`grojeda/distilbert-sentiment-imdb` fine-tunes `distilbert-base-uncased` for binary sentiment classification over the IMDB Large Movie Review dataset. Reviews are tokenized with the base WordPiece tokenizer, truncated to 384 tokens, and dynamically padded via `DataCollatorWithPadding`. Training uses Hugging Face Transformers (PyTorch) and produces weights that inherit the Apache 2.0 terms of the base checkpoint and dataset license constraints.

## Intended Uses & Limitations
- Recommended for English sentiment analysis of movie-style product reviews, benchmarking compact BERT-family encoders, and serving as a starting point for domain adaptation.
- Not suitable for multilingual sentiment, sarcasm detection, fine-grained emotion tagging, or high-stakes moderation without human review.
- Known limitations include potential degradation on inputs longer than 384 tokens, sensitivity to domain shifts (legal/medical jargon), and lack of calibration for imbalanced datasets.
- Ethical considerations: the model may reproduce societal biases present in IMDB reviews; avoid using outputs for demographic inference or automated enforcement without auditing.

## Training Details
- **Dataset:** `imdb` from Hugging Face Datasets. The 25k training split was partitioned 90/10 (seed 42) into train (≈22.5k) and validation (≈2.5k). The official 25k test split remained untouched until evaluation.
- **Hyperparameters:** learning rate 2e-5 with linear decay, weight decay 0.01, AdamW optimizer, max length 384, per-device batch sizes 16 (train) / 32 (eval), no gradient accumulation, dropout per DistilBERT defaults.
- **Schedule:** 2 epochs (~2,814 optimizer steps). Evaluation and checkpointing occur at each epoch with best-checkpoint reloading enabled.
- **Compute:** Trained on a single consumer GPU (RTX 3050 4GB). Environment: Transformers 4.57.3 and PyTorch 2.9.1.

## Evaluation Results
```json
{
  "task": {
    "type": "text-classification",
    "name": "Sentiment Analysis"
  },
  "dataset": {
    "name": "imdb",
    "type": "imdb",
    "split": "test"
  },
  "metrics": [
    {
      "type": "accuracy",
      "name": "Accuracy",
      "value": 0.9256
    },
    {
      "type": "loss",
      "name": "CrossEntropyLoss",
      "value": 0.2435
    }
  ]
}
```

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "grojeda/distilbert-sentiment-imdb"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "The pacing drags, but the performances are heartfelt."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    probs = model(**inputs).logits.softmax(dim=-1)

label_id = int(probs.argmax())
print(model.config.id2label[label_id], float(probs[0, label_id]))
```

## Limitations & Biases
- Domain bias toward movie reviews; expect weaker performance on other domains without fine-tuning.
- Only English data was used; multilingual inputs are unsupported.
- Dataset balance (50/50) can lead to overconfidence when deployed on skewed class distributions.
- User-generated content may embed stereotypes or offensive language that the model can echo.

## Citation
```
@article{sanh2019distilbert,
  title   = {DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter},
  author  = {Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf},
  journal = {NeurIPS EMC2 Workshop},
  year    = {2019}
}

@inproceedings{maas2011learning,
  title     = {Learning Word Vectors for Sentiment Analysis},
  author    = {Andrew L. Maas and Raymond E. Daly and Peter T. Pham and Dan Huang and Andrew Y. Ng and Christopher Potts},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  year      = {2011}
}
```