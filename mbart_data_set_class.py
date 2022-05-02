"""
This script exhibits a previous attempt in creating a custom dataset class.
Please use the script hindisumdataset/hindisumdataset.py for creating custom dataset classes for seq2seq models.
"""
import os
import sys
import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import MBartForConditionalGeneration, MBartTokenizer

class HindiSumDataset(Dataset):
    def __init__(self, article_encodings, summary_encodings):
        self.article_encodings = article_encodings
        self.summary_encodings = summary_encodings

    def __len__(self):
        return len(self.article_encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.article_encodings.items()}
        item['labels'] = self.summary_encodings['input_ids']
        return item

def main(train_df, test_df):
    train_articles, train_summaries = train_df['article'].fillna('').tolist(), train_df['headline'].fillna('').tolist()
    test_articles, test_summaries = test_df['article'].fillna('').tolist(), test_df['headline'].fillna('').tolist()

    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="hi_IN", tgt_lang="hi_IN")
    max_input_length = 1024
    max_target_length = 256

    train_articles_encodings = tokenizer(train_articles, truncation=True, padding=True, max_length=max_input_length)
    with tokenizer.as_target_tokenizer():
        train_summaries_encodings = tokenizer(train_summaries, truncation=True, padding=True, max_length=max_target_length)

    test_articles_encodings = tokenizer(test_articles, truncation=True, padding=True, max_length=max_input_length)
    with tokenizer.as_target_tokenizer():
        test_summaries_encodings = tokenizer(test_summaries, truncation=True, padding=True, max_length=max_target_length)

    train_dataset = HindiSumDataset(train_articles_encodings, train_summaries_encodings)
    test_dataset = HindiSumDataset(test_articles_encodings, test_summaries_encodings)

if __name__ == "__main__":
    base_pth = os.getcwd()
    train_pth = os.path.join(base_pth, 'archive-2', 'train.csv')
    test_pth = os.path.join(base_pth, 'archive-2', 'test.csv')

    train_df = pd.read_csv(train_pth)
    test_df = pd.read_csv(test_pth)

    main(train_df, test_df)
