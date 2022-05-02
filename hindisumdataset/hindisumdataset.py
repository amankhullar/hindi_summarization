# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""
Hindi news summarization dataset.
Reference taken from: https://github.com/huggingface/datasets/blob/master/datasets/xsum/xsum.py
And: https://huggingface.co/docs/datasets/dataset_script
"""


import json
import os

import datasets

import pandas as pd


_CITATION = """
https://www.kaggle.com/datasets/disisbig/hindi-text-short-summarization-corpus
"""

_DESCRIPTION = """
Hindi Text Short Summarization Corpus is a collection of ~330k articles with their headlines collected from Hindi News Websites.

There are three features:
  - article: Input news article.
  - headline: Summary of the news article
  - id: BBC ID of the article.
"""

# From https://www.kaggle.com/datasets/disisbig/hindi-text-short-summarization-corpus
base_pth = '/content/drive/MyDrive/GaTech/NLP/hindi_summarization/'
_URL_DATA = "/content/drive/MyDrive/GaTech/NLP/hindi_summarization/archive-2"

_URL_SPLITS = {
    "train": os.path.join(_URL_DATA, "train.csv"),
    "dev": os.path.join(_URL_DATA, 'test.csv'),
}

_DOCUMENT = 'article'
_SUMMARY = 'headline'

_REMOVE_LINES = set(
    [
        "Share this with\n",
        "Email\n",
        "Facebook\n",
        "Messenger\n",
        "Twitter\n",
        "Pinterest\n",
        "WhatsApp\n",
        "Linkedin\n",
        "LinkedIn\n",
        "Copy this link\n",
        "These are external links and will open in a new window\n",
    ]
)


class HindiSum(datasets.GeneratorBasedBuilder):
    """Hindi News Article Summarization Dataset."""

    # Version 1.2.0 expands coverage, includes ids, and removes web contents.
    VERSION = datasets.Version("1.2.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _DOCUMENT: datasets.Value("string"),
                    _SUMMARY: datasets.Value("string"),
                }
            ),
            supervised_keys=(_DOCUMENT, _SUMMARY),
            homepage="https://www.kaggle.com/datasets/disisbig/hindi-text-short-summarization-corpus",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        # files_to_download = {"data": _URL_DATA, "splits": _URL_SPLITS}
        # downloaded_files = dl_manager.download(files_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split_path": _URL_SPLITS['train'],
                    "split_name": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split_path": _URL_SPLITS['dev'],
                    "split_name": "validation",
                },
            ),
        ]

    def _generate_examples(self, split_path, split_name):
        """Yields examples."""

        fdf = pd.read_csv(split_path)[:100000]
        articles = fdf['article'].fillna('').str.strip().tolist()
        summaries = fdf['headline'].fillna('').str.strip().tolist()

        for idx, (article, summary) in enumerate(zip(articles, summaries)):
            yield str(idx), {_DOCUMENT: article, _SUMMARY: summary}
