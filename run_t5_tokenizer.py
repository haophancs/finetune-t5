import os
import sys
import datasets

from t5_tokenizer_model import SentencePieceUnigramTokenizer

corpus = 'vie_wikipedia_2021_1M'
os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.system(f'wget -P ./data/ https://downloads.wortschatz-leipzig.de/corpora/{corpus}.tar.gz')
os.system(f'tar -xvzf ./data/{corpus}.tar.gz -C ./data/')
dataset = datasets.load_dataset("text", data_files={'train': f"./data/{corpus}/{corpus}-sentences.txt"}, split='train')
tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")


# Build an iterator over this dataset
def batch_iterator(input_sentence_size=None):
    if input_sentence_size is None:
        input_sentence_size = len(dataset)
    batch_length = 100
    for i in range(0, input_sentence_size, batch_length):
        yield dataset[i: i + batch_length]["text"]


# Train tokenizer
tokenizer.train_from_iterator(
    iterator=batch_iterator(input_sentence_size=None),
    vocab_size=32_000,
    show_progress=True,
)
# Save files to disk
tokenizer.save("./outputs/tokenizer.json")
