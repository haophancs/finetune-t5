import os
import sys
import datasets

from t5_tokenizer_model import SentencePieceUnigramTokenizer

corpus = 'vie_wikipedia_2021'
corpus_source_prefix = 'https://downloads.wortschatz-leipzig.de/corpora'

os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.system(f'wget -P ./data/ {corpus_source_prefix}/{corpus}_1M.tar.gz')
os.system(f'wget -P ./data/ {corpus_source_prefix}/{corpus}_300K.tar.gz')
os.system(f'tar -xvzf ./data/{corpus}_1M.tar.gz -C ./data/')
os.system(f'tar -xvzf ./data/{corpus}_300K.tar.gz -C ./data/')

dataset = datasets.load_dataset("text", data_files={'train': f"./data/{corpus}_1M/{corpus}_1M-sentences.txt"}, split='train')
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
