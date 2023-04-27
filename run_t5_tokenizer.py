import os

import datasets
from transformers import T5Config
from underthesea import sent_tokenize, text_normalize, word_tokenize

from t5_tokenizer_model import SentencePieceUnigramTokenizer

corpus = 'vie_wikipedia_2021'
corpus_source_prefix = 'https://downloads.wortschatz-leipzig.de/corpora'

pretrained_name = 'google/flan-t5-base'
output_name = 'vi-flan-t5-base'

os.makedirs('data', exist_ok=True)
os.makedirs(os.path.join('outputs', output_name), exist_ok=True)
os.system(f'wget -P ./data/ {corpus_source_prefix}/{corpus}_300K.tar.gz')
os.system(f'wget -P ./data/ {corpus_source_prefix}/{corpus}_100K.tar.gz')
os.system(f'tar -xvzf ./data/{corpus}_300K.tar.gz -C ./data/')
os.system(f'tar -xvzf ./data/{corpus}_100K.tar.gz -C ./data/')

dataset = datasets.load_dataset("text",
                                data_files={'train': f"./data/{corpus}_300K/{corpus}_300K-sentences.txt"},
                                split='train')

tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")


def preprocess_function(texts):
    return map(
        lambda text: ' '.join(list(map(
            lambda s: ' '.join(word_tokenize(text_normalize(s))), sent_tokenize(text)
        ))), texts
    )


# Build an iterator over this dataset
def batch_iterator(input_sentence_size=None):
    if input_sentence_size is None:
        input_sentence_size = len(dataset)
    batch_length = 100
    for i in range(0, input_sentence_size, batch_length):
        yield preprocess_function(dataset[i: i + batch_length]["text"])


# Train tokenizer
tokenizer.train_from_iterator(
    iterator=batch_iterator(input_sentence_size=None),
    vocab_size=68_000,
    show_progress=True,
)
# Save files to disk
tokenizer.save(f"./outputs/{output_name}/tokenizer.json")

# Save config
config = T5Config.from_pretrained(pretrained_name, vocab_size=tokenizer.get_vocab_size())
config.save_pretrained(f"./outputs/{output_name}")
