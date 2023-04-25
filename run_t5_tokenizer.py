import datasets

from t5_tokenizer_model import SentencePieceUnigramTokenizer


vocab_size = 36_100
input_sentence_size = None

# Initialize a dataset
dataset = data.load_dataset("viwiki", data_files="./data/vie_wikipedia_2021_1M/vie_wikipedia_2021_1M-sentences.txt")

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
    iterator=batch_iterator(input_sentence_size=input_sentence_size),
    vocab_size=vocab_size,
    show_progress=True,
)

# Save files to disk
tokenizer.save("./outputs/tokenizer.json")