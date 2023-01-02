import re
from collections import Counter

class SimpleTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def fit(self, texts):
        words = []
        for text in texts:
            words.extend(re.findall(r'\w+', text.lower()))
        common_words = Counter(words).most_common(self.vocab_size - 4)
        for i, (word, _) in enumerate(common_words):
            self.word2idx[word] = i + 4
            self.idx2word[i + 4] = word

    def encode(self, text, max_len=32):
        tokens = re.findall(r'\w+', text.lower())
        encoded = [self.word2idx.get(t, 1) for t in tokens]
        encoded = [2] + encoded[:max_len-2] + [3]
        return encoded + [0] * (max_len - len(encoded))

    def decode(self, ids):
        return " ".join([self.idx2word.get(i, "<UNK>") for i in ids if i > 3])
