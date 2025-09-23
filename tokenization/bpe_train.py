import json
import os
from collections import defaultdict
from dataclasses import dataclass

from preprocessing import build_vocab, load_and_preprocess_data, pretokenize, tokenize


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""

    vocab: dict[int, bytes]  # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index
    special_tokens: list[str]  # List of special tokens (e.g., ["<|endoftext|>"])


class BPETokenizer:
    """BPE tokenizer given a set of merges and a vocabulary."""

    def __init__(self, params: BPETokenizerParams):
        self.params = params
        self.vocab_index_to_token = params.vocab
        self.vocab_token_to_index = {v: k for k, v in params.vocab.items()}
        self.special_token_to_index = {token: i for i, token in enumerate(params.special_tokens)}

    def encode(self, string: str) -> list[int]:
        """Encode a string into token indices."""
        tokens = pretokenize(string, self.params.special_tokens)
        indices = tokenize(tokens, self.vocab_token_to_index, self.special_token_to_index)
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        """Decode token indices back to a string."""
        tokens = []
        for index in indices:
            token_bytes = self.vocab_index_to_token.get(index, b"")
            tokens.append(token_bytes.decode("utf-8", errors="replace"))
        return "".join(tokens)


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


def find_most_frequent_pair(indecis: list[int], vocab) -> tuple[int, int]:
    """Return the most frequent pair of tokens: (index1, index2)"""
    pairs_counter = defaultdict(int)
    for index1, index2 in zip(indecis, indecis[1:]):
        pairs_counter[(index1, index2)] += 1

    # if tie by frequency value, compare keys lexicograhically
    pair = max(pairs_counter, key=lambda k: (pairs_counter[k], vocab[k[0]]))
    return pair


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocab_index_to_token = build_vocab(special_tokens)
    indecis = load_and_preprocess_data(input_path, special_tokens, vocab_index_to_token)
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged_index
    iterations = vocab_size - len(vocab_index_to_token)
    for i in range(iterations + 1):
        index1, index2 = find_most_frequent_pair(indecis, vocab_index_to_token)
        new_index = 256 + i
        merges[(index1, index2)] = new_index
        vocab_index_to_token[new_index] = vocab_index_to_token[index1] + vocab_index_to_token[index2]
        indecis = merge(indecis, (index1, index2), new_index)
    return vocab_index_to_token, merges


def bytes_to_str(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.hex()


def save_vocab(filename, vocab):
    vocab_inv = {}
    for k, v in vocab.items():
        key_str = bytes_to_str(v)
        vocab_inv[key_str] = k

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(vocab_inv, f, indent=4, ensure_ascii=False)


def save_merges(path: str, merges: list[tuple[bytes, bytes]], vocab: dict[int, bytes]):
    with open(path, "w") as f:
        for i, (token1, token2) in enumerate(merges):
            f.write(f"{i}: {bytes_to_str(vocab[token1])} {bytes_to_str(vocab[token2])}\n")


def main():
    """Performance comparison between different BPE implementations."""
    import cProfile
    import time

    # input_file = "./data/fake_file.txt"
    # input_file = "./tests/fixtures/corpus.en"
    # input_file = "./tests/fixtures/german.txt"
    input_file = "./tests/fixtures/tinystories_sample.txt"
    # input_file = "./data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    # special_tokens = []
    #
    profiler_naive = cProfile.Profile()
    start_time = time.time()
    #
    profiler_naive.enable()
    vocab_naive, merges_naive = train_bpe(input_file, vocab_size, special_tokens)
    profiler_naive.disable()
    #
    naive_time = time.time() - start_time
    print(f"Naive implementation time: {naive_time:.3f} seconds")
    # vocab = {k: v.decode("utf-8") for k, v in vocab_naive.items()}
    bpe_tokenizer_naive = BPETokenizer(BPETokenizerParams(vocab_naive, merges_naive, special_tokens))
    string = "the quick fox jump over grumpy dog <|endoftext|> hello again"
    tokens = bpe_tokenizer_naive.encode(string)
    print(tokens)
    print(bpe_tokenizer_naive.decode(tokens))

    save_merges("./dump/corpus_merges_naive.txt", merges_naive, vocab_naive)
    save_vocab("./dump/corpus_vocab_naive.json", vocab_naive)


if __name__ == "__main__":
    main()
