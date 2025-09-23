import os
from collections import defaultdict

from preprocessing import load_and_preprocess_data, pretokenize, tokenize


class BPETokenizer:
    """
    A Byte Pair Encoding (BPE) tokenizer for encoding text into indices and decoding indices back to text.

    It is initialized with a vocabulary, merge rules, and a list of special tokens.
    """

    def __init__(self, vocab: dict[int, bytes], merges: dict[tuple[int, int], int], special_tokens: list[str]):
        """
        Args:
            vocab (dict[int, bytes]):
                A dictionary mapping integer token IDs to their corresponding byte representations.
                This defines the vocabulary of the tokenizer.
            merges (dict[tuple[int, int], int]):
                A dictionary representing the BPE merge operations.
                Keys are tuples of two token IDs (index1, index2) that were merged,
                and values are the new token ID created by the merge.
            special_tokens (list[str]):
                A list of special tokens (e.g., ["<|endoftext|>", "<|startoftext|>"]).
                These tokens are treated as indivisible units and are not split during tokenization.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_token_to_index = {v: k for k, v in vocab.items()}
        self.special_token_to_index = {token: i for i, token in enumerate(special_tokens)}

    def encode(self, string: str) -> list[int]:
        """Encode a string into token indices."""
        tokens = pretokenize(string, self.special_tokens)
        indices = tokenize(tokens, self.vocab_token_to_index, self.special_token_to_index)
        # Note: this is a very slow implementation
        for pair, new_index in self.merges.items():
            indices = self.merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        """Decode token indices back to a string."""
        tokens = []
        for index in indices:
            token_bytes = self.vocab.get(index, b"")
            tokens.append(token_bytes.decode("utf-8", errors="replace"))
        return "".join(tokens)

    @staticmethod
    def build_vocab(special_tokens: list[str]) -> dict[int, bytes]:
        vocab_index_to_token: dict[int, bytes] = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
        for i in range(len(special_tokens), 256):
            vocab_index_to_token[i] = bytes([i])
        return vocab_index_to_token

    @staticmethod
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

    @staticmethod
    def find_most_frequent_pair(
        indecis: list[int], vocab: dict[int, bytes], special_token_ids: set[int]
    ) -> tuple[int, int]:
        """Return the most frequent pair of tokens: (index1, index2)"""
        pairs_counter = defaultdict(int)
        for index1, index2 in zip(indecis, indecis[1:]):
            if index1 in special_token_ids or index2 in special_token_ids:
                continue
            pairs_counter[(index1, index2)] += 1

        # If tie by frequency value, compare keys lexicograhically
        pair = max(pairs_counter, key=lambda k: (pairs_counter[k], vocab[k[0]]))
        return pair

    @classmethod
    def train(
        cls,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ) -> "BPETokenizer":
        """Train a BPE tokenizer on given corpus

        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.

        Returns:
            BPETokenizer
        """

        vocab_index_to_token = BPETokenizer.build_vocab(special_tokens)
        indecis = load_and_preprocess_data(input_path, special_tokens, vocab_index_to_token)
        special_token_ids = set([i for i, _ in enumerate(special_tokens)])
        merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged_index
        iterations = vocab_size - len(vocab_index_to_token)
        for i in range(iterations + 1):
            index1, index2 = BPETokenizer.find_most_frequent_pair(indecis, vocab_index_to_token, special_token_ids)
            new_index = 256 + i
            merges[(index1, index2)] = new_index
            vocab_index_to_token[new_index] = vocab_index_to_token[index1] + vocab_index_to_token[index2]
            indecis = BPETokenizer.merge(indecis, (index1, index2), new_index)
        return BPETokenizer(vocab_index_to_token, merges, special_tokens)
