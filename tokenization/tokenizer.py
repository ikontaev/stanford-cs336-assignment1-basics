import os
from collections import defaultdict

from tokenization.preprocessing import load_and_preprocess_data, pretokenize, tokenize


class BPETokenizer:
    """
    A Byte Pair Encoding (BPE) tokenizer for encoding text into indices and decoding indices back to text.

    It is initialized with a vocabulary, merge rules, and a list of special tokens.
    """

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        """
        Args:
            vocab (dict[int, bytes]):
                A dictionary mapping integer token IDs to their corresponding byte representations.
                This defines the vocabulary of the tokenizer.
            merges (list[tuple[bytes,bytes]):
                A list with BPE merge operations.
                Elements are tuples of two tokens (token1, token2) that were merged,
            special_tokens (list[str]):
                A list of special tokens (e.g., ["<|endoftext|>", "<|startoftext|>"]).
                These tokens are treated as indivisible units and are not split during tokenization.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.vocab_token_to_index = {v: k for k, v in vocab.items()}
        self.special_token_to_index = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes in self.vocab_token_to_index:
                self.special_token_to_index[token] = self.vocab_token_to_index[token_bytes]

    def encode(self, string: str) -> list[int]:
        """Encode a string into token indices."""
        tokens = pretokenize(string, self.special_tokens)
        indices = tokenize(tokens, self.vocab_token_to_index, self.special_token_to_index)
        # Apply merges in order. Merged tokens start at index 256
        for i, (token1_bytes, token2_bytes) in enumerate(self.merges):
            new_index = 256 + i
            token1_idx = self.vocab_token_to_index[token1_bytes]
            token2_idx = self.vocab_token_to_index[token2_bytes]

            indices = self.merge(indices, (token1_idx, token2_idx), new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        """Decode token indices back to a string."""
        tokens = b"".join(self.vocab.get(index, b"") for index in indices)
        return tokens.decode("utf-8", errors="replace")

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
        merge_dict: dict[tuple[int, int], int] = {}  # index1, index2 => merged_index
        iterations = vocab_size - 256
        for i in range(iterations):
            index1, index2 = BPETokenizer.find_most_frequent_pair(indecis, vocab_index_to_token, special_token_ids)
            new_index = 256 + i
            merge_dict[(index1, index2)] = new_index
            vocab_index_to_token[new_index] = vocab_index_to_token[index1] + vocab_index_to_token[index2]
            indecis = BPETokenizer.merge(indecis, (index1, index2), new_index)
        merges = [(vocab_index_to_token[index1], vocab_index_to_token[index2]) for index1, index2 in merge_dict]
        return BPETokenizer(vocab_index_to_token, merges, special_tokens)
