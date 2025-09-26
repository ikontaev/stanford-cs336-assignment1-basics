"""
Base tokenizer classes providing common functionality and interfaces.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from tokenization.preprocessing import pretokenize, tokenize


class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers.

    Provides common interface and functionality while allowing subclasses
    to implement specific training and merging algorithms.
    """

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        """
        Initialize tokenizer with vocabulary, merges, and special tokens.

        Args:
            vocab: Dictionary mapping token IDs to byte representations
            merges: List of BPE merge operations as (token1, token2) pairs
            special_tokens: List of special tokens that are never split
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Build reverse mappings for efficient lookup
        self.vocab_token_to_index = {v: k for k, v in vocab.items()}
        self.special_token_to_index = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes in self.vocab_token_to_index:
                self.special_token_to_index[token] = self.vocab_token_to_index[token_bytes]

    def encode(self, string: str) -> list[int]:
        """
        Encode a string into token indices.

        Args:
            string: Input string to encode

        Returns:
            List of token indices
        """
        tokens = pretokenize(string, self.special_tokens)
        indices = list(tokenize(tokens, self.vocab_token_to_index, self.special_token_to_index))
        indices = self._apply_merges(indices)
        return indices

    def decode(self, indices: list[int]) -> str:
        """
        Decode token indices back to a string.

        Args:
            indices: List of token indices to decode

        Returns:
            Decoded string
        """
        tokens = b"".join(self.vocab.get(index, b"") for index in indices)
        return tokens.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of strings, yielding token IDs lazily.

        Args:
            iterable: An iterable of strings (e.g., file lines)

        Yields:
            Token IDs one by one
        """
        for string_chunk in iterable:
            if string_chunk:  # Skip empty strings
                encoded_chunk = self.encode(string_chunk)
                yield from encoded_chunk

    def _apply_merges(self, indices: list[int]) -> list[int]:
        """
        Apply BPE merges to a list of token indices.

        Args:
            indices: Initial token indices before merges

        Returns:
            Token indices after applying all merges
        """
        # Apply merges in order. Merged tokens start at index 256
        for i, (token1_bytes, token2_bytes) in enumerate(self.merges):
            new_index = 256 + i
            token1_idx = self.vocab_token_to_index[token1_bytes]
            token2_idx = self.vocab_token_to_index[token2_bytes]
            indices = self._merge_pair(indices, (token1_idx, token2_idx), new_index)
        return indices

    @staticmethod
    def _merge_pair(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
        """
        Merge all instances of a token pair in the indices list.

        Args:
            indices: List of token indices
            pair: Tuple of (token1_idx, token2_idx) to merge
            new_index: New token index to replace the pair

        Returns:
            Updated indices with pairs merged
        """
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
    def build_vocab(special_tokens: list[str]) -> dict[int, bytes]:
        """
        Build initial vocabulary with special tokens and byte-level tokens.

        Args:
            special_tokens: List of special tokens to include

        Returns:
            Vocabulary mapping token indices to byte representations
        """
        vocab_index_to_token: dict[int, bytes] = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
        for i in range(len(special_tokens), 256):
            vocab_index_to_token[i] = bytes([i])
        return vocab_index_to_token

    @classmethod
    @abstractmethod
    def train(
        cls,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ) -> "BaseTokenizer":
        """
        Train a tokenizer on the given corpus.

        Args:
            input_path: Path to training data
            vocab_size: Total vocabulary size including special tokens
            special_tokens: List of special tokens

        Returns:
            Trained tokenizer instance
        """
        pass

