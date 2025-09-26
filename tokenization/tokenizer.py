import os

from tokenization.base_tokenizer import BaseTokenizer
from tokenization.preprocessing import load_and_preprocess_data


class Tokenizer(BaseTokenizer):
    """
    A Byte Pair Encoding (BPE) tokenizer for encoding text into indices and decoding indices back to text.

    This is the naive implementation that uses simple list operations for training.
    """

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        """
        Initialize BPE tokenizer.

        Args:
            vocab: Dictionary mapping token IDs to byte representations
            merges: List of BPE merge operations as (token1, token2) pairs
            special_tokens: List of special tokens that are never split
        """
        super().__init__(vocab, merges, special_tokens)

    @classmethod
    def train(
        cls,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ) -> "Tokenizer":
        """Train a BPE tokenizer on given corpus

        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.

        Returns:
            Tokenizer
        """
        vocab_index_to_token = cls.build_vocab(special_tokens)
        indecis = list(load_and_preprocess_data(input_path, special_tokens, vocab_index_to_token))
        special_token_ids = set([i for i, _ in enumerate(special_tokens)])
        merge_dict: dict[tuple[int, int], int] = {}  # index1, index2 => merged_index
        iterations = vocab_size - 256
        for i in range(iterations):
            index1, index2 = cls._find_most_frequent_pair(indecis, vocab_index_to_token, special_token_ids)
            new_index = 256 + i
            merge_dict[(index1, index2)] = new_index
            vocab_index_to_token[new_index] = vocab_index_to_token[index1] + vocab_index_to_token[index2]
            indecis = cls._merge_pair(indecis, (index1, index2), new_index)
        merges = [(vocab_index_to_token[index1], vocab_index_to_token[index2]) for index1, index2 in merge_dict]
        return cls(vocab_index_to_token, merges, special_tokens)

    @staticmethod
    def _find_most_frequent_pair(
        indices: list[int], vocab: dict[int, bytes], special_token_ids: set[int]
    ) -> tuple[int, int]:
        """
        Find the most frequent pair of adjacent tokens.

        Args:
            indices: List of token indices
            vocab: Vocabulary for tie-breaking
            special_token_ids: Set of special token IDs to skip

        Returns:
            Most frequent pair (token1_idx, token2_idx)
        """
        from collections import defaultdict

        pairs_counter = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):
            if index1 in special_token_ids or index2 in special_token_ids:
                continue
            pairs_counter[(index1, index2)] += 1

        # If tie by frequency value, compare keys lexicographically
        pair = max(pairs_counter, key=lambda k: (pairs_counter[k], vocab[k[0]]))
        return pair
