import heapq
import os
from collections import defaultdict

from tokenization.linked_list import DoublyLinkedList, ListNode
from tokenization.preprocessing import load_and_preprocess_data, pretokenize, tokenize


class BPETokenizerOptimized:
    """
    An optimized Byte Pair Encoding (BPE) tokenizer using heap and linked list for efficient training.

    This tokenizer has the same API as the naive BPETokenizer but uses optimized data structures
    for faster training on large corpora.
    """

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        """
        Args:
            vocab (dict[int, bytes]):
                A dictionary mapping integer token IDs to their corresponding byte representations.
                This defines the vocabulary of the tokenizer.
            merges (list[tuple[bytes, bytes]]):
                A list with BPE merge operations.
                Elements are tuples of two tokens (token1, token2) that were merged.
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
        """Build initial vocabulary with special tokens and byte-level tokens."""
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
        indices: list[int], vocab: dict[int, bytes], special_token_ids: set[int]
    ) -> tuple[int, int]:
        """Return the most frequent pair of tokens: (index1, index2)"""
        pairs_counter = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):
            if index1 in special_token_ids or index2 in special_token_ids:
                continue
            pairs_counter[(index1, index2)] += 1

        # If tie by frequency value, compare keys lexicographically
        pair = max(pairs_counter, key=lambda k: (pairs_counter[k], vocab[k[0]]))
        return pair

    @classmethod
    def train(
        cls,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ) -> "BPETokenizerOptimized":
        """Train a BPE tokenizer using optimized heap and linked list implementation.

        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.

        Returns:
            BPETokenizerOptimized
        """
        vocab_index_to_token = cls.build_vocab(special_tokens)
        token_sequence = load_and_preprocess_data(input_path, special_tokens, vocab_index_to_token)
        special_token_ids = set([i for i, _ in enumerate(special_tokens)])

        dll = DoublyLinkedList(token_sequence)

        # Initialize structures
        vocab = vocab_index_to_token
        merge_dict: dict[tuple[int, int], int] = {}

        # Initialize pair index and frequencies
        pair_index, pair_frequencies = cls._initialize_pair_index_and_counts(dll, special_token_ids)
        max_heap = cls._initialize_heap_structures(pair_frequencies, vocab)

        iterations = vocab_size - 256
        for i in range(iterations):
            # Get most frequent pair using heap - O(log n)
            try:
                index1, index2 = cls._get_most_frequent_pair_heap(max_heap, pair_frequencies)
            except ValueError:
                break  # No more pairs to merge

            # Skip if pair involves special tokens
            if index1 in special_token_ids or index2 in special_token_ids:
                continue

            # Perform incremental merge using linked list - O(k) where k = number of pair occurrences
            new_index = 256 + i
            merge_dict[(index1, index2)] = new_index
            vocab[new_index] = vocab[index1] + vocab[index2]

            # Merge in-place and update indices incrementally
            cls._incremental_merge_linked_list(
                max_heap, pair_frequencies, pair_index, dll, index1, index2, new_index, special_token_ids, vocab
            )

        merges = [(vocab[index1], vocab[index2]) for index1, index2 in merge_dict]
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _initialize_pair_index_and_counts(
        dll: DoublyLinkedList, special_token_ids: set[int]
    ) -> tuple[dict[tuple[int, int], set[ListNode]], dict[tuple[int, int], int]]:
        """Initialize pair index (node pointers) and frequency counts from linked list."""
        pair_index = defaultdict(set)
        pair_counts = defaultdict(int)
        current = dll.head
        while current and current.next:
            # Skip pairs involving special tokens (matching naive implementation)
            if current.token in special_token_ids or current.next.token in special_token_ids:
                current = current.next
                continue

            pair = (current.token, current.next.token)
            pair_index[pair].add(current)
            pair_counts[pair] += 1
            current = current.next
        return dict(pair_index), dict(pair_counts)

    @staticmethod
    def _negate_bytes(b: bytes) -> bytes:
        """Negate bytes for proper tie-breaking in min-heap to simulate max-heap behavior."""
        return bytes(255 - x for x in b)

    @staticmethod
    def _initialize_heap_structures(
        pair_frequencies: dict, vocab: dict[int, bytes]
    ) -> list[tuple[int, tuple[bytes, int, int]]]:
        """Creates a max-heap from pair frequencies for efficient retrieval."""
        # Python's heapq is a min-heap, so we store negative frequencies
        # to simulate a max-heap. For tie-breaking, we need to negate the bytes too
        # since max() chooses largest bytes but min-heap chooses smallest
        max_heap = [
            (-freq, (BPETokenizerOptimized._negate_bytes(vocab[pair[0]]), pair[0], pair[1]))
            for pair, freq in pair_frequencies.items()
        ]
        heapq.heapify(max_heap)
        return max_heap

    @staticmethod
    def _get_most_frequent_pair_heap(max_heap: list, pair_frequencies: dict) -> tuple[int, int]:
        """
        Extracts the most frequent pair from the heap using lazy deletion.

        Lazy Deletion: Some pairs in the heap may be "stale" (their true frequency
        has been updated). We pop from the heap and only return a pair if its
        frequency in the heap matches its current frequency in our source-of-truth dict.
        """
        while max_heap:
            neg_freq, (first_token_bytes, token1, token2) = heapq.heappop(max_heap)
            pair = (token1, token2)
            actual_freq = pair_frequencies.get(pair, 0)
            heap_freq = -neg_freq

            # Check if this heap entry is stale. If the frequency in our authoritative
            # map does not match, it means we have already updated this pair.
            # We discard it and check the next one.
            if actual_freq == heap_freq:
                return pair

        raise ValueError("Heap is empty, no more pairs to merge.")

    @staticmethod
    def _incremental_merge_linked_list(
        max_heap: list,
        pair_frequencies: dict[tuple[int, int], int],
        pair_index: dict[tuple[int, int], set[ListNode]],
        dll: DoublyLinkedList,
        idx1: int,
        idx2: int,
        new_idx: int,
        special_token_ids: set[int],
        vocab: dict[int, bytes],
    ) -> None:
        """
        Correctly merges ALL instances of a pair and updates frequencies incrementally.
        This version includes the fix for incorrect node removal from the pair_index.
        """
        pair_to_merge = (idx1, idx2)

        if pair_to_merge not in pair_index:
            return

        nodes_to_process = list(pair_index.pop(pair_to_merge, set()))
        pair_frequencies.pop(pair_to_merge, None)

        # Helper function to update pair frequencies and the heap
        def update_pair_frequency(pair, delta):
            # Do not track pairs involving special tokens
            if any(token in special_token_ids for token in pair):
                return

            current_freq = pair_frequencies.get(pair, 0)
            new_freq = current_freq + delta

            if new_freq > 0:
                pair_frequencies[pair] = new_freq
                heapq.heappush(
                    max_heap,
                    (-new_freq, (BPETokenizerOptimized._negate_bytes(vocab[pair[0]]), pair[0], pair[1])),
                )
            elif pair in pair_frequencies:
                del pair_frequencies[pair]

        for first_node in nodes_to_process:
            # Validity Check: Ensure this node is still part of the pair we intend to merge.
            # It might have been altered by a previous merge in this same loop (e.g., in a sequence like A-B-A-B).
            if not first_node.next or first_node.token != idx1 or first_node.next.token != idx2:
                continue

            second_node = first_node.next

            # Decrement counts of pairs that are about to be broken by this merge
            if first_node.prev:
                left_pair = (first_node.prev.token, idx1)
                update_pair_frequency(left_pair, -1)
                if left_pair in pair_index:
                    pair_index[left_pair].discard(first_node.prev)

            if second_node.next:
                right_pair = (idx2, second_node.next.token)
                update_pair_frequency(right_pair, -1)
                if right_pair in pair_index:
                    # The starting node for the right_pair is second_node.
                    pair_index[right_pair].discard(second_node)

            # Perform the merge in the linked list
            first_node.token = new_idx
            dll.remove_node(second_node)

            # Increment counts for new pairs that were just formed
            if first_node.prev:
                new_left_pair = (first_node.prev.token, new_idx)
                update_pair_frequency(new_left_pair, 1)
                if new_left_pair not in pair_index:
                    pair_index[new_left_pair] = set()
                pair_index[new_left_pair].add(first_node.prev)

            if first_node.next:
                new_right_pair = (new_idx, first_node.next.token)
                update_pair_frequency(new_right_pair, 1)
                if new_right_pair not in pair_index:
                    pair_index[new_right_pair] = set()
                pair_index[new_right_pair].add(first_node)
