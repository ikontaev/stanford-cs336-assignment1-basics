import heapq
import os
from collections import defaultdict

from .base_tokenizer import BaseTokenizer
from .linked_list import DoublyLinkedList, ListNode
from .preprocessing import load_and_preprocess_data
from tqdm import tqdm


class TokenizerOptimized(BaseTokenizer):
    """
    An optimized Byte Pair Encoding (BPE) tokenizer using heap and linked list for efficient training.

    This tokenizer uses optimized data structures (heap and doubly-linked list) for faster training
    on large corpora compared to the naive implementation.
    """

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        """
        Initialize optimized BPE tokenizer.

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
    ) -> "TokenizerOptimized":
        """Train a BPE tokenizer using optimized heap and linked list implementation.

        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.

        Returns:
            TokenizerOptimized
        """
        vocab_index_to_token = cls.build_vocab(special_tokens)
        token_sequence = list(load_and_preprocess_data(input_path, special_tokens, vocab_index_to_token))
        special_token_ids = set([i for i, _ in enumerate(special_tokens)])

        dll = DoublyLinkedList(token_sequence)

        # Initialize structures
        vocab = vocab_index_to_token
        merge_dict: dict[tuple[int, int], int] = {}

        # Initialize pair index and frequencies
        pair_index, pair_frequencies = cls._initialize_pair_index_and_counts(dll, special_token_ids)
        max_heap = cls._initialize_heap_structures(pair_frequencies)

        node_id_map = {}
        current = dll.head
        while current:
            node_id_map[current.id] = current
            current = current.next

        iterations = vocab_size - 256
        for i in tqdm(range(iterations)):
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
                max_heap,
                pair_frequencies,
                pair_index,
                dll,
                index1,
                index2,
                new_index,
                special_token_ids,
                node_id_map,
            )

        merges = [(vocab[index1], vocab[index2]) for index1, index2 in merge_dict]
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _initialize_pair_index_and_counts(
        dll: DoublyLinkedList, special_token_ids: set[int]
    ) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], int]]:
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
            pair_index[pair].add(current.id)
            pair_counts[pair] += 1
            current = current.next
        return dict(pair_index), dict(pair_counts)

    @staticmethod
    def _initialize_heap_structures(pair_frequencies: dict) -> list[tuple[int, int, int]]:
        """Creates a max-heap from pair frequencies for efficient retrieval."""
        # Python's heapq is a min-heap, so we store negative frequencies
        # to simulate a max-heap. For tie-breaking, we use negative token IDs
        max_heap = [(-freq, -pair[0], pair[1]) for pair, freq in pair_frequencies.items()]
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
            neg_freq, neg_token1, token2 = heapq.heappop(max_heap)
            token1 = -neg_token1  # Convert back from negative
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
        pair_index: dict[tuple[int, int], set[int]],
        dll: DoublyLinkedList,
        idx1: int,
        idx2: int,
        new_idx: int,
        special_token_ids: set[int],
        node_id_map: dict[int, ListNode],
    ) -> None:
        """Correctly merges ALL instances of a pair and updates frequencies incrementally.
        This version includes the fix for incorrect node removal from the pair_index.
        """
        pair_to_merge = (idx1, idx2)

        if pair_to_merge not in pair_index:
            return

        node_ids_to_process = list(pair_index.pop(pair_to_merge, set()))
        pair_frequencies.pop(pair_to_merge, None)

        # Helper function to update pair frequencies and the heap
        def update_pair_frequency(pair, delta):
            # Do not track pairs involving special tokens
            if pair[0] in special_token_ids or pair[1] in special_token_ids:
                return

            current_freq = pair_frequencies.get(pair, 0)
            new_freq = current_freq + delta

            if new_freq > 0:
                pair_frequencies[pair] = new_freq
                heapq.heappush(
                    max_heap,
                    (-new_freq, -pair[0], pair[1]),
                )
            elif pair in pair_frequencies:
                del pair_frequencies[pair]

        for node_id in node_ids_to_process:
            # Look up the actual node from the ID
            first_node = node_id_map.get(node_id)
            if first_node is None:
                continue  # Node may have been removed in a previous merge
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
                    pair_index[left_pair].discard(first_node.prev.id)

            if second_node.next:
                right_pair = (idx2, second_node.next.token)
                update_pair_frequency(right_pair, -1)
                if right_pair in pair_index:
                    # The starting node for the right_pair is second_node.
                    pair_index[right_pair].discard(second_node.id)

            # Perform the merge in the linked list
            first_node.token = new_idx
            dll.remove_node(second_node)

            # Maintain node_id_map by removing the deleted node
            node_id_map.pop(second_node.id, None)

            # Increment counts for new pairs that were just formed
            if first_node.prev:
                new_left_pair = (first_node.prev.token, new_idx)
                update_pair_frequency(new_left_pair, 1)
                if new_left_pair not in pair_index:
                    pair_index[new_left_pair] = set()
                pair_index[new_left_pair].add(first_node.prev.id)

            if first_node.next:
                new_right_pair = (new_idx, first_node.next.token)
                update_pair_frequency(new_right_pair, 1)
                if new_right_pair not in pair_index:
                    pair_index[new_right_pair] = set()
                pair_index[new_right_pair].add(first_node.id)
