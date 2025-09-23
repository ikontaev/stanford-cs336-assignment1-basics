import heapq
import os
from collections import defaultdict

from preprocessing import load_and_preprocess_data, pretokenize, tokenize


class ListNode:
    """Node for doubly linked list representation of token sequence."""

    def __init__(self, token: int):
        self.token = token
        self.prev: ListNode | None = None
        self.next: ListNode | None = None

    def __repr__(self):
        return f"Node({self.token})"

    def __eq__(self, other):
        if not isinstance(other, ListNode):
            return False
        return self.token == other.token

    def __hash__(self):
        return hash(self.token)


class DoublyLinkedList:
    """Doubly linked list for efficient in-place token sequence modifications."""

    def __init__(self, tokens: list[int]):
        self.head: ListNode | None = None
        self.tail: ListNode | None = None
        self.size = 0

        # Build the linked list from token sequence
        for token in tokens:
            self.append(token)

    def append(self, token: int) -> ListNode:
        """Add a token to the end of the list and return the new node."""
        new_node = ListNode(token)
        if self.head is None:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
        return new_node

    def remove_node(self, node: ListNode) -> None:
        """Remove a node from the list in O(1) time."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

        self.size -= 1

    def insert_after(self, node: ListNode, token: int) -> ListNode:
        """Insert a new token after the given node and return the new node."""
        new_node = ListNode(token)
        new_node.next = node.next
        new_node.prev = node

        if node.next:
            node.next.prev = new_node
        else:
            self.tail = new_node

        node.next = new_node
        self.size += 1
        return new_node

    def to_list(self) -> list[int]:
        """Convert linked list back to regular list for debugging/output."""
        result = []
        current = self.head
        while current:
            result.append(current.token)
            current = current.next
        return result


def initialize_pair_index_and_counts(
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


def negate_bytes(b: bytes) -> bytes:
    """Negate bytes for proper tie-breaking in min-heap to simulate max-heap behavior."""
    return bytes(255 - x for x in b)


def initialize_heap_structures(
    pair_frequencies: dict, vocab: dict[int, bytes]
) -> list[tuple[int, tuple[bytes, int, int]]]:
    """Creates a max-heap from pair frequencies for efficient retrieval."""
    # Python's heapq is a min-heap, so we store negative frequencies
    # to simulate a max-heap. For tie-breaking, we need to negate the bytes too
    # since max() chooses largest bytes but min-heap chooses smallest
    max_heap = [(-freq, (negate_bytes(vocab[pair[0]]), pair[0], pair[1])) for pair, freq in pair_frequencies.items()]
    heapq.heapify(max_heap)
    return max_heap


def get_most_frequent_pair_heap(max_heap: list, pair_frequencies: dict) -> tuple[int, int]:
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


def incremental_merge_linked_list(
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
    """
    pair_to_merge = (idx1, idx2)

    # 1. Get all nodes to process and REMOVE the pair from our data structures upfront.
    # This prevents the pair from being considered again. We copy to a list to iterate safely.
    if pair_to_merge not in pair_index:
        return
    nodes_to_process = list(pair_index.pop(pair_to_merge))
    pair_frequencies.pop(pair_to_merge, None)

    for first_node in nodes_to_process:
        # 2. Validity Check: Ensure this node hasn't been modified by a previous
        # merge in this same loop (e.g., in a sequence like A-B-A-B).
        if not first_node.next or first_node.token != idx1 or first_node.next.token != idx2:
            continue

        second_node = first_node.next

        # Helper function to update frequencies and heap
        def update_pair(pair, delta):
            if any(token in special_token_ids for token in pair):
                return

            # Update frequency
            current_freq = pair_frequencies.get(pair, 0)
            new_freq = current_freq + delta

            if new_freq > 0:
                pair_frequencies[pair] = new_freq
                heapq.heappush(
                    max_heap,
                    (-new_freq, (negate_bytes(vocab[pair[0]]), pair[0], pair[1])),
                )
            elif pair in pair_frequencies:
                # If frequency drops to 0 or below, remove it
                del pair_frequencies[pair]
                # No need to remove from pair_index as we handle it contextually

        # 3. Decrement counts of pairs that are about to be broken
        if first_node.prev:
            left_pair = (first_node.prev.token, idx1)
            update_pair(left_pair, -1)
            pair_index.get(left_pair, set()).discard(first_node.prev)

        if second_node.next:
            right_pair = (idx2, second_node.next.token)
            update_pair(right_pair, -1)
            # The starting node for this pair was `second_node`, which will be removed.
            # We discard `first_node` because after the merge, it will represent the new token.
            pair_index.get(right_pair, set()).discard(second_node)

        # 4. Perform the merge on the linked list
        first_node.token = new_idx
        dll.remove_node(second_node)

        # 5. Increment counts for new pairs that were just formed
        if first_node.prev:
            new_left_pair = (first_node.prev.token, new_idx)
            update_pair(new_left_pair, 1)
            if new_left_pair not in pair_index:
                pair_index[new_left_pair] = set()
            pair_index[new_left_pair].add(first_node.prev)

        if first_node.next:
            new_right_pair = (new_idx, first_node.next.token)
            update_pair(new_right_pair, 1)
            if new_right_pair not in pair_index:
                pair_index[new_right_pair] = set()
            pair_index[new_right_pair].add(first_node)


def build_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """Build initial vocabulary with special tokens and byte-level tokens."""
    vocab_index_to_token: dict[int, bytes] = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    for i in range(len(special_tokens), 256):
        vocab_index_to_token[i] = bytes([i])
    return vocab_index_to_token


def find_most_frequent_pair_heap(
    indices: list[int], vocab: dict[int, bytes], special_token_ids: set[int]
) -> tuple[int, int]:
    """Find most frequent pair using heap-based counting (for compatibility with naive version)."""
    pairs_counter = defaultdict(int)
    for index1, index2 in zip(indices, indices[1:]):
        if index1 in special_token_ids or index2 in special_token_ids:
            continue
        pairs_counter[(index1, index2)] += 1

    # If tie by frequency value, compare keys lexicographically
    pair = max(pairs_counter, key=lambda k: (pairs_counter[k], vocab[k[0]]))
    return pair


class BPETokenizerOptimized:
    """
    An optimized Byte Pair Encoding (BPE) tokenizer using heap and linked list for efficient training.

    This tokenizer has the same API as the naive BPETokenizer but uses optimized data structures
    for faster training on large corpora.
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
        # Apply merges in order
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
        """Build initial vocabulary with special tokens and byte-level tokens."""
        return build_vocab(special_tokens)

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
        return find_most_frequent_pair_heap(indices, vocab, special_token_ids)

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
        vocab, merges = train_bpe_linked_list_optimized(input_path, vocab_size, special_tokens)
        return cls(vocab, merges, special_tokens)


def train_bpe_linked_list_optimized(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], dict[tuple[int, int], int]]:
    """Linked list BPE optimization with O(1) in-place modifications."""
    # Load and preprocess data
    vocab_index_to_token = build_vocab(special_tokens)
    token_sequence = load_and_preprocess_data(input_path, special_tokens, vocab_index_to_token)

    # Convert to doubly linked list
    dll = DoublyLinkedList(token_sequence)

    # Initialize structures
    vocab = vocab_index_to_token
    merges: dict[tuple[int, int], int] = {}
    special_token_ids = set([i for i, _ in enumerate(special_tokens)])

    # Initialize pair index and frequencies
    pair_index, pair_frequencies = initialize_pair_index_and_counts(dll, special_token_ids)
    max_heap = initialize_heap_structures(pair_frequencies, vocab)

    iterations = vocab_size - len(vocab)
    for i in range(iterations + 1):
        # Get most frequent pair using heap - O(log n)
        index1, index2 = get_most_frequent_pair_heap(max_heap, pair_frequencies)

        # Skip if pair involves special tokens
        if index1 in special_token_ids or index2 in special_token_ids:
            # Continue to next iteration without incrementing merge_count
            # but we need to handle this properly to avoid infinite loop
            # Actually, let's filter out special tokens from the heap earlier
            continue

        # Perform incremental merge using linked list - O(k) where k = number of pair occurrences
        new_index = 256 + i
        merges[(index1, index2)] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]

        # Merge in-place and update indices incrementally
        incremental_merge_linked_list(
            max_heap, pair_frequencies, pair_index, dll, index1, index2, new_index, special_token_ids, vocab
        )

    return vocab, merges
