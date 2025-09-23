import heapq
from collections import defaultdict
from typing import Optional


class ListNode:
    """Node for doubly linked list representation of token sequence."""

    def __init__(self, token: int):
        self.token = token
        self.prev: Optional["ListNode"] = None
        self.next: Optional["ListNode"] = None

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
        self.head: Optional[ListNode] = None
        self.tail: Optional[ListNode] = None
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
    dll: DoublyLinkedList,
) -> tuple[dict[tuple[int, int], set[ListNode]], dict[tuple[int, int], int]]:
    """Initialize pair index (node pointers) and frequency counts from linked list."""
    pair_index = defaultdict(set)
    pair_counts = defaultdict(int)
    current = dll.head
    while current and current.next:
        pair = (current.token, current.next.token)
        pair_index[pair].add(current)
        pair_counts[pair] += 1
        current = current.next
    return dict(pair_index), dict(pair_counts)


def initialize_heap_structures(pair_frequencies: dict) -> list[tuple[int, tuple[int, int]]]:
    """Creates a max-heap from pair frequencies for efficient retrieval."""
    # Python's heapq is a min-heap, so we store negative frequencies
    # to simulate a max-heap.
    max_heap = [(-freq, (-pair[0], -pair[1])) for pair, freq in pair_frequencies.items()]
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
        neg_freq, neg_pair = heapq.heappop(max_heap)
        pair = (-neg_pair[0], -neg_pair[1])
        # Check if this heap entry is stale. If the frequency in our authoritative
        # map does not match, it means we have already updated this pair.
        # We discard it and check the next one.
        if pair_frequencies.get(pair) == -neg_freq:
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
) -> None:
    """
    Merges all instances of a pair in the linked list using O(1) operations.
    Updates pair index and frequencies incrementally.

    Assumes heap entries have the form: (-freq, (-pair[0], -pair[1]))
    (i.e. negative frequency, and a negated pair tuple as tie-breaker).
    """
    pair_to_merge = (idx1, idx2)

    # Get all nodes where this pair starts (using our index)
    if pair_to_merge not in pair_index:
        return

    nodes_to_process = list(pair_index[pair_to_merge])  # Copy to avoid modification during iteration

    for first_node in nodes_to_process:
        # Verify this is still a valid pair (may have been modified by previous merges)
        if not first_node.next or first_node.token != idx1 or first_node.next.token != idx2:
            continue

        second_node = first_node.next

        # 1. Update pair counts for broken pairs
        # Left pair (if exists)
        if first_node.prev:
            left_pair = (first_node.prev.token, idx1)
            if left_pair in pair_frequencies:
                # decrement frequency and update index
                pair_frequencies[left_pair] -= 1
                pair_index.get(left_pair, set()).discard(first_node.prev)

                if pair_frequencies[left_pair] > 0:
                    # push with same heap shape: (-freq, (-a, -b))
                    heapq.heappush(max_heap, (-pair_frequencies[left_pair], (-left_pair[0], -left_pair[1])))
                else:
                    # remove completely
                    del pair_frequencies[left_pair]
                    pair_index.pop(left_pair, None)

        # Right pair (if exists)
        if second_node.next:
            right_pair = (idx2, second_node.next.token)
            if right_pair in pair_frequencies:
                pair_frequencies[right_pair] -= 1
                pair_index.get(right_pair, set()).discard(first_node)

                if pair_frequencies[right_pair] > 0:
                    heapq.heappush(max_heap, (-pair_frequencies[right_pair], (-right_pair[0], -right_pair[1])))
                else:
                    del pair_frequencies[right_pair]
                    pair_index.pop(right_pair, None)

        # 2. Perform the merge: replace first_node token, remove second_node
        first_node.token = new_idx
        dll.remove_node(second_node)

        # 3. Add new pairs formed
        # New left pair (if exists)
        if first_node.prev:
            new_left_pair = (first_node.prev.token, new_idx)
            if new_left_pair not in pair_frequencies:
                pair_frequencies[new_left_pair] = 0
                pair_index[new_left_pair] = set()
            pair_frequencies[new_left_pair] += 1
            pair_index[new_left_pair].add(first_node.prev)
            heapq.heappush(max_heap, (-pair_frequencies[new_left_pair], (-new_left_pair[0], -new_left_pair[1])))

        # New right pair (if exists)
        if first_node.next:
            new_right_pair = (new_idx, first_node.next.token)
            if new_right_pair not in pair_frequencies:
                pair_frequencies[new_right_pair] = 0
                pair_index[new_right_pair] = set()
            pair_frequencies[new_right_pair] += 1
            pair_index[new_right_pair].add(first_node)
            heapq.heappush(max_heap, (-pair_frequencies[new_right_pair], (-new_right_pair[0], -new_right_pair[1])))

    # 4. Remove the merged pair from tracking
    pair_frequencies.pop(pair_to_merge, None)
    pair_index.pop(pair_to_merge, None)


# def incremental_merge_linked_list(
#     max_heap: list,
#     pair_frequencies: dict[tuple[int, int], int],
#     pair_index: dict[tuple[int, int], set[ListNode]],
#     dll: DoublyLinkedList,
#     idx1: int,
#     idx2: int,
#     new_idx: int,
# ) -> None:
#     """
#     Merges all instances of a pair in the linked list using O(1) operations.
#     Updates pair index and frequencies incrementally.
#     """
#     pair_to_merge = (idx1, idx2)
#
#     # Get all nodes where this pair starts (using our index)
#     if pair_to_merge not in pair_index:
#         return
#
#     nodes_to_process = list(pair_index[pair_to_merge])  # Copy to avoid modification during iteration
#
#     for first_node in nodes_to_process:
#         # Verify this is still a valid pair (may have been modified by previous merges)
#         if not first_node.next or first_node.token != idx1 or first_node.next.token != idx2:
#             continue
#
#         second_node = first_node.next
#
#         # 1. Update pair counts for broken pairs
#         # Left pair (if exists)
#         if first_node.prev:
#             left_pair = (first_node.prev.token, idx1)
#             if left_pair in pair_frequencies:
#                 pair_frequencies[left_pair] -= 1
#                 pair_index[left_pair].discard(first_node.prev)
#                 if pair_frequencies[left_pair] > 0:
#                     heapq.heappush(max_heap, (-pair_frequencies[left_pair], left_pair))
#                 else:
#                     del pair_frequencies[left_pair]
#                     if left_pair in pair_index:
#                         del pair_index[left_pair]
#
#         # Right pair (if exists)
#         if second_node.next:
#             right_pair = (idx2, second_node.next.token)
#             if right_pair in pair_frequencies:
#                 pair_frequencies[right_pair] -= 1
#                 pair_index[right_pair].discard(first_node)
#                 if pair_frequencies[right_pair] > 0:
#                     heapq.heappush(max_heap, (-pair_frequencies[right_pair], right_pair))
#                 else:
#                     del pair_frequencies[right_pair]
#                     if right_pair in pair_index:
#                         del pair_index[right_pair]
#
#         # 2. Perform the merge: replace first_node token, remove second_node
#         first_node.token = new_idx
#         dll.remove_node(second_node)
#
#         # 3. Add new pairs formed
#         # New left pair (if exists)
#         if first_node.prev:
#             new_left_pair = (first_node.prev.token, new_idx)
#             if new_left_pair not in pair_frequencies:
#                 pair_frequencies[new_left_pair] = 0
#                 pair_index[new_left_pair] = set()
#             pair_frequencies[new_left_pair] += 1
#             pair_index[new_left_pair].add(first_node.prev)
#             heapq.heappush(max_heap, (-pair_frequencies[new_left_pair], new_left_pair))
#
#         # New right pair (if exists)
#         if first_node.next:
#             new_right_pair = (new_idx, first_node.next.token)
#             if new_right_pair not in pair_frequencies:
#                 pair_frequencies[new_right_pair] = 0
#                 pair_index[new_right_pair] = set()
#             pair_frequencies[new_right_pair] += 1
#             pair_index[new_right_pair].add(first_node)
#             heapq.heappush(max_heap, (-pair_frequencies[new_right_pair], new_right_pair))
#
#     # 4. Remove the merged pair from tracking
#     if pair_to_merge in pair_frequencies:
#         del pair_frequencies[pair_to_merge]
#     if pair_to_merge in pair_index:
#         del pair_index[pair_to_merge]


def train_bpe_linked_list_optimized(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Linked list BPE optimization with O(1) in-place modifications."""
    # Load and preprocess data
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
    # token_sequence = load_and_preprocess_data(input_path, special_tokens)

    token_sequence = load_and_preprocess_data(input_path, special_tokens=[])
    # Convert to doubly linked list
    dll = DoublyLinkedList(token_sequence)

    # Initialize structures
    vocab = build_vocab(special_tokens)
    merges: dict[tuple[int, int], int] = {}

    # Initialize pair index and frequencies
    pair_index, pair_frequencies = initialize_pair_index_and_counts(dll)
    max_heap = initialize_heap_structures(pair_frequencies)
    # print(f"pair_index:{pair_index}")
    print(f"pair_frequencies:{pair_frequencies}")
    print(f"max_heap:{max_heap}")
    iterations = vocab_size - len(vocab)
    for i in range(iterations):
        try:
            # Get most frequent pair using heap - O(log n)
            index1, index2 = get_most_frequent_pair_heap(max_heap, pair_frequencies)
            print(f"iter:{i}, pair: ({index1},{index2})")

            # Perform incremental merge using linked list - O(k) where k = number of pair occurrences
            new_index = 256 + i
            merges[(index1, index2)] = new_index
            vocab[new_index] = vocab[index1] + vocab[index2]

            # Merge in-place and update indices incrementally
            incremental_merge_linked_list(max_heap, pair_frequencies, pair_index, dll, index1, index2, new_index)

        except ValueError as e:
            print(f"Stopping training early at iteration {i}: {e}")
            break

    # bytes_merges = [(vocab[x], vocab[y]) for x, y in list(merges.keys())]
    return vocab, merges
