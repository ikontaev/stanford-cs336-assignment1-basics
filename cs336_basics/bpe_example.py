from collections import defaultdict
from dataclasses import dataclass
from io import StringIO

fakefile = StringIO(
    """
    low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest
    """
)


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""

    vocab: dict[int, bytes]  # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index


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


def decode(indecis, vocab):
    return b" ".join([vocab[index] for index in indecis]).decode("utf-8")


def naive_bpe(f: StringIO, vocab_size: int):
    indecis = []
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged_index

    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(1, 256)}
    vocab[0] = b"<|endoftext|>"

    for line in f:
        for word in line.split():
            word_bytes = list(map(int, word.encode("utf-8")))
            indecis.extend(word_bytes)

    iterations = vocab_size - len(vocab)
    for i in range(iterations):
        index1, index2 = find_most_frequent_pair(indecis, vocab)
        # merge
        new_index = 256 + i
        merges[(index1, index2)] = new_index
        indecis = merge(indecis, (index1, index2), new_index)
        vocab[new_index] = vocab[index1] + vocab[index2]

    return BPETokenizerParams(vocab, merges)


print(naive_bpe(fakefile, 262))
