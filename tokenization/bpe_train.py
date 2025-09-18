import os
from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex as re


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


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(chunk: str, special_token: str) -> list[int]:
    """Preprocess the raw chunk of text and map each token to UTF-8 byte representation."""
    texts = re.split(rf"\s*{re.escape(special_token)}\s*", chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    chunk_indecis = []
    for text in texts:
        tokens = re.finditer(PAT, text)
        for token in tokens:
            token_bytes = list(map(int, token.group().encode("utf-8")))
            chunk_indecis.extend(token_bytes)
    return chunk_indecis


def build_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    for x in range(1, 256):
        vocab[x] = bytes([x])
    return vocab


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

    with open(input_path, "rb") as f:
        num_processes = 4
        end_of_text = "<|endoftext|>"
        boundaries = find_chunk_boundaries(f, num_processes, end_of_text.encode("utf-8"))
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        indecis = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_indecis = pretokenize(chunk, special_token=end_of_text)
            indecis.extend(chunk_indecis)

        merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged_index
        vocab = build_vocab(special_tokens)
        iterations = vocab_size - len(vocab)
        for i in range(iterations):
            index1, index2 = find_most_frequent_pair(indecis, vocab)
            # merge
            new_index = 256 + i
            merges[(index1, index2)] = new_index
            indecis = merge(indecis, (index1, index2), new_index)
            vocab[new_index] = vocab[index1] + vocab[index2]
        bytes_merges = [(vocab[x], vocab[y]) for x, y in list(merges.keys())]
        return vocab, bytes_merges


# train_bpe("./data/TinyStoriesV2-GPT4-valid.txt", 500, ["<|endoftext|>"])
print(train_bpe("./tests/fixtures/tinystories_sample.txt", 500, ["<|endoftextg|>"]))
