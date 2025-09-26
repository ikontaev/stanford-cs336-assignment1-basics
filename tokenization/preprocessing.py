import os
from typing import BinaryIO

import regex as re


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
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


def pretokenize(chunk: str, special_tokens: list[str]) -> list[str | bytes]:
    """
    Preprocess the raw chunk of text.
    - Splits chunk on texts by special tokens (if provided).
    - Splits the text into subword units using regex.
    - Returns a list of strings (special tokens) and bytes (subword units).
    """
    if not special_tokens:
        texts = [chunk]
    else:
        # Sort special tokens by length (longest first) to handle overlapping tokens correctly
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        delimiter = "|".join(map(re.escape, sorted_special_tokens))
        texts = re.split(rf"({delimiter})", chunk)  # Keep delimiters

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    chunk_tokens = []
    for text in texts:
        if text in special_tokens:
            chunk_tokens.append(text)  # Keep special tokens as strings
        else:
            # Split into subword units and encode as bytes
            tokens = re.finditer(PAT, text, re.UNICODE)
            for token in tokens:
                token_str = token.group()
                if token_str:  # Skip empty strings
                    chunk_tokens.append(token_str.encode("utf-8"))
    return chunk_tokens


def tokenize(pretokens: list[str | bytes], vocab: dict[bytes, int], special_token_to_id: dict[str, int]) -> list[int]:
    ids = []
    for token in pretokens:
        if isinstance(token, str):
            ids.append(special_token_to_id[token])
        else:
            ids.extend([vocab[bytes([b])] for b in token])
    return ids


def load_and_preprocess_data(
    input_path: str | os.PathLike, special_tokens: list[str], vocab: dict[int, bytes]
) -> list[int]:
    """Load data from file and preprocess into token indices."""
    vocab_token_to_index = {v: k for k, v in vocab.items()}
    special_token_to_index = {token: i for i, token in enumerate(special_tokens)}
    with open(input_path, "rb") as f:
        num_processes = 4
        end_of_text = "<|endoftext|>"
        boundaries = find_chunk_boundaries(f, num_processes, end_of_text.encode("utf-8"))

        indecis = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_tokens = pretokenize(chunk, special_tokens)
            chunk_indecis = tokenize(chunk_tokens, vocab_token_to_index, special_token_to_index)
            indecis.extend(chunk_indecis)

    return indecis
