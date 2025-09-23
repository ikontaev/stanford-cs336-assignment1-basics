import json


def bytes_to_str(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.hex()


def save_vocab(path: str, vocab: dict[int, bytes]):
    vocab_inv = {}
    for k, v in vocab.items():
        key_str = bytes_to_str(v)
        vocab_inv[key_str] = k

    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab_inv, f, indent=4, ensure_ascii=False)


def save_merges(path: str, merges: dict[tuple[int, int], int], vocab: dict[int, bytes]):
    with open(path, "w") as f:
        for i, (token1, token2) in enumerate(merges):
            f.write(f"{i}: {bytes_to_str(vocab[token1])} {bytes_to_str(vocab[token2])}\n")
