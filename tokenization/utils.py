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


def save_merges(path: str, merges: list[tuple[bytes, bytes]]):
    with open(path, "w") as f:
        for token1, token2 in merges:
            f.write(f"{bytes_to_str(token1)} {bytes_to_str(token2)}\n")
