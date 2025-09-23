from tokenizer import BPETokenizer
from utils import save_merges, save_vocab


def main():
    """Performance comparison between different BPE implementations."""
    import cProfile
    import time

    # input_file = "./data/fake_file.txt"
    input_file = "./tests/fixtures/corpus.en"
    # input_file = "./tests/fixtures/german.txt"
    # input_file = "./tests/fixtures/tinystories_sample.txt"
    # input_file = "./data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    # special_tokens = []
    profiler_naive = cProfile.Profile()
    start_time = time.time()
    profiler_naive.enable()
    bpe_tokenizer = BPETokenizer.train(input_file, vocab_size, special_tokens)
    vocab_naive, merges_naive = bpe_tokenizer.vocab, bpe_tokenizer.merges
    profiler_naive.disable()
    naive_time = time.time() - start_time
    print(f"Naive implementation time: {naive_time:.3f} seconds")

    bpe_tokenizer_naive = BPETokenizer(vocab_naive, merges_naive, special_tokens)
    string = "the quick fox jump over grumpy dog <|endoftext|> hello again"
    tokens = bpe_tokenizer_naive.encode(string)
    print(tokens)
    print(bpe_tokenizer_naive.decode(tokens))

    save_merges("./dump/corpus_merges_naive.txt", merges_naive, vocab_naive)
    save_vocab("./dump/corpus_vocab_naive.json", vocab_naive)


if __name__ == "__main__":
    main()
