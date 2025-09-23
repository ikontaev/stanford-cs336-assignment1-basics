from tokenizer import BPETokenizer
from tokenizer_optimized import BPETokenizerOptimized
from utils import save_merges, save_vocab


def compare_tokenizers():
    """Performance and correctness comparison between naive and optimized BPE implementations."""
    import cProfile
    import time

    # Test parameters
    input_file = "./tests/fixtures/corpus.en"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    test_string = "the quick fox jump over grumpy dog <|endoftext|> hello again"

    print("=" * 60)
    print("BPE Tokenizer Comparison")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print()

    # Train naive implementation
    print("Training naive BPE tokenizer...")
    profiler_naive = cProfile.Profile()
    start_time = time.time()
    profiler_naive.enable()
    bpe_tokenizer_naive = BPETokenizer.train(input_file, vocab_size, special_tokens)
    vocab_naive, merges_naive = bpe_tokenizer_naive.vocab, bpe_tokenizer_naive.merges
    profiler_naive.disable()
    naive_time = time.time() - start_time
    print(f"Naive implementation time: {naive_time:.3f} seconds")

    # Train optimized implementation
    print("\nTraining optimized optimized BPE tokenizer...")
    profiler_optimized = cProfile.Profile()
    start_time = time.time()
    profiler_optimized.enable()
    bpe_tokenizer_optimized = BPETokenizerOptimized.train(input_file, vocab_size, special_tokens)
    vocab_optimized, merges_optimized = bpe_tokenizer_optimized.vocab, bpe_tokenizer_optimized.merges
    profiler_optimized.disable()
    optimized_time = time.time() - start_time
    print(f"Simple optimized implementation time: {optimized_time:.3f} seconds")

    # Performance comparison
    print("Performance improvements:")
    print(f"  Optimized: {naive_time / optimized_time:.2f}x faster than naive")

    # Test encoding/decoding functionality
    print("\n" + "=" * 60)
    print("Functionality Test")
    print("=" * 60)
    print(f"Test string: '{test_string}'")

    # Test naive tokenizer
    tokens_naive = bpe_tokenizer_naive.encode(test_string)
    decoded_naive = bpe_tokenizer_naive.decode(tokens_naive)
    print("Naive tokenizer:")
    print(f"  Tokens: {tokens_naive}")
    print(f"  Decoded: '{decoded_naive}'")

    # Test optimized tokenizer
    tokens_optimized = bpe_tokenizer_optimized.encode(test_string)
    decoded_optimized = bpe_tokenizer_optimized.decode(tokens_optimized)
    print("Optimized tokenizer:")
    print(f"  Tokens: {tokens_optimized}")
    print(f"  Decoded: '{decoded_optimized}'")

    # Compare vocabularies and merges
    print("\n" + "=" * 60)
    print("Vocabulary and Merges Comparison")
    print("=" * 60)

    # Compare naive vs optimized optimized
    print("Naive vs Optimized:")
    vocab_match_optimized = vocab_naive == vocab_optimized
    merges_match_optimized = merges_naive == merges_optimized
    tokens_match_optimized = tokens_naive == tokens_optimized
    decode_match_optimized = decoded_naive == decoded_optimized

    print(f"  Vocabularies match: {vocab_match_optimized}")
    print(f"  Merges match: {merges_match_optimized}")
    print(f"  Tokenization results match: {tokens_match_optimized}")
    print(f"  Decoded strings match: {decode_match_optimized}")

    # Save outputs for inspection
    print("\nSaving outputs...")
    save_merges("./dump/corpus_merges_naive.txt", merges_naive, vocab_naive)
    save_vocab("./dump/corpus_vocab_naive.json", vocab_naive)
    save_merges("./dump/corpus_merges_optimized.txt", merges_optimized, vocab_optimized)
    save_vocab("./dump/corpus_vocab_optimized.json", vocab_optimized)


def main():
    """Main function to run the comparison."""
    compare_tokenizers()


if __name__ == "__main__":
    main()
