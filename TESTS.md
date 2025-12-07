# Testing Guide: The Lament Engine

## Unit Tests

### Tokenizer Tests
- **Round-trip encoding/decoding**: Verify that encoding then decoding returns original text
- **Special token handling**: Test that mode tokens are correctly identified
- **Edge cases**: Empty strings, very long strings, special characters

### Matrix Operations Tests
- **Matrix multiplication**: Verify correctness against known results
- **Vector operations**: Test addition, scaling, dot product
- **Numerical stability**: Test with extreme values

### Model Tests
- **Forward pass**: Verify output shape and range
- **Mode conditioning**: Test that different modes produce different outputs
- **KV cache**: Verify cache is updated correctly during generation

## Integration Tests

### End-to-End Generation
1. Load model and tokenizer
2. Generate text in each mode
3. Verify mode-specific behavior
4. Test self-termination in silence mode

### Safety Tests
1. Test content filtering with blocked phrases
2. Verify rate limiting works
3. Test content warnings trigger appropriately

## Evaluation Metrics

### Perplexity
- Measure on held-out test set
- Compare across modes
- Track during training

### Human Evaluation
- Style accuracy: Does output match requested mode?
- Safety: Are harmful outputs blocked?
- Quality: Is generated text coherent?

## Benchmarking

### Performance Metrics
- **Tokens per second**: Measure generation speed
- **Memory usage**: Track RAM consumption
- **Latency**: Time to first token, time per token

### Test Commands

```bash
# Run tokenizer tests
./tests/test_tokenizer.sh

# Run model tests
./tests/test_model.sh

# Run safety tests
./tests/test_safety.sh

# Benchmark
./bench/benchmark.sh
```

## Test Data

Test data should be:
- Representative of actual use cases
- Include edge cases
- Cover all modes
- Include safety test cases

## Continuous Testing

Tests should be run:
- Before each commit
- In CI/CD pipeline
- After model updates
- After code changes

