package main

import (
	"fmt"
	"github.com/temnok/llama2go"
	"os"
	"time"
)

func main() {
	checkpoint_path := "../stories15M.bin"
	tokenizer_path := "../tokenizer.bin"
	temperature := float32(0) // 0.0 = greedy deterministic. 1.0 = original. don't set higher
	//temperature := float32(1) // 0.0 = greedy deterministic. 1.0 = original. don't set higher
	topp := float32(0.9)  // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
	steps := 256          // number of steps to run for
	rng_seed := uint64(0) // seed rng with time by default

	// build the Transformer via the model .bin file
	transformer := llama2go.NewTransformer(checkpoint_path)

	// build the Tokenizer via the tokenizer .bin file
	tokenizer := llama2go.NewTokenizer(tokenizer_path, transformer.VocabSize())

	// build the Sampler
	sampler := llama2go.NewSampler(transformer.VocabSize(), temperature, topp, rng_seed)

	start := int(time.Now().UnixMilli())

	token_count := 0
	llama2go.Generate(transformer, tokenizer, sampler, steps, func(piece string) {
		fmt.Print(piece)
		token_count++
	})

	if token_count > 0 {
		tps := (token_count * 1000 * 1000) / (int(time.Now().UnixMilli()) - start)
		fmt.Fprintf(os.Stderr, "\n\nachieved tok/s: %v.%03v\n", tps/1000, tps%1000)
	}
}
