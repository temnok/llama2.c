package main

import (
	"fmt"
	"github.com/temnok/llama2go"
	"time"
)

func main() {
	// {dim:288 hiddenDim:768 nLayers:6 nHeads:6 nKvHeads:6 vocabSize:32000 seqLen:256}
	checkpointPath := "../stories15M.bin"

	tokenizerPath := "../tokenizer.bin"
	temperature := float32(0) // 0.0 = greedy deterministic. 1.0 = original. don't set higher
	//temperature := float32(1) // 0.0 = greedy deterministic. 1.0 = original. don't set higher
	topP := float32(0.9) // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
	steps := 256         // number of steps to run for
	rngSeed := uint64(0) // seed rng with time by default

	// build the Transformer via the model .bin file
	transformer := llama2go.NewTransformer(checkpointPath)

	// build the Tokenizer via the tokenizer .bin file
	tokenizer := llama2go.NewTokenizer(tokenizerPath, transformer.VocabSize())

	// build the Sampler
	sampler := llama2go.NewSampler(transformer.VocabSize(), temperature, topP, rngSeed)

	start := int(time.Now().UnixMilli())

	tokenCount := 0
	llama2go.Generate(transformer, tokenizer, sampler, steps, func(piece string) {
		fmt.Print(piece)
		tokenCount++
	})

	if tokenCount > 0 {
		tps := (tokenCount * 1000 * 1000) / (int(time.Now().UnixMilli()) - start)
		fmt.Printf("\n\nachieved tok/s: %v.%03v\n", tps/1000, tps%1000)
	}
}
