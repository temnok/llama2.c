package llama2go

import (
	"strings"
	"testing"
)

func TestGeneration(t *testing.T) {
	checkpoint_path := "../stories15M.bin"
	tokenizer_path := "../tokenizer.bin"

	var transformer Transformer
	build_transformer(&transformer, checkpoint_path)

	var tokenizer Tokenizer
	build_tokenizer(&tokenizer, tokenizer_path, int(transformer.config.vocab_size))

	tests := []struct {
		temp, topp float32
		steps      int
		seed       uint64
		want       string
	}{
		{
			temp: 0,
			seed: 1,
			want: `
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.
Lily found a stick and tried to hit the ball. But the stick was too short. She tried again and again, but she couldn't reach it. She felt sad.
Suddenly, a kind man came by and saw Lily. He asked her what was wrong. Lily told him about the ball. The man smiled and said, "I have a useful idea!" He took out a long stick and used it to knock the ball down. Lily was so happy! She thanked the man and they played together in the sunshine.
			`,
		},
		{
			temp: 1,
			seed: 1,
			want: `
Once upon a time, there was a little boy named Timmy. Timmy loved to play outside and explore. One day, he saw a big pile of coal in his backyard. He was curious and went to take a closer look.
As he reached for the coal, he noticed that it was dirty and covered in all the black and white coal. Timmy felt bad because he knew that coal could be dangerous. He decided to make a fire to stay warm and warm.
Timmy lit a match and lit the coal. He watched it burn until it cooled down and he got to keep one. He then rolled the coal on his hand and pretended it was his imagination. He felt happy and warm inside. From that day on, Timmy always made sure to keep an eye out for any trash and coal in the yard.
			`,
		},
		{
			temp: 1,
			seed: 2,
			want: `
Once upon a time, there was a big house with a chimney. A little birdie lived in the chimney. The birdie was very happy and sang songs all day long. One day, a big storm came and the birdie's nest fell down. The birdie was sad and scared.
A kind girl saw the birdie and wanted to help. She tried to put the birdie back into the chimney, but the birdie was too scared to fly up there. The girl had an idea. She got a big box and put the birdie inside. Then, she carried the birdie to a safe place.
The birdie was happy and started to sing again. The girl said, "I'm sorry I scared you, but I have to finish this project so you can sleep." The birdie was grateful and promised to never knock down another birdie again.
			`,
		},
	}

	for _, test := range tests {
		topp := float32(0.9)
		if test.topp != 0 {
			topp = test.topp
		}

		steps := 256
		if test.steps != 0 {
			steps = test.steps
		}

		var sampler Sampler
		build_sampler(&sampler, int(transformer.config.vocab_size), test.temp, topp, test.seed)
		got := generate_text(&transformer, &tokenizer, &sampler, steps)

		if want := strings.TrimSpace(test.want); got != want {
			t.Fatalf("generate_text(temp=%v, topp=%v, seed=%v, steps=%v):\nwant\n\n%v\n\ngot\n\n%v\n\n",
				test.temp, topp, test.seed, steps, want, got)
		}
	}
}

func xTestPrint(t *testing.T) {
	checkpoint_path := "../stories15M.bin"
	tokenizer_path := "../tokenizer.bin"
	temperature := float32(0) // 0.0 = greedy deterministic. 1.0 = original. don't set higher
	//temperature := float32(1) // 0.0 = greedy deterministic. 1.0 = original. don't set higher
	topp := float32(0.9)  // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
	steps := 256          // number of steps to run for
	rng_seed := uint64(0) // seed rng with time by default

	// build the Transformer via the model .bin file
	var transformer Transformer
	build_transformer(&transformer, checkpoint_path)

	// build the Tokenizer via the tokenizer .bin file
	var tokenizer Tokenizer
	build_tokenizer(&tokenizer, tokenizer_path, int(transformer.config.vocab_size))

	// build the Sampler
	var sampler Sampler
	build_sampler(&sampler, int(transformer.config.vocab_size), temperature, topp, rng_seed)

	generate_and_print(&transformer, &tokenizer, &sampler, steps)
}
