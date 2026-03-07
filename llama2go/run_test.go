package llama2go

import (
	"strings"
	"testing"
)

func TestGeneration(t *testing.T) {
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
			seed: 1,
			topp: 1,
			want: `
Once upon a time, there was a little boy named Timmy. Timmy loved to play outside and collect rocks. He had a big collection of shiny rocks that he wouldist with his own luck.
One day, Timmy's mom asked him what he was doing. Timmy said, "I'm looking at my rocks. I have a useful collection." His mom smiled and asked, "What kind of box you'll look if you straight off?"
Timmy thought for a moment and said, "I want to look at my pocket where my mom has a coin." Timmy's mom helped him tie his shoe and they went to the store. Timmy found the perfect rock to put in his pocket and Tiny to keep him and his mom's trust.
			`,
		},
		{
			temp: 1,
			seed: 1,
			topp: 0.5,
			want: `
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big yellow flower in the garden. She wanted to pick it, but it was too high for her to reach.
Lily asked her mommy for help. "Mommy, can you help me pick the yellow flower?" she said. "Sure, sweetie," her mommy replied. She lifted Lily up so she could pick the flower.
Lily was so happy to have the yellow flower. She showed it to her mommy and they both smiled. "Look, mommy! I picked the flower all by myself!" Lily said. "That's great, Lily. You did a great job," her mommy said.
From that day on, Lily loved to pick flowers in the garden. She always asked her mommy to help her pick them. And every time she saw a yellow flower, she smiled and felt happy.
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

	checkpointPath := "../stories15M.bin"
	tokenizerPath := "../tokenizer.bin"

	transformer := NewTransformer(checkpointPath)
	tokenizer := NewTokenizer(tokenizerPath, transformer.VocabSize())

	for _, test := range tests {
		topp := float32(0.9)
		if test.topp != 0 {
			topp = test.topp
		}

		steps := 256
		if test.steps != 0 {
			steps = test.steps
		}

		sampler := NewSampler(transformer.VocabSize(), test.temp, topp, test.seed)

		got := GenerateText(transformer, tokenizer, sampler, steps)

		if want := strings.TrimSpace(test.want); got != want {
			t.Errorf("GenerateText(temp=%v, topp=%v, seed=%v, steps=%v):\nwant\n\n%v\n\ngot\n\n%v\n\n",
				test.temp, topp, test.seed, steps, want, got)
		}
	}
}
