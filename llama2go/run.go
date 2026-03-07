package llama2go

import (
	"encoding/binary"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
)

// ----------------------------------------------------------------------------
// Transformer model

type config struct {
	dim        int32 // transformer dimension
	hidden_dim int32 // for ffn layers
	n_layers   int32 // number of layers
	n_heads    int32 // number of query heads
	n_kv_heads int32 // number of key/value heads (can be < query heads because of multiquery)
	vocab_size int32 // vocabulary size, usually 256 (byte-level)
	seq_len    int32 // max sequence length
}

type transformerWeights struct {
	// token embedding table
	token_embedding_table []float32 // (vocab_size, dim)
	// weights for rmsnorms
	rms_att_weight []float32 // (layer, dim) rmsnorm weights
	rms_ffn_weight []float32 // (layer, dim)
	// weights for matmuls. note dim == n_heads * head_size
	wq []float32 // (layer, dim, n_heads * head_size)
	wk []float32 // (layer, dim, n_kv_heads * head_size)
	wv []float32 // (layer, dim, n_kv_heads * head_size)
	wo []float32 // (layer, n_heads * head_size, dim)
	// weights for ffn
	w1 []float32 // (layer, hidden_dim, dim)
	w2 []float32 // (layer, dim, hidden_dim)
	w3 []float32 // (layer, hidden_dim, dim)
	// final rmsnorm
	rms_final_weight []float32 // (dim,)
	// (optional) classifier weights for the logits, on the last layer
	wcls []float32
}

type runState struct {
	// current wave of activations
	x      []float32 // activation at current time stamp (dim,)
	xb     []float32 // same, but inside a residual branch (dim,)
	xb2    []float32 // an additional buffer just for convenience (dim,)
	hb     []float32 // buffer for hidden dimension in the ffn (hidden_dim,)
	hb2    []float32 // buffer for hidden dimension in the ffn (hidden_dim,)
	q      []float32 // query (dim,)
	k      []float32 // key (dim,)
	v      []float32 // value (dim,)
	att    []float32 // buffer for scores/attention values (n_heads, seq_len)
	logits []float32 // output logits
	// kv cache
	key_cache   []float32 // (layer, seq_len, dim)
	value_cache []float32 // (layer, seq_len, dim)
}

type Transformer struct {
	config  config             // the hyperparameters of the architecture (the blueprint)
	weights transformerWeights // the weights of the model
	state   runState           // buffers for the "wave" of activations in the forward pass
}

func malloc_run_state(s *runState, p *config) {
	kv_dim := (p.dim * p.n_kv_heads) / p.n_heads
	s.x = make([]float32, p.dim)
	s.xb = make([]float32, p.dim)
	s.xb2 = make([]float32, p.dim)
	s.hb = make([]float32, p.hidden_dim)
	s.hb2 = make([]float32, p.hidden_dim)
	s.q = make([]float32, p.dim)
	s.key_cache = make([]float32, p.n_layers*p.seq_len*kv_dim)
	s.value_cache = make([]float32, p.n_layers*p.seq_len*kv_dim)
	s.att = make([]float32, p.n_heads*p.seq_len)
	s.logits = make([]float32, p.vocab_size)
}

func memory_map_weights(w *transformerWeights, p *config, ptr []float32, shared_weights bool) {
	head_size := p.dim / p.n_heads
	n_layers := p.n_layers
	w.token_embedding_table = ptr
	ptr = ptr[p.vocab_size*p.dim:]
	w.rms_att_weight = ptr
	ptr = ptr[n_layers*p.dim:]
	w.wq = ptr
	ptr = ptr[n_layers*p.dim*(p.n_heads*head_size):]
	w.wk = ptr
	ptr = ptr[n_layers*p.dim*(p.n_kv_heads*head_size):]
	w.wv = ptr
	ptr = ptr[n_layers*p.dim*(p.n_kv_heads*head_size):]
	w.wo = ptr
	ptr = ptr[n_layers*(p.n_heads*head_size)*p.dim:]
	w.rms_ffn_weight = ptr
	ptr = ptr[n_layers*p.dim:]
	w.w1 = ptr
	ptr = ptr[n_layers*p.dim*p.hidden_dim:]
	w.w2 = ptr
	ptr = ptr[n_layers*p.hidden_dim*p.dim:]
	w.w3 = ptr
	ptr = ptr[n_layers*p.dim*p.hidden_dim:]
	w.rms_final_weight = ptr
	ptr = ptr[p.dim:]
	ptr = ptr[p.seq_len*head_size/2:] // skip what used to be freq_cis_real (for RoPE)
	ptr = ptr[p.seq_len*head_size/2:] // skip what used to be freq_cis_imag (for RoPE)
	w.wcls = ptr
	if shared_weights {
		w.wcls = w.token_embedding_table
	}
}

func read_checkpoint(checkpoint string, cfg *config, weights *transformerWeights) {
	file := check1(os.Open(checkpoint))
	defer checkCall(file.Close)

	// read in the config header
	cw := make([]int32, 7)
	binaryRead(file, cw)
	*cfg = config{cw[0], cw[1], cw[2], cw[3], cw[4], cw[5], cw[6]}

	// negative vocab size is hacky way of signaling unshared weights. bit yikes.
	shared_weights := cfg.vocab_size > 0
	if !shared_weights {
		cfg.vocab_size = -cfg.vocab_size
	}

	// figure out the file size
	stat := check1(file.Stat())
	file_size := stat.Size() // get the file size, in bytes

	// memory map the Transformer weights into the data pointer
	data := make([]float32, file_size/4-int64(len(cw)))
	binaryRead(file, data)

	//for i, val := range data {
	//	val = math.Float32frombits(math.Float32bits(val) &^ 0xFFFFF)
	//	data[i] = val
	//}

	memory_map_weights(weights, cfg, data, shared_weights)
}

func NewTransformer(checkpoint_path string) *Transformer {
	t := &Transformer{}
	// read in the Config and the Weights from the checkpoint
	read_checkpoint(checkpoint_path, &t.config, &t.weights)
	// allocate the RunState buffers
	malloc_run_state(&t.state, &t.config)
	return t
}

func (t *Transformer) VocabSize() int {
	return int(t.config.vocab_size)
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

func rmsnorm(o, x, weight []float32, size int) {
	// calculate sum of squares
	ss := float32(0)
	for j := 0; j < size; j++ {
		ss += x[j] * x[j]
	}
	ss /= float32(size)
	ss += 1e-5
	ss = 1.0 / sqrtf(ss)
	// normalize and scale
	for j := 0; j < size; j++ {
		o[j] = weight[j] * (ss * x[j])
	}
}

func softmax(x []float32, size int) {
	// find max value (for numerical stability)
	max_val := x[0]
	for i := 1; i < size; i++ {
		if x[i] > max_val {
			max_val = x[i]
		}
	}
	// exp and sum
	sum := float32(0)
	for i := 0; i < size; i++ {
		x[i] = expf(x[i] - max_val)
		sum += x[i]
	}
	// normalize
	for i := 0; i < size; i++ {
		x[i] /= sum
	}
}

func matmul(xout, x, w []float32, n, d int) {
	// W (d,n) @ x (n,) -> xout (d,)
	// by far the most amount of time is spent inside this little function
	for i := 0; i < d; i++ {
		val := float32(0)
		for j := 0; j < n; j++ {
			val += w[i*n+j] * x[j]
		}
		xout[i] = val
	}
}

func forward(transformer *Transformer, token int, pos int) []float32 {
	// a few convenience variables
	p := &transformer.config
	w := &transformer.weights
	s := &transformer.state
	x := s.x

	dim := int(p.dim)
	hidden_dim := int(p.hidden_dim)
	n_layers := int(p.n_layers)
	n_heads := int(p.n_heads)
	n_kv_heads := int(p.n_kv_heads)
	vocab_size := int(p.vocab_size)
	seq_len := int(p.seq_len)

	kv_dim := (dim * n_kv_heads) / n_heads
	kv_mul := n_heads / n_kv_heads // integer multiplier of the kv sharing in multiquery
	head_size := dim / n_heads

	// copy the token embedding into x
	content_row := w.token_embedding_table[token*dim:]
	copy(x, content_row[:dim])

	// forward all the layers
	for l := 0; l < n_layers; l++ {

		// attention rmsnorm
		rmsnorm(s.xb, x, w.rms_att_weight[l*dim:], dim)

		// key and value point to the kv cache
		loff := l * seq_len * kv_dim // kv cache layer offset for convenience
		s.k = s.key_cache[loff+pos*kv_dim:]
		s.v = s.value_cache[loff+pos*kv_dim:]

		// qkv matmuls for this position
		matmul(s.q, s.xb, w.wq[l*dim*dim:], dim, dim)
		matmul(s.k, s.xb, w.wk[l*dim*kv_dim:], dim, kv_dim)
		matmul(s.v, s.xb, w.wv[l*dim*kv_dim:], dim, kv_dim)

		// RoPE relative positional encoding: complex-valued rotate q and k in each head
		for i := 0; i < dim; i += 2 {
			head_dim := i % head_size
			freq := 1.0 / powf(10000, float32(head_dim)/float32(head_size))
			val := float32(pos) * freq
			fcr := cosf(val)
			fci := sinf(val)

			// how many vectors? 2 = q & k, 1 = q only
			rotn := 1
			if i < kv_dim {
				rotn = 2
			}

			for v := 0; v < rotn; v++ {
				// the vector to rotate (query or key)
				vec := s.q
				if v != 0 {
					vec = s.k
				}

				v0 := vec[i]
				v1 := vec[i+1]
				vec[i] = v0*fcr - v1*fci
				vec[i+1] = v0*fci + v1*fcr
			}
		}

		for h := 0; h < n_heads; h++ {
			// get the query vector for this head
			q := s.q[h*head_size:]
			// attention scores for this head
			att := s.att[h*seq_len:]
			// iterate over all timesteps, including the current one
			for t := 0; t <= pos; t++ {
				// get the key vector for this head and at this timestep
				k := s.key_cache[loff+t*kv_dim+(h/kv_mul)*head_size:]
				// calculate the attention score as the dot product of q and k
				score := float32(0)
				for i := 0; i < head_size; i++ {
					score += q[i] * k[i]
				}
				score /= sqrtf(float32(head_size))
				// save the score to the attention buffer
				att[t] = score
			}

			// softmax the scores to get attention weights, from 0..pos inclusively
			softmax(att, pos+1)

			// weighted sum of the values, store back into xb
			xb := s.xb[h*head_size:]
			clear(xb[:head_size])
			for t := 0; t <= pos; t++ {
				// get the value vector for this head and at this timestep
				v := s.value_cache[loff+t*kv_dim+(h/kv_mul)*head_size:]
				// get the attention weight for this timestep
				a := att[t]
				// accumulate the weighted value into xb
				for i := 0; i < head_size; i++ {
					xb[i] += a * v[i]
				}
			}

		}

		// final matmul to get the output of the attention
		matmul(s.xb2, s.xb, w.wo[l*dim*dim:], dim, dim)

		// residual connection back into x
		for i := 0; i < dim; i++ {
			x[i] += s.xb2[i]
		}

		// ffn rmsnorm
		rmsnorm(s.xb, x, w.rms_ffn_weight[l*dim:], dim)

		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
		// first calculate self.w1(x) and self.w3(x)
		matmul(s.hb, s.xb, w.w1[l*dim*hidden_dim:], dim, hidden_dim)
		matmul(s.hb2, s.xb, w.w3[l*dim*hidden_dim:], dim, hidden_dim)

		// SwiGLU non-linearity
		for i := 0; i < hidden_dim; i++ {
			val := s.hb[i]
			// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
			val *= 1.0 / (1.0 + expf(-val))
			// elementwise multiply with w3(x)
			val *= s.hb2[i]
			s.hb[i] = val
		}

		// final matmul to get the output of the ffn
		matmul(s.xb, s.hb, w.w2[l*dim*hidden_dim:], hidden_dim, dim)

		// residual connection
		for i := 0; i < dim; i++ {
			x[i] += s.xb[i]
		}
	}

	// final rmsnorm
	rmsnorm(x, x, w.rms_final_weight, dim)

	// classifier into logits
	matmul(s.logits, x, w.wcls, dim, vocab_size)
	return s.logits
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

type TokenIndex struct {
	str string
	id  int
}

type Tokenizer struct {
	vocab        []string
	vocab_scores []float32
	sorted_vocab []TokenIndex
	vocab_size   int
}

func NewTokenizer(tokenizer_path string, vocab_size int) *Tokenizer {
	t := &Tokenizer{
		// i should have written the vocab_size into the tokenizer file... sigh
		vocab_size:   vocab_size,
		vocab:        make([]string, vocab_size),
		vocab_scores: make([]float32, vocab_size),
	}

	// read in the file
	file := check1(os.Open(tokenizer_path))
	defer checkCall(file.Close)

	var max_token_length int32
	binaryRead(file, &max_token_length)

	for i := 0; i < vocab_size; i++ {
		binaryRead(file, &t.vocab_scores[i])

		var strlen int32
		binaryRead(file, &strlen)

		buf := make([]byte, strlen)
		binaryRead(file, buf)
		t.vocab[i] = string(buf)
	}

	return t
}

func decode(t *Tokenizer, prev_token, token int) string {
	piece := t.vocab[token]
	// following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
	if prev_token == 1 && piece[0] == ' ' {
		piece = piece[1:]
	}
	// careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
	// parse this and convert and return the actual byte
	if len(piece) == 6 && strings.HasPrefix(piece, "<0x") && strings.HasSuffix(piece, ">") {
		b := check1(strconv.ParseInt(piece[3:5], 16, 8))
		piece = string(byte(b))
	}
	return piece
}

func is_printable(piece string) bool {
	if piece == "" {
		return false
	}

	// piece might be a raw byte token, and we only want to print printable chars or whitespace
	// because some of the other bytes can be various control codes, backspace, etc.
	if len(piece) == 1 && !unicode.IsPrint(rune(piece[0])) && !unicode.IsSpace(rune(piece[0])) {
		return false // bad byte, don't print it
	}

	return true
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

// struct used when sorting probabilities during top-p sampling
type ProbIndex struct {
	prob  float32
	index int
}

type Sampler struct {
	vocab_size  int
	probindex   []ProbIndex // buffer used in top-p sampling
	temperature float32
	topp        float32
	rng_state   uint64
}

func sample_argmax(probabilities []float32, n int) int {
	// return the index that has the highest probability
	max_i := 0
	max_p := probabilities[0]
	for i := 1; i < n; i++ {
		if probabilities[i] > max_p {
			max_i = i
			max_p = probabilities[i]
		}
	}
	return max_i
}

func sample_mult(probabilities []float32, n int, coin float32) int {
	// sample index from probabilities (they must sum to 1!)
	// coin is a random number in [0, 1), usually from random_f32()
	cdf := float32(0)
	for i := 0; i < n; i++ {
		cdf += probabilities[i]
		if coin < cdf {
			return i
		}
	}
	return n - 1 // in case of rounding errors
}

func sample_topp(probabilities []float32, n int, topp float32, probindex []ProbIndex, coin float32) int {
	// top-p sampling (or "nucleus sampling") samples from the smallest set of
	// tokens that exceed probability topp. This way we never sample tokens that
	// have very low probabilities and are less likely to go "off the rails".
	// coin is a random number in [0, 1), usually from random_f32()

	n0 := 0
	// quicksort indices in descending order of probabilities
	// values smaller than (1 - topp) / (n - 1) cannot be part of the result
	// so for efficiency we crop these out as candidates before sorting
	cutoff := (1.0 - topp) / float32(n-1)
	for i := 0; i < n; i++ {
		if probabilities[i] >= cutoff {
			probindex[n0].index = i
			probindex[n0].prob = probabilities[i]
			n0++
		}
	}

	sort.Slice(probindex[:n0], func(i, j int) bool {
		return probindex[i].prob > probindex[j].prob
	})

	// truncate the list where cumulative probability exceeds topp
	cumulative_prob := float32(0)
	last_idx := n0 - 1 // in case of rounding errors consider all elements
	for i := 0; i < n0; i++ {
		cumulative_prob += probindex[i].prob
		if cumulative_prob > topp {
			last_idx = i
			break // we've exceeded topp by including last_idx
		}
	}

	// sample from the truncated list
	r := coin * cumulative_prob
	cdf := float32(0)
	for i := 0; i <= last_idx; i++ {
		cdf += probindex[i].prob
		if r < cdf {
			return probindex[i].index
		}
	}
	return probindex[last_idx].index // in case of rounding errors
}

func NewSampler(vocab_size int, temperature, topp float32, rng_seed uint64) *Sampler {
	return &Sampler{
		vocab_size:  vocab_size,
		temperature: temperature,
		topp:        topp,
		rng_state:   rng_seed,
		// buffer only used with nucleus sampling; may not need but it's ~small
		probindex: make([]ProbIndex, vocab_size),
	}
}

func random_u32(state *uint64) int {
	// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
	*state ^= *state >> 12
	*state ^= *state << 25
	*state ^= *state >> 27
	return int((*state * 0x2545F4914F6CDD1D) >> 32)
}

func random_f32(state *uint64) float32 { // random float32 in [0,1)
	return float32(random_u32(state)>>8) / 16777216.0
}

func sample(sampler *Sampler, logits []float32) int {
	// sample the token given the logits and some hyperparameters
	var next int
	if sampler.temperature == 0 {
		// greedy argmax sampling: take the token with the highest probability
		next = sample_argmax(logits, sampler.vocab_size)
	} else {
		// apply the temperature to the logits
		for q := 0; q < sampler.vocab_size; q++ {
			logits[q] /= sampler.temperature
		}
		// apply softmax to the logits to get the probabilities for next token
		softmax(logits, sampler.vocab_size)
		// flip a (float) coin (this is our source of entropy for sampling)
		coin := random_f32(&sampler.rng_state)
		// we sample from this distribution to get the next token
		if sampler.topp <= 0 || sampler.topp >= 1 {
			// simply sample from the predicted probability distribution
			next = sample_mult(logits, sampler.vocab_size, coin)
		} else {
			// top-p (nucleus) sampling, clamping the least likely tokens to zero
			next = sample_topp(logits, sampler.vocab_size, sampler.topp, sampler.probindex, coin)
		}
	}
	return next
}

// ----------------------------------------------------------------------------
// generation loop

func Generate(transformer *Transformer, tokenizer *Tokenizer, sampler *Sampler, steps int,
	callback func(string)) {
	steps = min(steps, int(transformer.config.seq_len))

	//encode(tokenizer, prompt, true, false, prompt_tokens, &num_prompt_tokens)
	prompt_tokens := []int{1}
	num_prompt_tokens := 1

	start := 0                // used to time our code, only initialized after first iteration
	next := 0                 // will store the next token in the sequence
	token := prompt_tokens[0] // kick off with the first token in the prompt
	for pos := 0; pos < steps; pos++ {

		// forward the transformer to get logits for the next token
		logits := forward(transformer, token, pos)

		// advance the state machine
		if pos < num_prompt_tokens-1 {
			// if we are still processing the input prompt, force the next prompt token
			next = prompt_tokens[pos+1]
		} else {
			// otherwise sample the next token from the logits
			next = sample(sampler, logits)
		}

		// data-dependent terminating condition: the BOS (=1) token delimits sequences
		if next == 1 {
			break
		}

		// print the token as string, decode it with the Tokenizer object
		piece := decode(tokenizer, token, next)

		if is_printable(piece) {
			callback(piece)
		}
		token = next

		// init the timer here because the first iteration can be slower
		if start == 0 {
			start = int(time.Now().UnixMilli())
		}
	}
}

func GenerateText(transformer *Transformer, tokenizer *Tokenizer, sampler *Sampler, steps int) string {
	var buf []byte

	Generate(transformer, tokenizer, sampler, steps, func(piece string) {
		buf = append(buf, piece...)
	})

	return string(buf)
}

// ----------------------------------------------------------------------------
// Utils

func binaryRead(r io.Reader, data any) { check(binary.Read(r, binary.LittleEndian, data)) }

func expf(a float32) float32    { return float32(math.Exp(float64(a))) }
func powf(a, b float32) float32 { return float32(math.Pow(float64(a), float64(b))) }
func sqrtf(a float32) float32   { return float32(math.Sqrt(float64(a))) }
func sinf(a float32) float32    { return float32(math.Sin(float64(a))) }
func cosf(a float32) float32    { return float32(math.Cos(float64(a))) }

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func check1[A any](a A, err error) A { check(err); return a }
func checkCall(f func() error)       { check(f()) }
