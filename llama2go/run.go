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
	dim       int32 // transformer dimension
	hiddenDim int32 // for ffn layers
	nLayers   int32 // number of layers
	nHeads    int32 // number of query heads
	nKvHeads  int32 // number of key/value heads (can be < query heads because of multiquery)
	vocabSize int32 // vocabulary size, usually 256 (byte-level)
	seqLen    int32 // max sequence length
}

type transformerWeights struct {
	// token embedding table
	tokenEmbeddingTable []float32 // (vocabSize, dim)
	// weights for rmsnorms
	rmsAttWeight []float32 // (layer, dim) rmsnorm weights
	rmsFfnWeight []float32 // (layer, dim)
	// weights for matmuls. note dim == nHeads * headSize
	wq []float32 // (layer, dim, nHeads * headSize)
	wk []float32 // (layer, dim, nKvHeads * headSize)
	wv []float32 // (layer, dim, nKvHeads * headSize)
	wo []float32 // (layer, nHeads * headSize, dim)
	// weights for ffn
	w1 []float32 // (layer, hiddenDim, dim)
	w2 []float32 // (layer, dim, hiddenDim)
	w3 []float32 // (layer, hiddenDim, dim)
	// final rmsnorm
	rmsFinalWeight []float32 // (dim,)
	// (optional) classifier weights for the logits, on the last layer
	wcls []float32
}

type runState struct {
	// current wave of activations
	x      []float32 // activation at current time stamp (dim,)
	xb     []float32 // same, but inside a residual branch (dim,)
	xb2    []float32 // an additional buffer just for convenience (dim,)
	hb     []float32 // buffer for hidden dimension in the ffn (hiddenDim,)
	hb2    []float32 // buffer for hidden dimension in the ffn (hiddenDim,)
	q      []float32 // query (dim,)
	k      []float32 // key (dim,)
	v      []float32 // value (dim,)
	att    []float32 // buffer for scores/attention values (nHeads, seqLen)
	logits []float32 // output logits
	// kv cache
	keyCache   []float32 // (layer, seqLen, dim)
	valueCache []float32 // (layer, seqLen, dim)
}

type Transformer struct {
	config  config             // the hyperparameters of the architecture (the blueprint)
	weights transformerWeights // the weights of the model
	state   runState           // buffers for the "wave" of activations in the forward pass
}

func makeRunState(s *runState, p *config) {
	kvDim := (p.dim * p.nKvHeads) / p.nHeads
	s.x = make([]float32, p.dim)
	s.xb = make([]float32, p.dim)
	s.xb2 = make([]float32, p.dim)
	s.hb = make([]float32, p.hiddenDim)
	s.hb2 = make([]float32, p.hiddenDim)
	s.q = make([]float32, p.dim)
	s.keyCache = make([]float32, p.nLayers*p.seqLen*kvDim)
	s.valueCache = make([]float32, p.nLayers*p.seqLen*kvDim)
	s.att = make([]float32, p.nHeads*p.seqLen)
	s.logits = make([]float32, p.vocabSize)
}

func memoryMapWeights(w *transformerWeights, p *config, ptr []float32, sharedWeights bool) {
	headSize := p.dim / p.nHeads
	nLayers := p.nLayers
	w.tokenEmbeddingTable = ptr
	ptr = ptr[p.vocabSize*p.dim:]
	w.rmsAttWeight = ptr
	ptr = ptr[nLayers*p.dim:]
	w.wq = ptr
	ptr = ptr[nLayers*p.dim*(p.nHeads*headSize):]
	w.wk = ptr
	ptr = ptr[nLayers*p.dim*(p.nKvHeads*headSize):]
	w.wv = ptr
	ptr = ptr[nLayers*p.dim*(p.nKvHeads*headSize):]
	w.wo = ptr
	ptr = ptr[nLayers*(p.nHeads*headSize)*p.dim:]
	w.rmsFfnWeight = ptr
	ptr = ptr[nLayers*p.dim:]
	w.w1 = ptr
	ptr = ptr[nLayers*p.dim*p.hiddenDim:]
	w.w2 = ptr
	ptr = ptr[nLayers*p.hiddenDim*p.dim:]
	w.w3 = ptr
	ptr = ptr[nLayers*p.dim*p.hiddenDim:]
	w.rmsFinalWeight = ptr
	ptr = ptr[p.dim:]
	ptr = ptr[p.seqLen*headSize/2:] // skip what used to be freqCisReal (for RoPE)
	ptr = ptr[p.seqLen*headSize/2:] // skip what used to be freqCisImag (for RoPE)
	w.wcls = ptr
	if sharedWeights {
		w.wcls = w.tokenEmbeddingTable
	}
}

func readCheckpoint(checkpoint string, cfg *config, weights *transformerWeights) {
	file := check1(os.Open(checkpoint))
	defer checkCall(file.Close)

	// read in the config header
	cw := make([]int32, 7)
	binaryRead(file, cw)
	*cfg = config{cw[0], cw[1], cw[2], cw[3], cw[4], cw[5], cw[6]}

	// negative vocab size is hacky way of signaling unshared weights. bit yikes.
	sharedWeights := cfg.vocabSize > 0
	if !sharedWeights {
		cfg.vocabSize = -cfg.vocabSize
	}

	// figure out the file size
	stat := check1(file.Stat())
	fileSize := stat.Size() // get the file size, in bytes

	// memory map the Transformer weights into the data pointer
	data := make([]float32, fileSize/4-int64(len(cw)))
	binaryRead(file, data)

	//for i, val := range data {
	//	val = math.Float32frombits(math.Float32bits(val) &^ 0xFFFFF)
	//	data[i] = val
	//}

	memoryMapWeights(weights, cfg, data, sharedWeights)
}

func NewTransformer(checkpointPath string) *Transformer {
	t := &Transformer{}
	// read in the Config and the Weights from the checkpoint
	readCheckpoint(checkpointPath, &t.config, &t.weights)
	// allocate the RunState buffers
	makeRunState(&t.state, &t.config)
	return t
}

func (t *Transformer) VocabSize() int {
	return int(t.config.vocabSize)
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
	maxVal := x[0]
	for i := 1; i < size; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	// exp and sum
	sum := float32(0)
	for i := 0; i < size; i++ {
		x[i] = expf(x[i] - maxVal)
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
	hiddenDim := int(p.hiddenDim)
	nLayers := int(p.nLayers)
	nHeads := int(p.nHeads)
	nKvHeads := int(p.nKvHeads)
	vocabSize := int(p.vocabSize)
	seqLen := int(p.seqLen)

	kvDim := (dim * nKvHeads) / nHeads
	kvMul := nHeads / nKvHeads // integer multiplier of the kv sharing in multiquery
	headSize := dim / nHeads

	// copy the token embedding into x
	contentRow := w.tokenEmbeddingTable[token*dim:]
	copy(x, contentRow[:dim])

	// forward all the layers
	for l := 0; l < nLayers; l++ {

		// attention rmsnorm
		rmsnorm(s.xb, x, w.rmsAttWeight[l*dim:], dim)

		// key and value point to the kv cache
		loff := l * seqLen * kvDim // kv cache layer offset for convenience
		s.k = s.keyCache[loff+pos*kvDim:]
		s.v = s.valueCache[loff+pos*kvDim:]

		// qkv matmuls for this position
		matmul(s.q, s.xb, w.wq[l*dim*dim:], dim, dim)
		matmul(s.k, s.xb, w.wk[l*dim*kvDim:], dim, kvDim)
		matmul(s.v, s.xb, w.wv[l*dim*kvDim:], dim, kvDim)

		// RoPE relative positional encoding: complex-valued rotate q and k in each head
		for i := 0; i < dim; i += 2 {
			headDim := i % headSize
			freq := 1.0 / powf(10000, float32(headDim)/float32(headSize))
			val := float32(pos) * freq
			fcr := cosf(val)
			fci := sinf(val)

			// how many vectors? 2 = q & k, 1 = q only
			rotn := 1
			if i < kvDim {
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

		for h := 0; h < nHeads; h++ {
			// get the query vector for this head
			q := s.q[h*headSize:]
			// attention scores for this head
			att := s.att[h*seqLen:]
			// iterate over all timesteps, including the current one
			for t := 0; t <= pos; t++ {
				// get the key vector for this head and at this timestep
				k := s.keyCache[loff+t*kvDim+(h/kvMul)*headSize:]
				// calculate the attention score as the dot product of q and k
				score := float32(0)
				for i := 0; i < headSize; i++ {
					score += q[i] * k[i]
				}
				score /= sqrtf(float32(headSize))
				// save the score to the attention buffer
				att[t] = score
			}

			// softmax the scores to get attention weights, from 0..pos inclusively
			softmax(att, pos+1)

			// weighted sum of the values, store back into xb
			xb := s.xb[h*headSize:]
			clear(xb[:headSize])
			for t := 0; t <= pos; t++ {
				// get the value vector for this head and at this timestep
				v := s.valueCache[loff+t*kvDim+(h/kvMul)*headSize:]
				// get the attention weight for this timestep
				a := att[t]
				// accumulate the weighted value into xb
				for i := 0; i < headSize; i++ {
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
		rmsnorm(s.xb, x, w.rmsFfnWeight[l*dim:], dim)

		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
		// first calculate self.w1(x) and self.w3(x)
		matmul(s.hb, s.xb, w.w1[l*dim*hiddenDim:], dim, hiddenDim)
		matmul(s.hb2, s.xb, w.w3[l*dim*hiddenDim:], dim, hiddenDim)

		// SwiGLU non-linearity
		for i := 0; i < hiddenDim; i++ {
			val := s.hb[i]
			// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
			val *= 1.0 / (1.0 + expf(-val))
			// elementwise multiply with w3(x)
			val *= s.hb2[i]
			s.hb[i] = val
		}

		// final matmul to get the output of the ffn
		matmul(s.xb, s.hb, w.w2[l*dim*hiddenDim:], hiddenDim, dim)

		// residual connection
		for i := 0; i < dim; i++ {
			x[i] += s.xb[i]
		}
	}

	// final rmsnorm
	rmsnorm(x, x, w.rmsFinalWeight, dim)

	// classifier into logits
	matmul(s.logits, x, w.wcls, dim, vocabSize)
	return s.logits
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

type TokenIndex struct {
	str string
	id  int
}

type Tokenizer struct {
	vocab       []string
	vocabScores []float32
	sortedVocab []TokenIndex
	vocabSize   int
}

func NewTokenizer(tokenizerPath string, vocabSize int) *Tokenizer {
	t := &Tokenizer{
		// i should have written the vocabSize into the tokenizer file... sigh
		vocabSize:   vocabSize,
		vocab:       make([]string, vocabSize),
		vocabScores: make([]float32, vocabSize),
	}

	// read in the file
	file := check1(os.Open(tokenizerPath))
	defer checkCall(file.Close)

	var maxTokenLength int32
	binaryRead(file, &maxTokenLength)

	for i := 0; i < vocabSize; i++ {
		binaryRead(file, &t.vocabScores[i])

		var strlen int32
		binaryRead(file, &strlen)

		buf := make([]byte, strlen)
		binaryRead(file, buf)
		t.vocab[i] = string(buf)
	}

	return t
}

func decode(t *Tokenizer, prevToken, token int) string {
	piece := t.vocab[token]
	// following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
	if prevToken == 1 && piece[0] == ' ' {
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

func isPrintable(piece string) bool {
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
type probIndex struct {
	prob  float32
	index int
}

type Sampler struct {
	vocabSize   int
	probIndex   []probIndex // buffer used in top-p sampling
	temperature float32
	topp        float32
	rngState    uint64
}

func sampleArgmax(probabilities []float32, n int) int {
	// return the index that has the highest probability
	maxI := 0
	maxP := probabilities[0]
	for i := 1; i < n; i++ {
		if probabilities[i] > maxP {
			maxI = i
			maxP = probabilities[i]
		}
	}
	return maxI
}

func sampleMult(probabilities []float32, n int, coin float32) int {
	// sample index from probabilities (they must sum to 1!)
	// coin is a random number in [0, 1), usually from randomF32()
	cdf := float32(0)
	for i := 0; i < n; i++ {
		cdf += probabilities[i]
		if coin < cdf {
			return i
		}
	}
	return n - 1 // in case of rounding errors
}

func sampleTopP(probabilities []float32, n int, topp float32, probindex []probIndex, coin float32) int {
	// top-p sampling (or "nucleus sampling") samples from the smallest set of
	// tokens that exceed probability topp. This way we never sample tokens that
	// have very low probabilities and are less likely to go "off the rails".
	// coin is a random number in [0, 1), usually from randomF32()

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
	cumulativeProb := float32(0)
	lastIdx := n0 - 1 // in case of rounding errors consider all elements
	for i := 0; i < n0; i++ {
		cumulativeProb += probindex[i].prob
		if cumulativeProb > topp {
			lastIdx = i
			break // we've exceeded topp by including lastIdx
		}
	}

	// sample from the truncated list
	r := coin * cumulativeProb
	cdf := float32(0)
	for i := 0; i <= lastIdx; i++ {
		cdf += probindex[i].prob
		if r < cdf {
			return probindex[i].index
		}
	}
	return probindex[lastIdx].index // in case of rounding errors
}

func NewSampler(vocabSize int, temperature, topp float32, rngSeed uint64) *Sampler {
	return &Sampler{
		vocabSize:   vocabSize,
		temperature: temperature,
		topp:        topp,
		rngState:    rngSeed,
		// buffer only used with nucleus sampling; may not need but it's ~small
		probIndex: make([]probIndex, vocabSize),
	}
}

func randomU32(state *uint64) int {
	// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
	*state ^= *state >> 12
	*state ^= *state << 25
	*state ^= *state >> 27
	return int((*state * 0x2545F4914F6CDD1D) >> 32)
}

func randomF32(state *uint64) float32 { // random float32 in [0,1)
	return float32(randomU32(state)>>8) / 16777216.0
}

func sample(sampler *Sampler, logits []float32) int {
	// sample the token given the logits and some hyperparameters
	var next int
	if sampler.temperature == 0 {
		// greedy argmax sampling: take the token with the highest probability
		next = sampleArgmax(logits, sampler.vocabSize)
	} else {
		// apply the temperature to the logits
		for q := 0; q < sampler.vocabSize; q++ {
			logits[q] /= sampler.temperature
		}
		// apply softmax to the logits to get the probabilities for next token
		softmax(logits, sampler.vocabSize)
		// flip a (float) coin (this is our source of entropy for sampling)
		coin := randomF32(&sampler.rngState)
		// we sample from this distribution to get the next token
		if sampler.topp <= 0 || sampler.topp >= 1 {
			// simply sample from the predicted probability distribution
			next = sampleMult(logits, sampler.vocabSize, coin)
		} else {
			// top-p (nucleus) sampling, clamping the least likely tokens to zero
			next = sampleTopP(logits, sampler.vocabSize, sampler.topp, sampler.probIndex, coin)
		}
	}
	return next
}

// ----------------------------------------------------------------------------
// generation loop

func Generate(transformer *Transformer, tokenizer *Tokenizer, sampler *Sampler, steps int,
	callback func(string)) {
	steps = min(steps, int(transformer.config.seqLen))

	//encode(tokenizer, prompt, true, false, promptTokens, &numPromptTokens)
	promptTokens := []int{1}
	numPromptTokens := 1

	start := 0               // used to time our code, only initialized after first iteration
	next := 0                // will store the next token in the sequence
	token := promptTokens[0] // kick off with the first token in the prompt
	for pos := 0; pos < steps; pos++ {

		// forward the transformer to get logits for the next token
		logits := forward(transformer, token, pos)

		// advance the state machine
		if pos < numPromptTokens-1 {
			// if we are still processing the input prompt, force the next prompt token
			next = promptTokens[pos+1]
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

		if isPrintable(piece) {
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
