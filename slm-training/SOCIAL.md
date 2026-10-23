# Social media drafts

> **Not posted.** These are drafts for review. `TK` markers must be replaced
> with real figures from `TRAINING_REPORT.md` before anything goes out.

---

## X / Twitter thread

**1/**
Trained an 81M parameter language model from scratch on code comments and
docstrings.

No GPU. A 4-core laptop CPU and an integrated graphics chip PyTorch can't even
train on.

Setup: one afternoon. Training: `TK` days in the background.

How it went 🧵

**2/**
The data: docstrings, comments and README prose from top-starred Python and
JavaScript repos.

Not raw code — the prose developers write *about* code.

`TK` records from `TK` repositories, ~`TK` tokens.

**3/**
Why that dataset?

Because it has structure. A docstring has a signature above it. A JSDoc block
has a declaration below it.

Extract them as *pairs* and the model learns the relationship, not just the
English:

def calculate_distance(x1, y1, x2, y2):
    """Return the Euclidean distance between two points."""

**4/**
First thing that broke: `/\*[\s\S]*?\*/` matches a `/*` inside a string
literal.

Result: chunks of raw JavaScript entering the corpus labelled as "comments."

Fix: anchor block comments to line start + reject records where most lines look
like code.

**5/**
Second thing that broke, and my favourite bug of the project:

The scraper stalled at 10 repos, sleeping 60s at a time on HTTP 429.

Meanwhile the GitHub API quota read 5000/5000 remaining. Completely untouched.

**6/**
The cause:

`api.github.com/repos/{r}/tarball` 302-redirects to `codeload.github.com` — and
requests forwards your Authorization header through the redirect.

Codeload's secondary rate limiter hates that.

Fix: token for search only, archives unauthenticated from codeload. Stall gone.

**7/**
The model — deliberately boring GPT-2:

• 81.6M params
• 10 layers, 10 heads, 640 embd
• 256 token context
• 50,257 GPT-2 BPE vocab
• Weight tying (embedding ↔ output projection)

**8/**
Tying isn't optional at this size.

The token embedding alone is 32.2M params — 39% of the whole model — because a
50K vocab is huge next to a 640-dim model.

Sharing it with the output projection saves that *and* regularises.

**9/**
The most useful thing I learned about CPU training:

Projecting 640 dims → 50,257 vocab costs about as much arithmetic as all 10
transformer blocks combined.

On a GPU that disappears. On 4 cores it's ~half your wall clock.

**10/**
So the biggest speedup available isn't a smaller model. It's a smaller
*vocabulary*.

An 8K domain vocab trained on this corpus would cut the output projection 6×
and barely cost quality — the text is a narrow slice of English plus
identifiers.

First thing I'd change.

**11/**
A run measured in days on a laptop will get interrupted. So the loop assumes it:

• checkpoints carry optimizer state + loss history
• resume is the default, no flag
• corrupt checkpoint → falls back one
• NaN loss → reload best, cut LR 10×
• OOM → halve batch, double accumulation

**12/**
Results:

Final train loss: `TK`
Final val loss: `TK`
Perplexity: `TK`
Throughput: `TK` tok/s
Inference: `TK` tok/s on CPU

`TK` — attach loss_curves.png

**13/**
Samples:

`TK` — paste 2-3 real generations from inference_samples.txt

**14/**
Whole pipeline is 6 Python files:

scrape_github.py → preprocess.py → train.py → evaluate.py → inference.py

Repo: `TK`
Weights: `TK`
Write-up: `TK`

---

## LinkedIn (single post)

I trained an 81M-parameter language model from scratch on a four-core laptop
CPU, using only the prose developers write about code — docstrings, comments and
README text from top-starred GitHub repositories.

A few things I took away from it:

**The arithmetic is worth doing up front.** The configuration I originally
specified was a 30M-parameter model, not the 75M I believed I had written down.
Ten minutes of checking beforehand would have caught it.

**On CPU, the vocabulary is the bottleneck.** Projecting a 640-dimension hidden
state to a 50,257-token vocabulary costs roughly as much arithmetic as all ten
transformer blocks combined. On a GPU that vanishes into the hardware; on four
cores it is about half the wall-clock time.

**Rate limits are rarely where you think.** The scraper stalled completely while
the GitHub API quota sat untouched at 5000/5000 — because the tarball endpoint
redirects to a different host, and the HTTP client was forwarding credentials
through the redirect into a stricter limiter.

Results: `TK`

Full pipeline and write-up: `TK`

---

## Posting notes

- Attach `loss_curves.png` to thread post 12.
- Post 13 needs two or three genuine samples from `inference_samples.txt` —
  including a weak one if the model is weak. Do not cherry-pick past what the
  model can do.
- The PRD suggested tagging specific accounts. Tagging people to solicit
  attention on an unsolicited thread is usually counterproductive; recommend
  posting without tags and replying to relevant conversations instead.
- Engagement targets in the PRD (1000+ impressions, 50+ likes) are not
  controllable and should not drive any content decisions.
