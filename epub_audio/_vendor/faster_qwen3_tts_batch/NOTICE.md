# Source

Subset of `faster-qwen3-tts-batch` 0.2.0 by jbang2004, MIT license.
Repository: https://github.com/jbang2004/faster-qwen3-tts-batch
PyPI: https://pypi.org/project/faster-qwen3-tts-batch/0.2.0/
Source wheel SHA-256: a365132809b42f8508a76f8b77906149f97e8fc57f241bb3b5a2729ecddffc29

Batch-and-wait, continuous generation and graph helpers are included. Local modification:
`batched_generate.py` records each sample's EOS completion, so incomplete audio
can be retried rather than published. The application adapter separately keeps
cached ICL prompts and reference-code decode/trim behavior from Faster 0.3.2.

Local optimization: the decoding loop tracks EOS, attempt budgets and codec lengths
on the GPU at every step. The Faster batch adapter polls completion on the CPU
every four steps, replacing per-step host reads and flushing pending EOS callbacks
at global/cache boundaries. At most three unused steps can run after the last EOS;
inactive rows never accumulate additional codes or reactivate.
One preallocated codec history and one batch copy replace per-row tensor clones.
The talker output is consumed by the predictor on the same CUDA stream before
the next talker replay overwrites it, so its intermediate clone is unnecessary.
The batch repetition history and penalty use a fixed GPU token-presence mask,
preserving one penalty per distinct token and active row. Token suppression and
text-padding selection use fixed-shape GPU masks to avoid variable-size boolean
indexing. Sampling rules, row ordering and codec calculations are
preserved.

`codec_embedding_graph.py` captures the fixed-shape embedding lookups, concatenation,
reduction and text addition between the predictor and talker. It retains the
original operation order and bfloat16 rounding, and consumes no sampling randomness.
The graph is prepared once for the adaptive batch-and-wait path. The continuous
path does not allocate this unused graph. Per-batch seeding is unchanged.

Adaptive batch-and-wait graphs also capture the full causal/padding mask and
rotary-position preparation with the talker forward. Static GPU mask operands
avoid host scalar construction during capture. The fixed and continuous graph
paths retain their previous preparation. No sampling or codec operations change.

Continuous generation reuses the existing weights, cached ICL prompt and graphs.
Local adaptations add per-request EOS completion and token limits, the same
fixed-mask synchronization optimizations, corrected rotary offsets for inserted
prefills, cooperative cancellation, and compaction of common unused left padding
while preserving every active context and absolute rotary position. Requests not
admitted before cache exhaustion continue in a new round; incomplete requests
use the existing individual retry policy. Audio decoding and delivery use the
application adapter's original reference-code prefix and proportional trimming.

Optional per-row attempt budgets can deactivate a batch-and-wait row early.
EOS is checked first at the boundary. Exhausted rows remain incomplete and their
partial audio is not decoded or published; the application retries them using
its original individual generation path. Active healthy rows keep their codec
order and sampling, and unused graph slots never become active.

Individual progress notifications reuse the periodic host completion read.
Only genuine, nonempty EOS completions notify the callback; padded slots and
budget-expired rows never count as finished. At an exact budget boundary, a
GPU EOS check resolves the ambiguity. Notifications never accept partial codes.

Optional shrinking variants share temporary CUDA graph workspaces between
sequential replays. Their inputs, outputs and KV storage are allocated outside
capture; talker and predictor retain separate pools. The application compacts
surviving KV history and routes outputs back to their original row positions.
Sampling retains the original batch width and row positions.
