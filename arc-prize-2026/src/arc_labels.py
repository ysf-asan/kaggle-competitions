"""Building the training labels for completion-only test-time training.

The reference collator scans for two hard-coded token ids — 11 for `user` and
12 for `assistant` — and supervises the span from two tokens past an assistant
marker to just past the next end-of-turn. That works as long as the tokeniser
lays the chat markers out exactly as it did for the reference. When it does
not, the scan finds nothing, every label stays at -100, and test-time training
silently becomes a no-op: no error, no loss, an adapter that never moves.

So derive the marker token sequences from the tokeniser at run time and search
for those sequences instead. With the expected 16-token vocabulary this
produces exactly the same labels as the reference; when the ids differ it still
produces the right ones.
"""

IGNORE_INDEX = -100


def find_subsequence(ids, seq):
    """Every start position at which `seq` occurs in `ids`."""
    if not seq:
        return []
    n, m = len(ids), len(seq)
    return [i for i in range(n - m + 1) if ids[i:i + m] == seq]


def assistant_starts(ids, assistant_header, user_header=None):
    """Positions of the assistant headers in the transcript.

    This model's vocabulary has no room for the words "user" and "assistant",
    so both headers tokenise to the same thing and cannot be told apart by
    content. They can be told apart by position: the transcript the formatter
    writes strictly alternates user, assistant, user, assistant, so with
    indistinguishable headers every second occurrence, starting from the
    second, is an assistant turn.
    """
    starts = find_subsequence(ids, assistant_header)
    if user_header is not None and list(user_header) == list(assistant_header):
        return starts[1::2]
    return starts


def build_completion_labels(ids, assistant_header, end_marker,
                            user_header=None, ignore_index=IGNORE_INDEX):
    """Supervise each assistant turn: its content plus the closing marker.

    Everything else — the prompt, the demonstration inputs, the headers
    themselves and the padding — is ignored. Supervising the inputs too would
    spend half the gradient teaching the model to copy grids it was given.
    An assistant header with no end marker after it is skipped rather than run
    to the end of the sequence.
    """
    ids = list(ids)
    labels = [ignore_index] * len(ids)

    ends = find_subsequence(ids, end_marker)
    for pos in assistant_starts(ids, assistant_header, user_header):
        start = pos + len(assistant_header)
        end = next((e for e in ends if e >= start), None)
        if end is None:
            continue
        stop = end + len(end_marker)
        labels[start:stop] = ids[start:stop]

    return labels


def marker_sequences(tokenizer):
    """Token ids for the chat markers the formatter writes."""
    def encode(text):
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))

    return {
        "user": encode("<|im_start|>user\n"),
        "assistant": encode("<|im_start|>assistant\n"),
        "end": encode("<|im_end|>"),
    }
