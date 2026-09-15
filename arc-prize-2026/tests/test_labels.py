"""Tests for the training-label construction.

This is the code whose failure mode is invisible: get it wrong and every label
is -100, the loss is zero, the adapter never moves, and the run completes
looking healthy while doing nothing. The reference's hard-coded marker ids did
exactly that here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arc_labels import (
    IGNORE_INDEX, build_completion_labels, find_subsequence, marker_sequences,
)

USER = [14, 11, 10]
ASST = [14, 12, 10]
END = [15]


def turn(header, content):
    return header + content + END


def test_find_subsequence_reports_every_occurrence():
    assert find_subsequence([1, 2, 3, 1, 2], [1, 2]) == [0, 3]
    assert find_subsequence([1, 2, 3], [9]) == []
    assert find_subsequence([1, 2], [1, 2, 3]) == []
    assert find_subsequence([1, 2], []) == []


def test_only_the_assistant_content_is_supervised():
    ids = turn(USER, [1, 2, 10, 3, 4]) + turn(ASST, [5, 6, 10, 7, 8])
    labels = build_completion_labels(ids, ASST, END)

    supervised = [i for i, l in enumerate(labels) if l != IGNORE_INDEX]
    start = len(turn(USER, [1, 2, 10, 3, 4])) + len(ASST)
    assert supervised == list(range(start, len(ids)))
    assert [labels[i] for i in supervised] == [5, 6, 10, 7, 8, 15]


def test_the_prompt_is_never_supervised():
    ids = turn(USER, [1, 2]) + turn(ASST, [3, 4])
    labels = build_completion_labels(ids, ASST, END)
    prompt_len = len(turn(USER, [1, 2]))
    assert all(l == IGNORE_INDEX for l in labels[:prompt_len])


def test_every_demonstration_turn_is_supervised():
    """The objective is to predict each demonstration output, not just the
    last one; supervising only the final turn would throw away most of the
    signal a task's few examples carry."""
    ids = (turn(USER, [1]) + turn(ASST, [2])
           + turn(USER, [3]) + turn(ASST, [4])
           + turn(USER, [5]) + turn(ASST, [6]))
    labels = build_completion_labels(ids, ASST, END)
    assert [l for l in labels if l != IGNORE_INDEX] == [2, 15, 4, 15, 6, 15]


def test_padding_after_the_last_turn_stays_masked():
    ids = turn(USER, [1]) + turn(ASST, [2]) + [13, 13, 13]
    labels = build_completion_labels(ids, ASST, END)
    assert labels[-3:] == [IGNORE_INDEX] * 3


def test_a_truncated_final_turn_is_skipped():
    ids = turn(USER, [1]) + ASST + [2, 3]
    labels = build_completion_labels(ids, ASST, END)
    assert all(l == IGNORE_INDEX for l in labels)


def test_nothing_is_supervised_when_the_markers_are_absent():
    ids = [1, 2, 3, 4, 5]
    assert build_completion_labels(ids, ASST, END) == [IGNORE_INDEX] * 5


def test_multi_token_and_single_token_markers_agree():
    """Whether the tokenizer emits the header as one token or three, the
    supervised span is the same."""
    content = [7, 8, 10, 9]
    long_ids = USER + [1] + END + ASST + content + END
    short_ids = [11] + [1] + END + [12] + content + END

    long_labels = build_completion_labels(long_ids, ASST, END)
    short_labels = build_completion_labels(short_ids, [12], END)

    assert ([l for l in long_labels if l != IGNORE_INDEX]
            == [l for l in short_labels if l != IGNORE_INDEX]
            == content + END)


def test_marker_sequences_uses_the_tokenizer():
    class FakeTokenizer:
        table = {
            "<|im_start|>user\n": [14, 11, 10],
            "<|im_start|>assistant\n": [14, 12, 10],
            "<|im_end|>": [15],
        }

        def encode(self, text, add_special_tokens=True):
            return self.table[text]

    assert marker_sequences(FakeTokenizer()) == {
        "user": [14, 11, 10], "assistant": [14, 12, 10], "end": [15],
    }


def test_marker_sequences_tolerates_a_tokenizer_without_the_keyword():
    class OldTokenizer:
        def encode(self, text):
            return [1, 2]

    assert marker_sequences(OldTokenizer())["end"] == [1, 2]


SAME = [14, 10]


def test_indistinguishable_headers_supervise_only_the_assistant_turns():
    ids = (SAME + [1, 2] + END
           + SAME + [3, 4] + END
           + SAME + [5, 6] + END
           + SAME + [7, 8] + END)
    labels = build_completion_labels(ids, SAME, END, user_header=SAME)
    assert [l for l in labels if l != IGNORE_INDEX] == [3, 4, 15, 7, 8, 15]


def test_indistinguishable_headers_leave_the_inputs_masked():
    ids = SAME + [1, 2] + END + SAME + [3, 4] + END
    labels = build_completion_labels(ids, SAME, END, user_header=SAME)
    assert all(l == IGNORE_INDEX for l in labels[:len(SAME) + 3])


def test_roughly_half_the_content_is_supervised_when_headers_collide():
    """The symptom that gave this away: 515 of 531 positions supervised meant
    the inputs were being trained on too."""
    turns = []
    for i in range(6):
        turns += SAME + [i] * 10 + END
    labels = build_completion_labels(turns, SAME, END, user_header=SAME)
    supervised = sum(1 for l in labels if l != IGNORE_INDEX)
    assert 0.4 < supervised / len(turns) < 0.6


def test_distinguishable_headers_are_matched_directly():
    ids = turn(USER, [1, 2]) + turn(ASST, [3, 4])
    with_user = build_completion_labels(ids, ASST, END, user_header=USER)
    without = build_completion_labels(ids, ASST, END)
    assert with_user == without
    assert [l for l in with_user if l != IGNORE_INDEX] == [3, 4, 15]


def test_assistant_starts_alternates_only_when_headers_collide():
    ids = SAME + [1] + END + SAME + [2] + END + SAME + [3] + END
    from arc_labels import assistant_starts
    assert len(assistant_starts(ids, SAME, user_header=SAME)) == 1
    assert len(assistant_starts(ids, SAME, user_header=[9, 9])) == 3
    assert len(assistant_starts(ids, SAME)) == 3
