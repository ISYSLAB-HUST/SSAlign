from typing import Tuple, Optional
from numba import jit as nb_jit
from numba import njit
import numpy as np
import numba as nb
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# Constants for operation codes
MATCH, INSERT, DELETE, SUBSTITUTE = 0, 1, 2, 3

# By default, we use BLOSUM62 with affine gap penalties
# default_proteins_alphabet: str = "ARNDCQEGHILKMFPSTWYVBZX"
default_proteins_alphabet: str = "ACDEFGHIKLMNPQRSTVWYX"

default_proteins_matrix = (
        np.array(
            [
                # fmt: off
                4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0, -4,
                -1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4,
                -2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4,
                -2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4,
                0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4,
                -1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4,
                -1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4,
                0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4,
                -2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4,
                -1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4,
                -1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4,
                -1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4,
                -1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4,
                -2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4,
                -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2, -4,
                1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4,
                0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4,
                -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4,
                -2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4,
                0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4,
                -2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4,
                -1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4,
                0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4,
                -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1,
                # fmt: on
            ],
            dtype=np.int8,
        ).reshape(24, 24)
        * 5
)
default_gap_opening: int = -4 * 5  # -20
default_gap_extension: int = int(-0.2 * 5)  # -1

# Precompute mapping arrays for substitution alphabets to speed up sequence translation
# _mapping_cache = {}

# def _get_mapping_array(alphabet: str) -> np.ndarray:
#     """
#     Create a mapping array from ASCII characters to their indices in the substitution alphabet.
#     Unknown characters are mapped to the last index.
#     This function caches the mapping arrays to avoid recomputation.
#     """
#     if alphabet in _mapping_cache:
#         return _mapping_cache[alphabet]
#
#     mapping = np.full(256, len(alphabet) - 1, dtype=np.uint8)  # Default to last index for unknowns
#     for idx, char in enumerate(alphabet):
#         char_ord = ord(char)
#         if char_ord < 256:
#             mapping[char_ord] = idx
#     _mapping_cache[alphabet] = mapping
#     return mapping

@nb_jit(nopython=True)
def _get_mapping_array(alphabet: str) -> np.ndarray:
    mapping = np.full(256, len(alphabet) - 1, dtype=np.uint8)
    for idx, char in enumerate(alphabet):
        char_ord = ord(char)
        if char_ord < 256:
            mapping[char_ord] = idx
    return mapping


# @nb.njit
# def _translate_sequence(seq: str, alphabet: str) -> np.ndarray:
#     """
#     Translate a sequence string into a NumPy array of indices based on the substitution alphabet.
#     Uses a precomputed mapping array for fast translation.
#     """
#     mapping = _get_mapping_array(alphabet)
#     seq_bytes = seq.encode('ascii', errors='ignore')  # Ignore non-ASCII characters
#     seq_array = np.frombuffer(seq_bytes, dtype=np.uint8)
#     if seq_array.size == 0:
#         return np.array([], dtype=np.uint8)
#     seq_encoded = mapping[seq_array]
#     return seq_encoded


empty_array = np.array([], dtype=np.uint8)

@njit(nopython=True)
def _translate_sequence(seq: str, alphabet: str) -> np.ndarray:
    mapping = _get_mapping_array(alphabet)
    seq_len = len(seq)
    seq_array = np.zeros(seq_len, dtype=np.uint8)
    for i in range(seq_len):
        char = seq[i]
        if 'A' <= char <= 'Z':
            seq_array[i] = ord(char)
        else:
            seq_array[i] = 255  # 未知字符，根据需要定义
    if seq_array.size == 0:
        return empty_array
    seq_encoded = mapping[seq_array]
    return seq_encoded




@nb.njit(fastmath=True, boundscheck=False, cache=True, inline='always')
def _levenshtein_alignment_kernel(seq1: np.ndarray, seq2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    scores = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    changes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.uint8)

    scores[0, 0] = 0
    for i in range(1, seq1_len + 1):
        scores[i, 0] = i
        changes[i, 0] = DELETE
    for j in range(1, seq2_len + 1):
        scores[0, j] = j
        changes[0, j] = INSERT

    for i in range(1, seq1_len + 1):
        for j in range(1, seq2_len + 1):
            a = seq1[i - 1]
            b = seq2[j - 1]
            substitution = 0 if a == b else 1

            score_diag = scores[i - 1, j - 1] + substitution
            score_up = scores[i - 1, j] + 1
            score_left = scores[i, j - 1] + 1

            # Manually compute minimum
            min_score = score_diag
            if score_up < min_score:
                min_score = score_up
            if score_left < min_score:
                min_score = score_left

            scores[i, j] = min_score

            if substitution == 0:
                changes[i, j] = MATCH
            elif min_score == score_diag:
                changes[i, j] = SUBSTITUTE
            elif min_score == score_up:
                changes[i, j] = DELETE
            else:
                changes[i, j] = INSERT

    return scores, changes


@nb.njit(fastmath=True, boundscheck=False, cache=True, inline='always')
def _needleman_wunsch_gotoh_kernel(
        seq1: np.ndarray,
        seq2: np.ndarray,
        substitution_matrix: np.ndarray,
        gap_opening: int,
        gap_extension: int,
) -> Tuple[np.ndarray, np.ndarray]:
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    scores = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    deletes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    inserts = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    changes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.uint8)

    scores[0, 0] = 0
    for j in range(1, seq2_len + 1):
        scores[0, j] = gap_opening + (j - 1) * gap_extension
        deletes[0, j] = scores[0, j] + gap_opening + gap_extension
        changes[0, j] = INSERT
    for i in range(1, seq1_len + 1):
        scores[i, 0] = gap_opening + (i - 1) * gap_extension
        inserts[i, 0] = scores[i, 0] + gap_opening + gap_extension
        changes[i, 0] = DELETE

    for i in range(1, seq1_len + 1):
        for j in range(1, seq2_len + 1):
            a = seq1[i - 1]
            b = seq2[j - 1]
            substitution = substitution_matrix[a, b]

            score_diag = scores[i - 1, j - 1] + substitution
            score_up = max(scores[i - 1, j] + gap_opening, deletes[i - 1, j] + gap_extension)
            score_left = max(scores[i, j - 1] + gap_opening, inserts[i, j - 1] + gap_extension)

            # Manually compute maximum
            max_score = score_diag
            operation = SUBSTITUTE
            if score_up > max_score:
                max_score = score_up
                operation = DELETE
            if score_left > max_score:
                max_score = score_left
                operation = INSERT

            scores[i, j] = max_score
            deletes[i, j] = score_up
            inserts[i, j] = score_left

            if substitution == 0:
                changes[i, j] = MATCH
            elif operation == SUBSTITUTE:
                changes[i, j] = SUBSTITUTE
            elif operation == DELETE:
                changes[i, j] = DELETE
            else:
                changes[i, j] = INSERT

    return scores, changes


@nb.njit(fastmath=True, boundscheck=False, cache=True, inline='always')
def needleman_wunsch_gotoh_score_kernel(
        seq1: np.ndarray,
        seq2: np.ndarray,
        substitution_matrix: np.ndarray,
        gap_opening: int,
        gap_extension: int,
) -> int:
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    old_scores = np.empty(seq2_len + 1, dtype=np.int32)
    old_deletes = np.empty(seq2_len + 1, dtype=np.int32)
    old_inserts = np.empty(seq2_len + 1, dtype=np.int32)
    new_scores = np.empty(seq2_len + 1, dtype=np.int32)
    new_deletes = np.empty(seq2_len + 1, dtype=np.int32)
    new_inserts = np.empty(seq2_len + 1, dtype=np.int32)

    old_scores[0] = 0
    for j in range(1, seq2_len + 1):
        old_scores[j] = gap_opening + (j - 1) * gap_extension
        old_deletes[j] = old_scores[j] + gap_opening + gap_extension

    for i in range(1, seq1_len + 1):
        new_scores[0] = gap_opening + (i - 1) * gap_extension
        new_inserts[0] = new_scores[0] + gap_opening + gap_extension

        for j in range(1, seq2_len + 1):
            a = seq1[i - 1]
            b = seq2[j - 1]
            substitution = substitution_matrix[a, b]

            score_diag = old_scores[j - 1] + substitution
            score_up = old_scores[j] + gap_opening
            if old_deletes[j] + gap_extension > score_up:
                score_up = old_deletes[j] + gap_extension

            score_left = new_scores[j - 1] + gap_opening
            if new_inserts[j - 1] + gap_extension > score_left:
                score_left = new_inserts[j - 1] + gap_extension

            # Manually compute maximum
            max_score = score_diag
            if score_up > max_score:
                max_score = score_up
            if score_left > max_score:
                max_score = score_left

            new_scores[j] = max_score
            new_deletes[j] = score_up
            new_inserts[j] = score_left

        # Swap rows
        old_scores, new_scores = new_scores, old_scores
        old_deletes, new_deletes = new_deletes, old_deletes
        old_inserts, new_inserts = new_inserts, old_inserts

    return old_scores[-1]


@nb.njit(fastmath=True, boundscheck=False, cache=True, inline='always')
def _smith_waterman_gotoh_kernel(
        seq1: np.ndarray,
        seq2: np.ndarray,
        substitution_matrix: np.ndarray,
        gap_opening: int,
        gap_extension: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    scores = np.zeros((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    deletes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    inserts = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    changes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.uint8)

    # Initialize first row
    for j in range(1, seq2_len + 1):
        scores[0, j] = 0
        deletes[0, j] = gap_opening + gap_extension
        changes[0, j] = INSERT

    max_score = 0
    max_pos_i = 0
    max_pos_j = 0

    for i in range(1, seq1_len + 1):
        scores[i, 0] = 0
        inserts[i, 0] = gap_opening + gap_extension
        changes[i, 0] = DELETE

        for j in range(1, seq2_len + 1):
            a = seq1[i - 1]
            b = seq2[j - 1]
            substitution = substitution_matrix[a, b]

            score_diag = scores[i - 1, j - 1] + substitution
            score_up = scores[i - 1, j] + gap_opening
            if deletes[i - 1, j] + gap_extension > score_up:
                score_up = deletes[i - 1, j] + gap_extension

            score_left = scores[i, j - 1] + gap_opening
            if inserts[i, j - 1] + gap_extension > score_left:
                score_left = inserts[i, j - 1] + gap_extension

            # Compute max(score_diag, score_up, score_left, 0)
            max_score_cell = score_diag
            change = SUBSTITUTE
            if score_up > max_score_cell:
                max_score_cell = score_up
                change = DELETE
            if score_left > max_score_cell:
                max_score_cell = score_left
                change = INSERT
            if max_score_cell < 0:
                max_score_cell = 0

            scores[i, j] = max_score_cell
            deletes[i, j] = score_up
            inserts[i, j] = score_left

            if substitution == 0:
                changes[i, j] = MATCH
            elif max_score_cell == score_diag:
                changes[i, j] = SUBSTITUTE
            elif max_score_cell == score_up:
                changes[i, j] = DELETE
            else:
                changes[i, j] = INSERT

            if max_score_cell > max_score:
                max_score = max_score_cell
                max_pos_i = i
                max_pos_j = j

    return scores, changes, (max_pos_i, max_pos_j)


@nb.njit(fastmath=True, boundscheck=False, cache=True, inline='always')
def smith_waterman_gotoh_score_kernel(
        seq1: np.ndarray,
        seq2: np.ndarray,
        substitution_matrix: np.ndarray,
        gap_opening: int,
        gap_extension: int,
) -> int:
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    old_scores = np.zeros(seq2_len + 1, dtype=np.int32)
    old_deletes = np.empty(seq2_len + 1, dtype=np.int32)
    old_inserts = np.empty(seq2_len + 1, dtype=np.int32)
    new_scores = np.zeros(seq2_len + 1, dtype=np.int32)
    new_deletes = np.empty(seq2_len + 1, dtype=np.int32)
    new_inserts = np.empty(seq2_len + 1, dtype=np.int32)

    for j in range(1, seq2_len + 1):
        old_scores[j] = 0
        old_deletes[j] = gap_opening + gap_extension

    max_score = 0

    for i in range(1, seq1_len + 1):
        new_scores[0] = 0
        new_inserts[0] = gap_opening + gap_extension

        for j in range(1, seq2_len + 1):
            a = seq1[i - 1]
            b = seq2[j - 1]
            substitution = substitution_matrix[a, b]

            score_diag = old_scores[j - 1] + substitution
            score_up = old_scores[j] + gap_opening
            if old_deletes[j] + gap_extension > score_up:
                score_up = old_deletes[j] + gap_extension

            score_left = new_scores[j - 1] + gap_opening
            if new_inserts[j - 1] + gap_extension > score_left:
                score_left = new_inserts[j - 1] + gap_extension

            # Compute max(score_diag, score_up, score_left, 0)
            max_score_cell = score_diag
            if score_up > max_score_cell:
                max_score_cell = score_up
            if score_left > max_score_cell:
                max_score_cell = score_left
            if max_score_cell < 0:
                max_score_cell = 0

            new_scores[j] = max_score_cell
            new_deletes[j] = score_up
            new_inserts[j] = score_left

            if max_score_cell > max_score:
                max_score = max_score_cell

        # Swap rows
        old_scores, new_scores = new_scores, old_scores
        old_deletes, new_deletes = new_deletes, old_deletes
        old_inserts, new_inserts = new_inserts, old_inserts

    return max_score


@nb.njit(fastmath=True, boundscheck=False, cache=True, inline='always')
def _reconstruct_alignment_kernel(
        changes: np.ndarray,
        seq1: np.ndarray,
        seq2: np.ndarray,
        max_i: int,
        max_j: int,
) -> Tuple[np.ndarray, np.ndarray]:
    align1 = []
    align2 = []
    i, j = max_i, max_j

    while i > 0 and j > 0:
        change = changes[i, j]
        if change == MATCH or change == SUBSTITUTE:
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif change == DELETE:
            align1.append(seq1[i - 1])
            align2.append(255)  # Use 255 as a placeholder for '-'
            i -= 1
        elif change == INSERT:
            align1.append(255)  # Use 255 as a placeholder for '-'
            align2.append(seq2[j - 1])
            j -= 1
        else:
            break  # For local alignment, stop if no valid change

    # Convert to arrays for output
    align1_array = np.empty(len(align1), dtype=np.uint8)
    align2_array = np.empty(len(align2), dtype=np.uint8)
    for idx in range(len(align1)):
        align1_array[idx] = align1[len(align1) - 1 - idx]
        align2_array[idx] = align2[len(align2) - 1 - idx]
    return align1_array, align2_array


def _reconstruct_alignment(
        changes: np.ndarray,
        seq1: np.ndarray,
        seq2: np.ndarray,
        substitution_alphabet: str,
        max_pos: Optional[Tuple[int, int]] = None,
        is_local: bool = False
) -> Tuple[str, str]:
    if is_local and max_pos is not None:
        max_i, max_j = max_pos
    else:
        max_i, max_j = len(seq1), len(seq2)

    align1_array, align2_array = _reconstruct_alignment_kernel(
        changes,
        seq1,
        seq2,
        max_i,
        max_j
    )

    # Convert placeholder 255 back to '-'
    align1 = ''.join([substitution_alphabet[c] if c != 255 else '-' for c in align1_array])
    align2 = ''.join([substitution_alphabet[c] if c != 255 else '-' for c in align2_array])

    return align1, align2

@nb.njit
def _validate_gotoh_arguments(
        substitution_alphabet: Optional[str],
        substitution_matrix: Optional[np.ndarray],
        gap_opening: Optional[int],
        gap_extension: Optional[int],
        match: Optional[int],
        mismatch: Optional[int],
) -> Tuple[str, np.ndarray, int, int]:
    if (match is not None) != (mismatch is not None):
        raise ValueError("Both match and mismatch must be provided.")
    if (match is not None) and (substitution_matrix is not None):
        raise ValueError("Cannot provide both match/mismatch and a substitution matrix.")

    if substitution_alphabet is None:
        substitution_alphabet = default_proteins_alphabet
    if substitution_matrix is None:
        n = len(substitution_alphabet)
        if match is None:
            substitution_matrix = default_proteins_matrix.copy()
        else:
            substitution_matrix = np.full((n, n), mismatch, dtype=np.int8)
            for i in range(n):
                substitution_matrix[i, i] = match
    else:
        # Ensure substitution_matrix is of type int8 and C contiguous
        # substitution_matrix = substitution_matrix.astype(np.int8, copy=False)
        substitution_matrix = np.asarray(substitution_matrix, dtype=np.int8)

        # if not substitution_matrix.flags['C_CONTIGUOUS']:
        #     substitution_matrix = np.ascontiguousarray(substitution_matrix)
        if not substitution_matrix.flags.c_contiguous:
            # 处理非连续数组
            substitution_matrix =  np.ascontiguousarray(substitution_matrix)

    if gap_opening is None:
        gap_opening = default_gap_opening
    if gap_extension is None:
        gap_extension = default_gap_extension

    return substitution_alphabet, substitution_matrix, gap_opening, gap_extension


def levenshtein_alignment(str1: str, str2: str) -> Tuple[str, str, int]:
    # Use all ASCII characters for Levenshtein
    substitution_alphabet = ''.join([chr(i) for i in range(256)])
    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)
    scores, changes = _levenshtein_alignment_kernel(seq1, seq2)
    align1, align2 = _reconstruct_alignment(
        changes,
        seq1,
        seq2,
        substitution_alphabet=substitution_alphabet
    )
    return align1, align2, int(scores[-1, -1])


def needleman_wunsch_gotoh_alignment(
        str1: str,
        str2: str,
        substitution_alphabet: Optional[str] = None,
        substitution_matrix: Optional[np.ndarray] = None,
        gap_opening: Optional[int] = None,
        gap_extension: Optional[int] = None,
        match: Optional[int] = None,
        mismatch: Optional[int] = None,
) -> Tuple[str, str, int]:
    substitution_alphabet, substitution_matrix, gap_opening, gap_extension = _validate_gotoh_arguments(
        substitution_alphabet=substitution_alphabet,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
        match=match,
        mismatch=mismatch,
    )

    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)
    scores, changes = _needleman_wunsch_gotoh_kernel(
        seq1,
        seq2,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
    )

    align1, align2 = _reconstruct_alignment(
        changes,
        seq1,
        seq2,
        substitution_alphabet
    )
    return align1, align2, int(scores[-1, -1])

@nb.njit
def needleman_wunsch_gotoh_score_opm(
        str1: str,
        str2: str,
        substitution_alphabet: Optional[str] = None,
        substitution_matrix: Optional[np.ndarray] = None,
        gap_opening: Optional[int] = None,
        gap_extension: Optional[int] = None,
        match: Optional[int] = None,
        mismatch: Optional[int] = None,
) -> int:
    substitution_alphabet, substitution_matrix, gap_opening, gap_extension = _validate_gotoh_arguments(
        substitution_alphabet=substitution_alphabet,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
        match=match,
        mismatch=mismatch,
    )

    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)

    score = needleman_wunsch_gotoh_score_kernel(
        seq1,
        seq2,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
    )

    return int(score)


@nb.njit
def saligner(target, query):
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    substitutions = np.array([
        [6  , -3 , 1, 2, 3, -2, -2, -7, -3, -3, -10, -5, -1, 1, -4, -7, -5, -6, 0, -2, 0],
        [-3 , 6  , -2, -8, -5, -4, -4, -12, -13, 1, -14, 0, 0, 1, -1, 0, -8, 1, -7, -9, 0],
        [1  , -2 , 4, -3, 0, 1, 1, -3, -5, -4, -5, -2, 1, -1, -1, -4, -2, -3, -2, -2, 0],
        [2  , -8 , -3, 9, -2, -7, -4, -12, -10, -7, -17, -8, -6, -3, -8, -10, -10, -13, -6, -3, 0],
        [3  , -5 , 0, -2, 7, -3, -3, -5, 1, -3, -9, -5, -2, 2, -5, -8, -3, -7, 4, -4, 0],
        [-2 , -4 , 1, -7, -3, 6, 3, 0, -7, -7, -1, -2, -2, -4, 3, -3, 4, -6, -4, -2, 0],
        [-2 , -4 , 1, -4, -3, 3, 6, -4, -7, -6, -6, 0, -1, -3, 1, -3, -1, -5, -5, 3, 0],
        [-7 , -12, -3, -12, -5, 0, -4, 8, -5, -11, 7, -7, -6, -6, -3, -9, 6, -12, -5, -8, 0],
        [-3 , -13, -5, -10, 1, -7, -7, -5, 9, -11, -8, -12, -6, -5, -9, -14, -5, -15, 5, -8, 0],
        [-3 , 1, -4, -7, -3, -7, -6, -11, -11, 6, -16, -3, -2, 2, -4, -4, -9, 0, -8, -9, 0],
        [10 , -14, -5, -17, -9, -1, -6, 7, -8, -16, 10, -9, -9, -10, -5, -10, 3, -16, -6, -9, 0],
        [-5 , 0, -2, -8, -5, -2, 0, -7, -12, -3, -9, 7, 0, -2, 2, 3, -4, 0, -8, -5, 0],
        [-1 , 0, 1, -6, -2, -2, -1, -6, -6, -2, -9, 0, 4, 0, 0, -2, -4, 0, -4, -5, 0],
        [1  , 1, -1, -3, 2, -4, -3, -6, -5, 2, -10, -2, 0, 5, -2, -4, -5, -1, -2, -5, 0],
        [-4 , -1, -1, -8, -5, 3, 1, -3, -9, -4, -5, 2, 0, -2, 6, 2, 0, -1, -6, -3, 0],
        [-7 , 0, -4, -10, -8, -3, -3, -9, -14, -4, -10, 3, -2, -4, 2, 6, -6, 0, -11, -9, 0],
        [-5 , -8, -2, -10, -3, 4, -1, 6, -5, -9, 3, -4, -4, -5, 0, -6, 8, -9, -5, -5, 0],
        [-6 , 1, -3, -13, -7, -6, -5, -12, -15, 0, -16, 0, 0, -1, -1, 0, -9, 3, -10, -11, 0],
        [0  , -7, -2, -6, 4, -4, -5, -5, 5, -8, -6, -8, -4, -2, -6, -11, -5, -10, 8, -6, 0],
        [-2 , -9, -2, -3, -4, -2, 3, -8, -8, -9, -9, -5, -5, -5, -3, -9, -5, -11, -6, 9, 0],
        [0  , 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.int8)

    score = needleman_wunsch_gotoh_score_opm(
        target, query,
        substitution_alphabet=alphabet,
        substitution_matrix=substitutions,
        gap_opening=-10,
        gap_extension=-1)


    return score



def smith_waterman_gotoh_alignment(
        str1: str,
        str2: str,
        substitution_alphabet: Optional[str] = None,
        substitution_matrix: Optional[np.ndarray] = None,
        gap_opening: Optional[int] = None,
        gap_extension: Optional[int] = None,
        match: Optional[int] = None,
        mismatch: Optional[int] = None,
) -> Tuple[str, str, int]:
    substitution_alphabet, substitution_matrix, gap_opening, gap_extension = _validate_gotoh_arguments(
        substitution_alphabet=substitution_alphabet,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
        match=match,
        mismatch=mismatch,
    )

    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)
    scores, changes, max_pos = _smith_waterman_gotoh_kernel(
        seq1,
        seq2,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
    )

    align1, align2 = _reconstruct_alignment(
        changes[:max_pos[0] + 1, :max_pos[1] + 1],
        seq1[:max_pos[0]],
        seq2[:max_pos[1]],
        substitution_alphabet
    )
    return align1, align2, int(scores[max_pos])


def smith_waterman_gotoh_score(
        str1: str,
        str2: str,
        substitution_alphabet: Optional[str] = None,
        substitution_matrix: Optional[np.ndarray] = None,
        gap_opening: Optional[int] = None,
        gap_extension: Optional[int] = None,
        match: Optional[int] = None,
        mismatch: Optional[int] = None,
) -> int:
    substitution_alphabet, substitution_matrix, gap_opening, gap_extension = _validate_gotoh_arguments(
        substitution_alphabet=substitution_alphabet,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
        match=match,
        mismatch=mismatch,
    )

    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)

    score = smith_waterman_gotoh_score_kernel(
        seq1,
        seq2,
        substitution_matrix=substitution_matrix,
        gap_opening=gap_opening,
        gap_extension=gap_extension,
    )

    return int(score)


def _translate_sequence_optimized(seq: str, alphabet: str) -> np.ndarray:
    """
    Optimized translation function using precomputed mapping.
    """
    return _translate_sequence(seq, alphabet)


def colorize_alignment(align1: str, align2: str, background: str = "dark") -> Tuple[str, str]:
    if background not in ("dark", "light"):
        raise ValueError("Background must be either 'dark' or 'light'")

    # Define color schemes
    if background == "dark":
        match_color = Fore.GREEN
        mismatch_color = Fore.RED
        gap_color = Fore.WHITE
    else:
        match_color = Fore.GREEN
        mismatch_color = Fore.RED
        gap_color = Fore.BLACK

    colored_align1 = []
    colored_align2 = []

    for a, b in zip(align1, align2):
        if a == b and a != "-":
            colored_align1.append(match_color + a + Style.RESET_ALL)
            colored_align2.append(match_color + b + Style.RESET_ALL)
        elif a == "-" or b == "-":
            colored_align1.append(gap_color + a + Style.RESET_ALL)
            colored_align2.append(gap_color + b + Style.RESET_ALL)
        else:
            colored_align1.append(mismatch_color + a + Style.RESET_ALL)
            colored_align2.append(mismatch_color + b + Style.RESET_ALL)

    return ''.join(colored_align1), ''.join(colored_align2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Affine Gaps alignment CLI utility")
    parser.add_argument(
        "seq1",
        type=str,
        help="The first sequence to be aligned, like insulin (GIVEQCCTSICSLYQLENYCN)",
    )
    parser.add_argument(
        "seq2",
        type=str,
        help="The second sequence to be aligned, like glucagon (HSQGTFTSDYSKYLDSRAEQDFV)",
    )
    parser.add_argument(
        "--match",
        type=int,
        default=None,
        help="The score for a match, to compose the substitution matrix; uses scaled BLOSUM62 by default",
    )
    parser.add_argument(
        "--mismatch",
        type=int,
        default=None,
        help="The score for a mismatch, to compose the substitution matrix; uses scaled BLOSUM62 by default",
    )
    parser.add_argument(
        "--gap-opening",
        type=int,
        default=None,
        help=f"The penalty for opening a gap; uses {default_gap_opening} by default",
    )
    parser.add_argument(
        "--gap-extension",
        type=int,
        default=None,
        help=f"The penalty for extending a gap; uses {default_gap_extension} by default",
    )
    parser.add_argument(
        "--substitution-alphabet",
        type=str,
        default=None,
        help="The substitution alphabet used for scoring.",
    )
    parser.add_argument(
        "--substitution-matrix",
        type=str,
        default=None,
        help="The substitution matrix as a flattened list, separated by commas.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use the Smith-Waterman algorithm for local alignment instead of Needleman-Wunsch",
    )
    args = parser.parse_args()

    try:
        if args.substitution_matrix is not None:
            # Ensure that substitution_alphabet is provided when substitution_matrix is given
            if args.substitution_alphabet is None:
                raise ValueError("Substitution alphabet must be provided when substitution matrix is given.")
            substitution_matrix = np.array(
                [int(x) for x in args.substitution_matrix.split(",")],
                dtype=np.int8
            )
            alphabet_length = len(args.substitution_alphabet)
            if substitution_matrix.size != alphabet_length * alphabet_length:
                raise ValueError(
                    f"Substitution matrix size {substitution_matrix.size} does not match alphabet size {alphabet_length}.")
            substitution_matrix = substitution_matrix.reshape((alphabet_length, alphabet_length))
        else:
            substitution_matrix = None

        if args.local:
            aligner = smith_waterman_gotoh_alignment
        else:
            aligner = needleman_wunsch_gotoh_alignment

        align1, align2, score = aligner(
            args.seq1,
            args.seq2,
            substitution_alphabet=args.substitution_alphabet,
            substitution_matrix=substitution_matrix,
            gap_opening=args.gap_opening,
            gap_extension=args.gap_extension,
            match=args.match,
            mismatch=args.mismatch,
        )
    except Exception as exc:
        print("Error:", exc)
        exit(1)

    colored1, colored2 = colorize_alignment(align1, align2)
    print()
    print("Sequence 1:", args.seq1)
    print("Sequence 2:", args.seq2)
    print()
    print("Alignment 1:", colored1)
    print("Alignment 2:", colored2)
    print("Score:      ", score)


def saligner(target, query):
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    substitutions = np.array([
        [6  , -3 , 1, 2, 3, -2, -2, -7, -3, -3, -10, -5, -1, 1, -4, -7, -5, -6, 0, -2, 0],
        [-3 , 6  , -2, -8, -5, -4, -4, -12, -13, 1, -14, 0, 0, 1, -1, 0, -8, 1, -7, -9, 0],
        [1  , -2 , 4, -3, 0, 1, 1, -3, -5, -4, -5, -2, 1, -1, -1, -4, -2, -3, -2, -2, 0],
        [2  , -8 , -3, 9, -2, -7, -4, -12, -10, -7, -17, -8, -6, -3, -8, -10, -10, -13, -6, -3, 0],
        [3  , -5 , 0, -2, 7, -3, -3, -5, 1, -3, -9, -5, -2, 2, -5, -8, -3, -7, 4, -4, 0],
        [-2 , -4 , 1, -7, -3, 6, 3, 0, -7, -7, -1, -2, -2, -4, 3, -3, 4, -6, -4, -2, 0],
        [-2 , -4 , 1, -4, -3, 3, 6, -4, -7, -6, -6, 0, -1, -3, 1, -3, -1, -5, -5, 3, 0],
        [-7 , -12, -3, -12, -5, 0, -4, 8, -5, -11, 7, -7, -6, -6, -3, -9, 6, -12, -5, -8, 0],
        [-3 , -13, -5, -10, 1, -7, -7, -5, 9, -11, -8, -12, -6, -5, -9, -14, -5, -15, 5, -8, 0],
        [-3 , 1, -4, -7, -3, -7, -6, -11, -11, 6, -16, -3, -2, 2, -4, -4, -9, 0, -8, -9, 0],
        [10 , -14, -5, -17, -9, -1, -6, 7, -8, -16, 10, -9, -9, -10, -5, -10, 3, -16, -6, -9, 0],
        [-5 , 0, -2, -8, -5, -2, 0, -7, -12, -3, -9, 7, 0, -2, 2, 3, -4, 0, -8, -5, 0],
        [-1 , 0, 1, -6, -2, -2, -1, -6, -6, -2, -9, 0, 4, 0, 0, -2, -4, 0, -4, -5, 0],
        [1  , 1, -1, -3, 2, -4, -3, -6, -5, 2, -10, -2, 0, 5, -2, -4, -5, -1, -2, -5, 0],
        [-4 , -1, -1, -8, -5, 3, 1, -3, -9, -4, -5, 2, 0, -2, 6, 2, 0, -1, -6, -3, 0],
        [-7 , 0, -4, -10, -8, -3, -3, -9, -14, -4, -10, 3, -2, -4, 2, 6, -6, 0, -11, -9, 0],
        [-5 , -8, -2, -10, -3, 4, -1, 6, -5, -9, 3, -4, -4, -5, 0, -6, 8, -9, -5, -5, 0],
        [-6 , 1, -3, -13, -7, -6, -5, -12, -15, 0, -16, 0, 0, -1, -1, 0, -9, 3, -10, -11, 0],
        [0  , -7, -2, -6, 4, -4, -5, -5, 5, -8, -6, -8, -4, -2, -6, -11, -5, -10, 8, -6, 0],
        [-2 , -9, -2, -3, -4, -2, 3, -8, -8, -9, -9, -5, -5, -5, -3, -9, -5, -11, -6, 9, 0],
        [0  , 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.int8)

    score = needleman_wunsch_gotoh_score_opm(
        target, query,
        substitution_alphabet=alphabet,
        substitution_matrix=substitutions,
        gap_opening=-10,
        gap_extension=-1)


    return score


if __name__ == "__main__":
    main()

