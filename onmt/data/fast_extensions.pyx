# cython: language_level=3

import numpy as np
cimport cython
cimport numpy as np

DTYPE=np.int64
ctypedef np.int64_t DTYPE_t

cdef _oversized(list batch, long new_sent_length, list cur_batch_sizes, long batch_size_words, long batch_size_sents,
                ):

    if len(batch) == 0:
        return 0

    if len(batch) >= batch_size_sents:
        return 1

    if max(max(cur_batch_sizes), new_sent_length) * (len(batch) + 1) > batch_size_words:
        return 1

    return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list fast_batch_allocate(
        np.ndarray[DTYPE_t, ndim=1] indices, np.ndarray[DTYPE_t, ndim=1] lengths,
        np.ndarray[DTYPE_t, ndim=1] src_sizes, np.ndarray[DTYPE_t, ndim=1] tgt_sizes,
        long batch_size_words, long batch_size_sents, long batch_size_multiplier,
        long max_src_len, long max_tgt_len,
        long min_src_len, long min_tgt_len, int cleaning):

    cdef long batch_size = 0
    cdef list batch = []
    cdef list batches = []
    cdef list batch_ = []
    cdef list cur_batch_sizes = []
    cdef long mod_len 
    cdef long i
    cdef long idx
    cdef long sent_length
    cdef long src_size
    cdef long tgt_size
    cdef long current_size
    cdef long scaled_size

    cdef DTYPE_t[:] indices_view = indices

    for i in range(len(indices_view)):

        idx = indices_view[i]

        sent_length = lengths[idx]
        src_size = src_sizes[idx]
        tgt_size = tgt_sizes[idx]

        if cleaning == 1:
            if not (min_src_len < src_size < max_src_len and min_tgt_len < tgt_size < max_tgt_len):
                continue

        oversized = _oversized(batch, sent_length, cur_batch_sizes, batch_size_words, batch_size_sents)

        if oversized:
            current_size = len(batch)
            scaled_size = max(batch_size_multiplier * (current_size // batch_size_multiplier),
                              current_size % batch_size_multiplier)
            batch_ = batch[:scaled_size]
            batches.append(batch_)
            batch = batch[scaled_size:]
            cur_batch_sizes = cur_batch_sizes[scaled_size:]

        batch.append(idx)
        cur_batch_sizes.append(sent_length)

    if len(batch) > 0:
        batches.append(batch)

    return batches



cdef _oversized_frames(list batch, long new_size_frames, long new_size_words,
                       list cur_batch_size_frames, list cur_batch_size_words,
                       long batch_size_frames, long batch_size_words, long batch_size_sents,
                       long cut_off_size, long smallest_batch_size):

    if len(batch) == 0:
        return 0

    if len(batch) >= batch_size_sents:
        return 1
    #
    # if max(max(cur_batch_sizes), new_sent_length) * (len(batch) + 1) > batch_size_words:
    #     return 1
    # check if the current batch is too long
    if max(max(cur_batch_size_frames), new_size_frames) > cut_off_size:
        if len(batch) >= smallest_batch_size:
            return 1

    # try adding the new utterance and check if it's oversized in frame limit?
    if max(max(cur_batch_size_frames), new_size_frames) * (len(batch) + 1) > batch_size_frames:
        return 1

    # try adding the new sentence and check if it's oversized in word limit?
    if max(max(cur_batch_size_words), new_size_words) * (len(batch) + 1) > batch_size_words:
        return 1

    return 0


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list fast_batch_allocate_unbalance(
        np.ndarray[DTYPE_t, ndim=1] indices,
        np.ndarray[DTYPE_t, ndim=1] src_sizes, np.ndarray[DTYPE_t, ndim=1] tgt_sizes,
        long batch_size_frames, long batch_size_words, long batch_size_sents, long batch_size_multiplier,
        long max_src_len, long max_tgt_len,
        long min_src_len, long min_tgt_len, int cleaning,
        long cut_off_size, long smallest_batch_size
):

    cdef long batch_size = 0
    cdef list batch = []
    cdef list batches = []
    cdef list batch_ = []
    cdef list cur_batch_size_words = []
    cdef list cur_batch_size_frames = []
    cdef long mod_len
    cdef long i
    cdef long idx
    cdef long sent_length
    cdef long src_size
    cdef long tgt_size
    cdef long current_size
    cdef long scaled_size

    cdef DTYPE_t[:] indices_view = indices

    for i in range(len(indices_view)):

        idx = indices_view[i]

        src_size = src_sizes[idx]
        tgt_size = tgt_sizes[idx]

        if cleaning == 1:
            if not (min_src_len < src_size < max_src_len and min_tgt_len < tgt_size < max_tgt_len):
                continue

        oversized = _oversized_frames(batch, src_size, tgt_size,
                                      cur_batch_size_frames, cur_batch_size_words,
                                      batch_size_frames, batch_size_words, batch_size_sents,
                                      cut_off_size, smallest_batch_size)

        if oversized:
            current_size = len(batch)
            scaled_size = max(batch_size_multiplier * (current_size // batch_size_multiplier),
                              current_size % batch_size_multiplier)
            batch_ = batch[:scaled_size]
            batches.append(batch_)
            batch = batch[scaled_size:]
            cur_batch_size_words = cur_batch_size_words[scaled_size:]
            cur_batch_size_frames = cur_batch_size_frames[scaled_size:]

        batch.append(idx)
        cur_batch_size_words.append(tgt_size)
        cur_batch_size_frames.append(src_size)

    if len(batch) > 0:
        batches.append(batch)

    return batches