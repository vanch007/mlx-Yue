def sliding_window_plan(duration, window_seconds=300.0, overlap_seconds=200.0, lookahead_seconds=100.0):
    if not duration > 0 or not window_seconds > 0:
        raise ValueError("Duration and window length must be positive")
    if not 0 <= lookahead_seconds <= overlap_seconds < window_seconds:
        raise ValueError("Require 0 <= lookahead <= overlap < window length")
    hop = window_seconds - overlap_seconds
    start, accepted = 0.0, 0.0
    result = []
    while True:
        last = start + window_seconds >= duration - 1e-6
        accept_end = duration if last else start + window_seconds - lookahead_seconds
        result.append(dict(start=start, end=min(duration, start + window_seconds),
                           accept_start=accepted, accept_end=accept_end, prefix_end=accepted,
                           generation_stop=None if last else window_seconds - lookahead_seconds))
        if last:
            return result
        accepted = accept_end
        start = min(start + hop, duration - window_seconds)
