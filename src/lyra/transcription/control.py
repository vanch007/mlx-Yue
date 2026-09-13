"""Cooperative interruption shared by transcription loading and inference."""


def check_cancelled(cancelled):
    # Resource callbacks may raise directly; preserve their original exception.
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled during transcription")
