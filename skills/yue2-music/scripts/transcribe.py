"""Compatibility entry point; canonical implementation is lyra.music_tools.transcribe."""
from lyra.music_tools.transcribe import *  # noqa: F403

if __name__ == "__main__":
    import sys
    from lyra.music_tools.transcribe import main
    sys.exit(main())
