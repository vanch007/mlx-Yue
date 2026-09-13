"""Compatibility entry point; canonical implementation is lyra.music_tools.listen."""
from lyra.music_tools.listen import *  # noqa: F403

if __name__ == "__main__":
    import sys
    from lyra.music_tools.listen import main
    sys.exit(main())
