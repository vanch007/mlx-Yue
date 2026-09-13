"""Compatibility entry point; canonical implementation is lyra.music_tools.abc_tools."""
from lyra.music_tools.abc_tools import *  # noqa: F403

if __name__ == "__main__":
    import sys
    from lyra.music_tools.abc_tools import main
    sys.exit(main())
