"""Compatibility entry point; canonical implementation is lyra.music_tools.run_yue2."""
from lyra.music_tools.run_yue2 import *  # noqa: F403

if __name__ == "__main__":
    import sys
    from lyra.music_tools.run_yue2 import main
    sys.exit(main())
