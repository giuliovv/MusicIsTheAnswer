"""
Interactive Spatial Audio Demo - Control sound position with keyboard.

Controls:
  ← →  : Move sound left/right
  ↑ ↓  : Move sound closer/farther (volume)
  R    : Reset to center
  Q    : Quit
"""

import sys
import tty
import termios
from spatial_audio import SpatialAudioPlayer


def get_key():
    """Read a single keypress."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Handle arrow keys (escape sequences)
        if ch == '\x1b':
            ch += sys.stdin.read(2)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def draw_position(pos, dist):
    """Draw a simple visualization."""
    width = 40
    center = width // 2
    marker_pos = int((pos + 1) / 2 * (width - 1))

    line = ['-'] * width
    line[center] = '|'
    line[marker_pos] = 'O'

    bar = ''.join(line)

    # Simple distance indicator
    dist_bar = '#' * max(1, int(5 / dist)) + '.' * (5 - max(1, int(5 / dist)))

    print(f"\r  L [{bar}] R  dist={dist:.1f} [{dist_bar}]    ", end="", flush=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python interactive_demo.py <audio_file.wav>")
        return

    audio_file = sys.argv[1]
    print(f"Loading: {audio_file}\n")

    player = SpatialAudioPlayer(audio_file)

    pos = 0.0
    dist = 1.0
    step = 0.1
    dist_step = 0.5

    print("Controls:")
    print("  ← →  : Move sound left/right")
    print("  ↑ ↓  : Move sound closer/farther")
    print("  R    : Reset to center")
    print("  Q    : Quit")
    print()

    player.play(loop=True)
    draw_position(pos, dist)

    try:
        while True:
            key = get_key()

            if key == 'q' or key == 'Q':
                break
            elif key == 'r' or key == 'R':
                pos = 0.0
                dist = 1.0
            elif key == '\x1b[D':  # Left arrow
                pos = max(-1.0, pos - step)
            elif key == '\x1b[C':  # Right arrow
                pos = min(1.0, pos + step)
            elif key == '\x1b[A':  # Up arrow (closer)
                dist = max(0.5, dist - dist_step)
            elif key == '\x1b[B':  # Down arrow (farther)
                dist = min(5.0, dist + dist_step)

            player.set_position(pos)
            player.set_distance(dist)
            draw_position(pos, dist)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nStopping...")
        player.stop()


if __name__ == "__main__":
    main()
