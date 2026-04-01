# main.py
import ctypes
try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("zFISHer.App")
except Exception:
    pass

from zfisher.ui.viewer import launch_zfisher


def main():
    launch_zfisher()


if __name__ == "__main__":
    main()