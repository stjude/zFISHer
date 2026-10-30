"""Start the zFISHer app: the ``zfisher`` command, or ``python -m zfisher``."""


def main():
    # On Windows, group zFISHer's windows under its own taskbar icon rather
    # than under python.exe's.
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("zFISHer.App")
    except Exception:
        pass

    from zfisher.ui.viewer import launch_zfisher
    launch_zfisher()


if __name__ == "__main__":
    main()
