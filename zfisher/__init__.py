import logging

# Configure the zfisher logger with its own console handler and disable
# propagation to the root logger.  napari installs a notification handler
# on the root logger that converts every log record into a GUI toast
# popup — by cutting propagation we prevent all zfisher log messages
# from triggering those popups while still printing to the console.
_zf_logger = logging.getLogger("zfisher")
_zf_logger.setLevel(logging.INFO)
_zf_logger.propagate = False

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(name)s [%(levelname)s] %(message)s"))
_zf_logger.addHandler(_console_handler)
