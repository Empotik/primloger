import logging
import sys
import shutil
import re
import numpy as np
from PIL import Image
import subprocess
import io
import cv2
import os

# Shared color table
BASE_COLORS = {
    'debug': '\033[94m',
    'info': '\033[96m',
    'success': '\033[92m',
    'warn': '\033[93m',
    'warning': '\033[93m',
    'error': '\033[91m',
    'critical': '\033[95m',
    'bold': '\033[1m',
    'dim': '\033[2m',
    'reset': '\033[0m'
}

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class ColorFormatter(logging.Formatter):
    def __init__(self, *, use_color=True, **kwargs):
        super().__init__(**kwargs)
        self.use_color = use_color

    def format(self, record):
        msg = super().format(record)
        if self.use_color:
            color = BASE_COLORS.get(record.levelname.lower(), BASE_COLORS["reset"])
            return f"{color}{msg}{BASE_COLORS['reset']}"
        else:
            return _strip_ansi(msg)

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

def get_logger(name: str = None) -> logging.Logger:
    """
    ⚠️ DEPRECATED: Migrate to `get_plog()` for combined printing + logging.
    """
    print("⚠️ [DEPRECATED] `get_logger()` is deprecated. Migrate to `get_plog()`.", file=sys.stderr)
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

class Plog:
    COLORS = BASE_COLORS

    BOX_CHARS = {
        "tl": "┌", "tr": "┐", "bl": "└", "br": "┘", "h": "─", "v": "│"
    }

    def __init__(self, logger=None, print_level="info", log_level="info", use_color=True):
        self.logger = logger or get_logger("plog")
        self.print_level = LEVELS.get(print_level.lower(), logging.INFO)
        self.log_level = LEVELS.get(log_level.lower(), logging.INFO)
        self.use_color = use_color and sys.stdout.isatty()
        self.width = shutil.get_terminal_size((80, 20)).columns
        self.stream = sys.stdout

    def _apply(self, msg, style=None):
        if not self.use_color or not style:
            return msg
        return f"{self.COLORS.get(style, '')}{msg}{self.COLORS['reset']}"

    def _should_log(self, level):
        current = self.log_level
        return level >= current if isinstance(current, int) else False

    def __call__(self, msg=""):
        self.plain(msg)

    def plain(self, msg=""):
        print(msg, file=self.stream)

    def success(self, msg):
        self._print(msg, "success")
        if self._should_log(logging.INFO):
            self.logger.info(msg)

    def warn(self, msg):
        self._print(msg, "warn")
        if self._should_log(logging.WARNING):
            self.logger.warning(msg)

    def error(self, msg):
        self._print(msg, "error")
        if self._should_log(logging.ERROR):
            self.logger.error(msg)

    def info(self, msg):
        self._print(msg, "info")
        if self._should_log(logging.INFO):
            self.logger.info(msg)

    def bold(self, msg):
        print(self._apply(msg, "bold"), file=self.stream)

    def dim(self, msg):
        print(self._apply(msg, "dim"), file=self.stream)

    def center(self, msg):
        print(msg.center(self.width), file=self.stream)

    def right(self, msg):
        print(msg.rjust(self.width), file=self.stream)

    def divider(self, char="─"):
        print(char * self.width, file=self.stream)

    def indent(self, msg, level=1):
        print("  " * level + msg, file=self.stream)

    def boxed(self, msg):
        if not self.use_color:
            self._ascii_box(msg)
            return

        w = self.width
        top = self.BOX_CHARS["tl"] + self.BOX_CHARS["h"] * (w - 2) + self.BOX_CHARS["tr"]
        mid = self.BOX_CHARS["v"] + msg.center(w - 2) + self.BOX_CHARS["v"]
        bot = self.BOX_CHARS["bl"] + self.BOX_CHARS["h"] * (w - 2) + self.BOX_CHARS["br"]
        print(top, file=self.stream)
        print(mid, file=self.stream)
        print(bot, file=self.stream)

    def _ascii_box(self, msg):
        w = self.width
        top = "+" + "-" * (w - 2) + "+"
        mid = "|" + msg.center(w - 2) + "|"
        bot = "+" + "-" * (w - 2) + "+"
        print(top, file=self.stream)
        print(mid, file=self.stream)
        print(bot, file=self.stream)

    def _print(self, msg, level_name):
        level = LEVELS.get(level_name.lower(), logging.INFO)
        if level >= self.print_level:
            styled = self._apply(msg, level_name.lower())
            print(styled, file=self.stream)

    def _log(self, msg, level_name):
        level = LEVELS.get(level_name.lower(), logging.INFO)
        if self.logger and level >= self.log_level:
            self.logger.log(level, msg)

    def _emit(self, msg, level_name):
        self._print(msg, level_name)
        self._log(msg, level_name)

    def debug(self, msg): self._emit(msg, "debug")
    def critical(self, msg): self._emit(msg, "critical")

    def showimg(self, img, caption=None, fit_terminal=True, scale=1.0):
        """
        Display an image in the terminal using img2sixel, if available.
        - img: str path, PIL.Image, or np.ndarray (BGR)
        - fit_terminal: whether to resize to match terminal
        - scale: scale multiplier for the resized image
        """
        if not shutil.which("img2sixel"):
            self.warn("🚫 Cannot display image: `img2sixel` not found. Showing ASCII fallback.")
            if isinstance(img, str) and os.path.exists(img):
                img = Image.open(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif not isinstance(img, Image.Image):
                self.warn("⚠️ Unsupported image format for ASCII fallback.")
                return

            _ascii_fallback(img, width=80)
            if caption:
                self.dim(f"[📷] {caption}")
            return

        try:
            # Load image
            if isinstance(img, str) and os.path.exists(img):
                img = Image.open(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif not isinstance(img, Image.Image):
                self.warn("⚠️ Unsupported image format for showimg().")
                return

            # Resize to fit terminal
            if fit_terminal:
                term = shutil.get_terminal_size()
                target_w = int(term.columns * 8 * scale)
                target_h = int(term.lines * 16 * scale)

                # Preserve aspect ratio
                img.thumbnail((target_w, target_h), Image.LANCZOS)

            # Display via img2sixel
            with io.BytesIO() as buf:
                img.save(buf, format="PNG")
                buf.seek(0)
                subprocess.run(["img2sixel"], input=buf.read(), check=True)

            if caption:
                self.dim(f"[📷] {caption}")

        except Exception as e:
            self.warn(f"🚩 Failed to display image: {e}")
        
    def settings(self):
        return {
            "print_level": self.print_level,
            "log_level": self.log_level,
            "use_color": self.use_color,
            "logger": self.logger.name if self.logger else None,
        }

def get_primlog(print_level="info", log_level="info", logger=None, use_color=True):
    return Plog(logger=logger, print_level=print_level, log_level=log_level, use_color=use_color)

def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub('', text)

def _ascii_fallback(self, img, width=80):
    """
    Render an image as ASCII grayscale for fallback display.
    Accepts PIL.Image, resizes and prints to terminal.
    """
    ASCII_CHARS = "@%#*+=-:. "
    img = img.convert("L")  # grayscale

    aspect_ratio = img.height / img.width
    new_height = int(aspect_ratio * width * 0.5)
    img = img.resize((width, new_height))

    pixels = list(img.getdata())
    chars = [ASCII_CHARS[pixel * len(ASCII_CHARS) // 256] for pixel in pixels]

    lines = [
        "".join(chars[i:i+width])
        for i in range(0, len(chars), width)
    ]
    for line in lines:
        print(line, file=self.stream)
