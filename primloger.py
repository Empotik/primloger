import logging
import platform
from pathlib import Path
import sys
import shutil
import re
import numpy as np
from PIL import Image
import subprocess
import io
import cv2
import os
import datetime

ANSI_COLORS = {
    # Foreground
    **{k: f"\033[{30+i}m" for i, k in enumerate([
        'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
    ])},
    # Bright foregrounds
    **{f"bright_{k}": f"\033[{90+i}m" for i, k in enumerate([
        'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
    ])},
    # Backgrounds (extend as needed)
    'bg_red': '\033[41m',
    'bg_green': '\033[42m',

    # Styles
    'bold': '\033[1m',
    'dim': '\033[2m',
    'italic': '\033[3m',
    'underline': '\033[4m',
    'reverse': '\033[7m',
    'strike': '\033[9m',

    # Reset
    'reset': '\033[0m',
}

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "success": logging.INFO,
}

LEVEL_ICONS = {
    "debug": "🐛", 
    "info": "ℹ️", 
    "success": "✅",
    "warn": "⚠️", 
    "error": "❌", 
    "critical": "🔥",
}

class ColorFormatter(logging.Formatter):
    def __init__(self, use_color=True, **kwargs):
        super().__init__(**kwargs)
        self.use_color = use_color

    def format(self, record):
        msg = super().format(record)
        color = ANSI_COLORS.get(record.levelname.lower(), ANSI_COLORS["reset"])
        icon = LEVEL_ICONS.get(record.levelname.lower(), "")
        timestamp = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        output = f"[{timestamp}] [{record.levelname.upper()}] {icon} {msg}"
        return f"{color}{output}{ANSI_COLORS['reset']}" if self.use_color else output
    
def get_logger(name: str = None) -> logging.Logger:
    """
    ⚠️ DEPRECATED: Migrate to `get_primlog()` for combined printing + logging.
    """
    print("⚠️ [DEPRECATED] `get_logger()` is deprecated. Migrate to `get_primlog()`.", file=sys.stderr)
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

def get_primlog(
    print_level="info",
    log_level="info",
    logger=None,
    use_color=True,
    use_icon=True,
    log_file=None,
    log_name="plog",
    log_mode=None
):
    """
    Parameters:
    - log_mode:
        - "none"     → disables logging entirely
        - "journal"  → only logs to journalctl
        - "file"     → only logs to file
        - "both"     → logs to both journalctl and file
        - None       → default, acts like "journal"
    """

    if logger is None:
        mode = (log_mode or "journal").lower()

        if mode == "none":
            logger = _get_disabled_logger(log_name)

        elif mode == "journal":
            if _is_journal_available():
                logger = _get_journal_only_logger(log_name)
            else:
                print("⚠️ Journald is not available. Logging is disabled. Please configure logging behaviour or your system.")
                logger = _get_disabled_logger(log_name)

        elif mode == "file":
            logger = _get_file_logger(log_name, log_file)

        elif mode == "both":
            logger = _get_combined_logger(log_name, log_file)

        else:
            raise ValueError(f"Invalid log_mode: {log_mode}")

    return Plim(logger=logger, print_level=print_level, log_level=log_level, use_color=use_color, use_icon=use_icon)

class Plim:
    COLORS = ANSI_COLORS
    BOX_CHARS = {"tl": "┌", "tr": "┐", "bl": "└", "br": "┘", "h": "─", "v": "│"}

    def __init__(self, logger=None, print_level="info", log_level="info", use_color=True, use_icon=True):
        self.logger = logger or _get_silent_logger("plim")
        self.print_level = LEVELS.get(print_level.lower(), logging.INFO)
        self.log_level = LEVELS.get(log_level.lower(), logging.INFO)
        self.use_color = use_color and sys.stdout.isatty()
        self.use_icon = use_icon and sys.stdout.isatty()
        self.width = shutil.get_terminal_size((80, 20)).columns
        self.stream = sys.stdout

    def _apply(self, msg: str, color_key: str | None = None) -> str:
        if not self.use_color or not color_key:
            return msg

        color  = self.COLORS.get(color_key, "")
        reset  = self.COLORS["reset"]

        # reuse your highlighter
        msg = self._highlight(msg, parent_style=color_key)
        return f"{color}{msg}{reset}"

    def _highlight(self, msg, parent_style="info"):
        """Highlight numeric and structured patterns within the message."""
        if not self.use_color:
            return msg

        from collections import namedtuple
        Highlight = namedtuple("Highlight", ["start", "end", "color"])

        patterns = [
            (r"\b\d+(\.\d+)?\b", "bright_green"),   # Numbers
            (r'"[^"]*"|\'[^\']*\'', "dim"),         # Quoted strings
            (r"\[[^\]]*\]", "cyan"),                # [options]
            (r"\{[^}]*\}", "red"),                  # {dicts}
            (r"\([^)]+\)", "bright_yellow"),        # (default)
        ]

        matches = []
        for pattern, color_key in patterns:
            for m in re.finditer(pattern, msg):
                matches.append(Highlight(m.start(), m.end(), self.COLORS[color_key]))

        # Remove overlaps — keep only the outermost (earliest)
        matches.sort(key=lambda h: h.start)
        filtered = []
        last_end = -1
        for h in matches:
            if h.start >= last_end:
                filtered.append(h)
                last_end = h.end

        # Apply highlights in reverse to not mess up positions
        reset = self.COLORS["reset"]
        parent_color = self.COLORS.get(parent_style, "")
        for h in reversed(filtered):
            colored = f"{h.color}{msg[h.start:h.end]}{reset}{parent_color}"
            msg = msg[:h.start] + colored + msg[h.end:]

        return msg

    def _split_style(self, style: str) -> tuple[str, str, int]:
        """
        Returns (color_key, name_key, numeric_level)
          - style may be "bright_green+success" or just "warning"
          - name_key (right part) drives icons *and* log level
        """
        if "+" in style:
            color_key, name_key = style.split("+", 1)
        else:
            color_key = name_key = style
        level = LEVELS.get(name_key.lower(), logging.INFO)
        return color_key, name_key, level
    
    # Essentially print + log if the level is high enough
    def _emit(self, msg: str, style: str) -> None:
        self._print(msg, style)
        self._log(msg, style)

    def _print(self, msg: str, style: str, end=None, flush: bool = False) -> None:
        color_key, name_key, level = self._split_style(style)
        if level < self.print_level:
            return                                    # below console threshold

        icon = LEVEL_ICONS.get(name_key, "") if self.use_icon else ""
        if icon:
            msg = f"{icon} {msg}"

        print(self._apply(msg, color_key), end=end, file=self.stream, flush=flush) if end else print(self._apply(msg, color_key), file=self.stream, flush=flush)

    def _log(self, msg: str, style: str) -> None:
        if not self.logger:
            return                                    # silent logger

        _, name_key, level = self._split_style(style)
        if level < self.log_level:
            return                                    # below file threshold

        self.logger.log(level, msg)

    def __call__(self, msg=""): self.plain(msg)

    # ── public methods  ──────────────────────────────────
    def plain(self, msg: str = ""):
        """Print & log a plain line (INFO level)."""
        self._emit(msg, "reset+info")

    def success(self, msg: str):
        """Bright‑green ✅ success message (INFO level)."""
        self._emit(msg, "bright_green+success")

    def warn(self, msg: str):
        """Yellow ⚠️ warning (WARN level)."""
        self._emit(msg, "yellow+warn")

    def error(self, msg: str):
        """Bright‑red ❌ error (ERROR level)."""
        self._emit(msg, "bright_red+error")

    def debug(self, msg: str):
        """Grey 🐛 debug line (DEBUG level)."""
        self._emit(msg, "bright_black+debug")

    def info(self, msg: str):
        """Blue ℹ️ info line (INFO level)."""
        self._emit(msg, "bright_blue+info")

    def critical(self, msg: str):
        """Red 🔥 critical line (CRITICAL level)."""
        self._emit(msg, "red+critical")

    def bold(self, msg: str):
        """Bold text (DEBUG level)."""
        self._emit(msg, "bold+info")

    def dim(self, msg: str):
        """Dim / secondary text (DEBUG level)."""
        self._emit(msg, "dim+info")

    def center(self, msg: str):
        """Center a line in the terminal width, log at (DEBUG level)."""
        self._emit(msg.center(self.width), "reset+info")

    def right(self, msg: str):
        """Right‑align a line, log at (DEBUG level)."""
        self._emit(msg.rjust(self.width), "reset+info")

    def divider(self, char: str = "─"):
        """Full‑width horizontal rule, logged at (DEBUG level)."""
        self._emit(char * self.width, "dim+info")

    def indent(self, msg: str, level: int = 1):
        """Indent message by `level`×2 spaces, log at (DEBUG level)."""
        self._emit("  " * level + msg, "reset+info")

    def box(self, msg: str, style: str = "dim+info"):
        """
        Draw a single‑line or multi‑line message inside an ASCII box. (DEBUG level).

        Parameters
        ----------
        msg : str
            Text to wrap (may contain line breaks).
        style : str
            Style passed to `_emit()` for each box line.
            Default is dim colour + DEBUG log level.
        """
        lines = msg.splitlines() or [""]
        width = min(self.width, max(len(line) for line in lines) + 4)

        tl, tr = self.BOX_CHARS["tl"], self.BOX_CHARS["tr"]
        bl, br = self.BOX_CHARS["bl"], self.BOX_CHARS["br"]
        h,  v  = self.BOX_CHARS["h"],  self.BOX_CHARS["v"]

        self._emit(f"{tl}{h * (width - 2)}{tr}", style)
        for line in lines:
            padded = f"{v} {line.ljust(width - 3)}{v}"
            self._emit(padded, style)
        self._emit(f"{bl}{h * (width - 2)}{br}", style)


    # def ask(self, prompt="", newline=False, style="info", default=None, password=False):
    #     import getpass
    #     suffix = f"[default: {default}] " if default else ""
    #     full_prompt = f"{prompt.strip()} {suffix}▶\n" if newline else f"{prompt.strip()} {suffix}▶ "
    #     rendered = self._apply(full_prompt, style) if self.use_color else full_prompt
    #     print(rendered, end="", file=self.stream, flush=True)
    #     try:
    #         response = getpass.getpass("") if password else input("")
    #         return response.strip() or default
    #     except EOFError:
    #         return default
        
    def ask(
        self,
        prompt: str = "",
        *,
        newline: bool = False,
        style: str = "reset+info",
        default: str | None = None,
        password: bool = False,
    ):
        """
        Ask the user for input and log the result.

        Parameters
        ----------
        prompt : str
            Question to display.
        newline : bool
            If True, place the input cursor on the next line.
        style : str
            Style for the on‑screen prompt (e.g. "info", "bright_blue+info").
        default : str | None
            Value returned if the user replies with empty input or hits EOF.
        password : bool
            Use getpass to hide input; the actual password is never logged.
        log_style : str
            Style (color+level) for the log entry that records the reply.
            Only the *level* part is used when mapping to `logging` levels.
        """
        import getpass

        # Compose the visible prompt
        suffix = f"[default: {default}] " if default else ""
        arrow  = "▶"
        base   = f"{prompt.strip()} {suffix}{arrow}"
        full_prompt = f"{base}\n" if newline else f"{base} "

        rendered = self._apply(full_prompt, style) if self.use_color else full_prompt
        # --- print to terminal ---------------------------------------------------
        print(rendered, end="", file=self.stream, flush=True)
        # self._print(full_prompt, style, "",  True) #this for some reason do \n before user input

        # --- read user input -----------------------------------------------------
        try:
            raw = getpass.getpass("") if password else input("")
        except EOFError:
            self._log("⚠️ User input aborted (EOF)", logging.DEBUG)
            return default

        final = (raw or "").strip() or default

        # --- log the response ----------------------------------------------------
        if password:
            self._log("🔒 User input: (password hidden)", style)
        else:
            self._log(f"📝 User input: '{final}'", style)

        return final

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

            self._ascii_fallback(img, width=80)
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

    def set_logger(self, logger):
        self.logger = logger

    def set_log_path(self, path):
        new_logger = _get_silent_logger(name=self.logger.name if self.logger else "plog", logfile=path)
        self.set_logger(new_logger)
        
    def settings(self):
        return {
            "print_level": self.print_level,
            "log_level": self.log_level,
            "use_color": self.use_color,
            "logger": self.logger.name if self.logger else None,
        }

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
            self.plain(line)

def _get_silent_logger(name="plog", logfile=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = None

    if platform.system() == "Linux":
        try:
            from systemd import journal
            handler = journal.JournalHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
        except ImportError:
            pass

    if handler is None:
        logfile = (
            logfile or
            os.getenv("PLOG_LOGFILE") or
            _default_log_path(name=name)
        )
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        handler = logging.FileHandler(logfile, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger

def _get_journal_only_logger(name="plog"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    if platform.system() == "Linux":
        try:
            from systemd import journal
            handler = journal.JournalHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
            logger.addHandler(handler)
        except ImportError:
            logger.addHandler(logging.NullHandler())
    else:
        logger.addHandler(logging.NullHandler())

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger

def _get_disabled_logger(name="plog"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 10)
        logger.propagate = False
    return logger

def _default_log_path(name):
    system = platform.system()
    if system == "Linux":
        return f"/var/log/{name}.log"
    elif system == "Windows":
        base = os.getenv("PROGRAMDATA") or os.getenv("APPDATA") or "C:\\Logs"
        return str(Path(base) / name / f"{name}.log")
    else:
        return str(Path.home() / f"{name}.log")
    
def _is_journal_available():
    if platform.system() != "Linux":
        return False
    try:
        from systemd import journal
        return True
    except ImportError:
        return False

def _get_file_logger(name="plog", logfile=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logfile = logfile or os.getenv("PLOG_LOGFILE") or _default_log_path()
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    handler = logging.FileHandler(logfile, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger

def _get_combined_logger(name="plog", logfile=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handlers = []

    if _is_journal_available():
        from systemd import journal
        journal_handler = journal.JournalHandler()
        journal_handler.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
        handlers.append(journal_handler)

    logfile = logfile or os.getenv("PLOG_LOGFILE") or _default_log_path()
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    handlers.append(file_handler)

    for h in handlers:
        logger.addHandler(h)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger