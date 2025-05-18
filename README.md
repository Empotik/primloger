# Primloger

**Primloger** (Printer + Image + Logger) is a lightweight, terminal-friendly logging utility that combines styled text logging with optional image preview in the terminal using `img2sixel`. Now that ChatGPT is done talking, now its my turn [insert a B-word here. No like here not a D-word in your mom please?]. Well this started from like "you know what, i hate re importing logger lib again and agian and i has to set it up and stuff." So that's the initial version of this which called `logger.py`. I used it for Capstone as well but don't realized the need to update it just yet. Now that Im graduated, YAY. Means I have stupid amount of time until gradschool. So I come back to do my originals. I've been using my `logger.py` in a few of my projects. Recently, there were a increasing need for img2sixel. Idk, I just liked it I guess. So I also slaped it in here as well. I also spent stupid amount of time finetuning CLI so that it'd print something nice out. Now F that, I have this thing that would do everything I'd ever needed. Henze this project. I asked ChatGPT if this was a stupid idea. Like you know, I though, "well people should already figured this out already. So WTF im doing?" But it said back to me that this project is kinda unique (and stupid, now that I think about this). Thus I opensource this. No thank you, you're welcome. But if sh*t break thats not on me okay? You can def hit me up as ask like Ayo, WTF?!. Or like can we add this? I guess, maybe? I suppose? I presume?


## 🔧 Features

- ✅ Clean, colored logging output
- ✅ Terminal-aware layout: boxed, centered, and indented printing
- ✅ Unified interface for both `print()` and `logging`
- ✅ `showimg()` to preview images in the terminal
- ✅ Fallback to ASCII art preview when `img2sixel` is unavailable


## 🖼 Image Display

### Image Modes Supported
- File path (`.jpg`, `.png`, etc.)
- OpenCV `np.ndarray` (BGR format)
- `PIL.Image.Image`

### Terminal Rendering
If `img2sixel` is installed:
- Image is resized to fit current terminal dimensions
- Printed inline via SIXEL protocol

If not:
- Grayscale ASCII art is printed as a fallback


## 📦 Installation

```bash
pip install pillow opencv-python
sudo apt install libsixel-bin   # For img2sixel
```


## 🧪 Example Usage

```python
from primloger import get_primlog
import cv2

plim = get_primlog(
    print_level="INFO", 
    log_level="WARNING")

plim.boxed("Welcome to Primloger!")
plim.success("Everything is working fine.")
plim.warn("Just a test warning.")
plim.info("This will only print, not log.")
plim.debug(f"this wont print nor log nothing: {val1}")

# Show image (from file or numpy)
plim.showimg("example.jpg", caption="Sample Image")
# plim.showimg(cv2.imread("frame.jpg"))
```


## 📘 Notes

* Terminal size is estimated in **characters**, and scaled assuming \~8x16 pixels per cell.
* ASCII fallback is used when `img2sixel` is not installed or the terminal doesn't support it.


## 🧼 Roadmap Ideas

* [ ] Optional timestamped logging
* [ ] Inline frame updates (like GIFs or camera preview)
* [ ] Color-coded ASCII fallback
* [ ] Log to file