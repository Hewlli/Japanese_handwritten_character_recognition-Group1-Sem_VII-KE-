import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
from katakanajapanese import label as katakana_labels
from hiraganajapanese import label as hiragana_labels
from kanjijapanese import label as kanji_labels
from kuzujapanese import label as kuzushiji_labels

PALETTE = {
    "bg": "#ffffff",
    "panel": "#f2f2f2",
    "canvas_bg": "#e6e6e6",
    "text": "#111111",
    "muted": "#6b6b6b",
    "accent": "#2b2b2b",
    "border": "#a0a0a0",
    "grid": "#9e9e9e",
}

class JapaneseCharacterRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Japanese Character Recognizer")
        self.root.configure(bg=PALETTE["bg"])
        self.root.minsize(880, 500)

        # ttk styling
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=PALETTE["bg"])
        style.configure("Panel.TFrame", background=PALETTE["panel"])
        style.configure("TLabelframe", background=PALETTE["panel"], bordercolor=PALETTE["border"])
        style.configure("TLabelframe.Label", background=PALETTE["panel"], foreground=PALETTE["accent"])
        style.configure("TLabel", background=PALETTE["panel"], foreground=PALETTE["text"])
        style.configure("Title.TLabel", background=PALETTE["panel"], foreground=PALETTE["accent"], font=("Helvetica", 20, "bold"))
        style.configure("Muted.TLabel", background=PALETTE["panel"], foreground=PALETTE["muted"])
        style.configure("TButton", padding=8)
        style.map("TButton", background=[("active", "#dcdcdc")])
        style.configure("Status.TLabel", background=PALETTE["panel"], foreground=PALETTE["muted"], font=("Helvetica", 10))

        # Load models
        try:
            self.hiragana_model = load_model("hiragana.h5")
            self.katakana_model = load_model("katakana.h5")
            self.kanji_model = load_model("kanji.h5")
            self.kuzushiji_model = load_model("kuzushiji.h5")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
            self.root.destroy()
            return

        # Default mode
        self.current_mode = "hiragana"
        self.current_labels = hiragana_labels

        # Language mode
        self.current_language = "en"

        # Drawing settings
        self.canvas_size = 320
        self.model_inputs = {"hiragana": 48, "katakana": 48, "kanji": 48, "kuzushiji": 28}
        self.line_width = 7  # Adjusted for finer Kuzushiji strokes
        self.last_x = None
        self.last_y = None

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

        # Undo/Redo stacks
        self.strokes = []
        self.redo_stack = []

        # Build UI
        self._build_ui()

        # Keybinds
        self.root.bind("<r>", lambda e: self.recognize())
        self.root.bind("<R>", lambda e: self.recognize())
        self.root.bind("<c>", lambda e: self.clear_canvas())
        self.root.bind("<C>", lambda e: self.clear_canvas())
        self.root.bind("<z>", lambda e: self.undo())
        self.root.bind("<Z>", lambda e: self.undo())
        self.root.bind("<y>", lambda e: self.redo())
        self.root.bind("<Y>", lambda e: self.redo())

    # ----------------------------
    # UI
    # ----------------------------
    def _build_ui(self):
        outer = ttk.Frame(self.root, style="TFrame", padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        # Left: Canvas
        left = ttk.Frame(outer, style="Panel.TFrame", padding=12)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvas_wrap = ttk.Frame(left, style="Panel.TFrame")
        canvas_wrap.pack(anchor=tk.CENTER, pady=(0, 8))

        # Grey border
        self.canvas_border = tk.Frame(canvas_wrap, bg=PALETTE["border"], bd=1, relief="flat")
        self.canvas_border.pack()
        self.canvas = tk.Canvas(
            self.canvas_border,
            width=self.canvas_size,
            height=self.canvas_size,
            bg=PALETTE["canvas_bg"],
            highlightthickness=0,
            cursor="tcross"
        )
        self.canvas.pack()
        self._draw_grid()

        # Bind mouse
        self.canvas.bind("<Button-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._end_stroke)

        # Brush & controls
        brush_panel = ttk.Frame(left, style="Panel.TFrame")
        brush_panel.pack(fill=tk.X, pady=(8, 0))

        self.brush_label = ttk.Label(brush_panel, text="Brush Size", style="Muted.TLabel")
        self.brush_label.pack(side=tk.TOP, anchor=tk.W)

        self.brush_scale = ttk.Scale(brush_panel, from_=4, to=12, value=self.line_width, command=self._update_brush)  # Narrowed range
        self.brush_scale.pack(fill=tk.X, padx=(0, 10))

        tick_frame = ttk.Frame(brush_panel, style="Panel.TFrame")
        tick_frame.pack(fill=tk.X)
        for i in range(4, 14, 2):  # Adjusted ticks
            ttk.Label(tick_frame, text=str(i), style="Muted.TLabel").pack(side=tk.LEFT, expand=True)

        btns = ttk.Frame(brush_panel, style="Panel.TFrame")
        btns.pack(fill=tk.X, pady=(8, 0))
        self.recognize_btn = ttk.Button(btns, text="Recognize (R)", command=self.recognize)
        self.recognize_btn.pack(side=tk.LEFT, padx=5)
        self.undo_btn = ttk.Button(btns, text="Undo (Z)", command=self.undo)
        self.undo_btn.pack(side=tk.LEFT, padx=5)
        self.redo_btn = ttk.Button(btns, text="Redo (Y)", command=self.redo)
        self.redo_btn.pack(side=tk.LEFT, padx=5)
        self.clear_btn = ttk.Button(btns, text="Clear (C)", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        lang_frame = ttk.Frame(brush_panel, style="Panel.TFrame")
        lang_frame.pack(fill=tk.X, pady=(10, 0))
        self.lang_label = ttk.Label(lang_frame, text="Language:", style="Muted.TLabel")
        self.lang_label.pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="English")
        self.lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var, values=["English", "日本語"], state="readonly")
        self.lang_combo.pack(side=tk.LEFT, padx=5)
        self.lang_combo.bind("<<ComboboxSelected>>", self.change_language)

        # Right: Controls & Results
        right = ttk.Frame(outer, style="Panel.TFrame", padding=12)
        right.pack(side=tk.LEFT, fill=tk.Y)

        self.title_label = ttk.Label(right, text="Japanese Character Recognizer", style="Title.TLabel")
        self.title_label.pack(anchor=tk.W, pady=(0, 8))

        # Mode selection
        self.mode_box = ttk.LabelFrame(right, text="Character Set", padding=10)
        self.mode_box.pack(fill=tk.X, pady=(4, 8))
        self.mode_var = tk.StringVar(value="hiragana")
        self.hira_rb = ttk.Radiobutton(self.mode_box, text="Hiragana", variable=self.mode_var, value="hiragana", command=self.change_mode)
        self.hira_rb.pack(side=tk.LEFT, padx=8)
        self.kata_rb = ttk.Radiobutton(self.mode_box, text="Katakana", variable=self.mode_var, value="katakana", command=self.change_mode)
        self.kata_rb.pack(side=tk.LEFT, padx=8)
        self.kanji_rb = ttk.Radiobutton(self.mode_box, text="Kanji", variable=self.mode_var, value="kanji", command=self.change_mode)
        self.kanji_rb.pack(side=tk.LEFT, padx=8)
        self.kutsu_rb = ttk.Radiobutton(self.mode_box, text="Kuzushiji", variable=self.mode_var, value="kuzushiji", command=self.change_mode)
        self.kutsu_rb.pack(side=tk.LEFT, padx=8)

        # Results
        self.result_box = ttk.LabelFrame(right, text="Recognition", padding=10)
        self.result_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.result_label = ttk.Label(self.result_box, text="Please draw a character", font=("Helvetica", 24, "bold"))
        self.result_label.pack(anchor=tk.CENTER, pady=(4, 6))

        self.conf_wrap = ttk.Frame(self.result_box, style="Panel.TFrame")
        self.conf_wrap.pack(fill=tk.X, pady=(0, 8))
        self.conf_label = ttk.Label(self.conf_wrap, text="Confidence", style="Muted.TLabel")
        self.conf_label.pack(anchor=tk.W)
        self.conf_bar = ttk.Progressbar(self.conf_wrap, orient="horizontal", mode="determinate", maximum=100, value=0)
        self.conf_bar.pack(fill=tk.X)

        self.mode_label = ttk.Label(self.result_box, text="Current mode: Hiragana", style="Status.TLabel")
        self.mode_label.pack(anchor=tk.W, pady=(4, 2))

        self.auto_var = tk.BooleanVar(value=True)
        self.auto_check = ttk.Checkbutton(self.result_box, text="Auto-recognize on stroke end", variable=self.auto_var)
        self.auto_check.pack(anchor=tk.W, pady=(0, 6))

        table_box = ttk.Frame(self.result_box, style="Panel.TFrame")
        table_box.pack(fill=tk.BOTH, expand=True)
        cols = ("char", "prob")
        self.top_table = ttk.Treeview(table_box, columns=cols, show="headings", height=6)
        self.top_table.heading("char", text="Prediction")
        self.top_table.heading("prob", text="Probability")
        self.top_table.column("char", width=120, anchor=tk.CENTER)
        self.top_table.column("prob", width=100, anchor=tk.CENTER)
        self.top_table.pack(fill=tk.BOTH, expand=True)

    # ----------------------------
    # Language switching
    # ----------------------------
    def change_language(self, event=None):
        lang = self.lang_var.get()
        self.current_language = "en" if lang == "English" else "jp"

        if self.current_language == "en":
            self.root.title("Japanese Character Recognizer")
            self.title_label.config(text="Japanese Character Recognizer")
            self.mode_box.config(text="Character Set")
            self.hira_rb.config(text="Hiragana")
            self.kata_rb.config(text="Katakana")
            self.kanji_rb.config(text="Kanji")
            self.kutsu_rb.config(text="Kuzushiji")
            self.result_box.config(text="Recognition")
            self.result_label.config(text="Please draw a character")
            self.conf_label.config(text="Confidence")
            self.mode_label.config(text=f"Current mode: {self.current_mode.capitalize()}")
            self.auto_check.config(text="Auto-recognize on stroke end")
            self.recognize_btn.config(text="Recognize (R)")
            self.undo_btn.config(text="Undo (Z)")
            self.redo_btn.config(text="Redo (Y)")
            self.clear_btn.config(text="Clear (C)")
            self.brush_label.config(text="Brush Size")
            self.lang_label.config(text="Language:")
            self.top_table.heading("char", text="Prediction")
            self.top_table.heading("prob", text="Probability")
        else:
            self.root.title("日本語文字認識")
            self.title_label.config(text="日本語文字認識")
            self.mode_box.config(text="文字セット")
            self.hira_rb.config(text="ひらがな")
            self.kata_rb.config(text="カタカナ")
            self.kanji_rb.config(text="漢字")
            self.kutsu_rb.config(text="くずし字")
            self.result_box.config(text="認識")
            self.result_label.config(text="文字を描いてください")
            self.conf_label.config(text="信頼度")
            self.mode_label.config(text=f"現在のモード: {self.current_mode}")
            self.auto_check.config(text="筆を離したら自動認識")
            self.recognize_btn.config(text="認識 (R)")
            self.undo_btn.config(text="元に戻す (Z)")
            self.redo_btn.config(text="やり直す (Y)")
            self.clear_btn.config(text="消去 (C)")
            self.brush_label.config(text="ブラシサイズ")
            self.lang_label.config(text="言語:")
            self.top_table.heading("char", text="予測")
            self.top_table.heading("prob", text="確率")

        self._reset_results()

    # ----------------------------
    # Mode
    # ----------------------------
    def change_mode(self):
        self.current_mode = self.mode_var.get()
        if self.current_mode == "hiragana":
            self.current_labels = hiragana_labels
            self.mode_label.config(text="Current mode: Hiragana" if self.current_language=="en" else "現在のモード: ひらがな")
        elif self.current_mode == "katakana":
            self.current_labels = katakana_labels
            self.mode_label.config(text="Current mode: Katakana" if self.current_language=="en" else "現在のモード: カタカナ")
        elif self.current_mode == "kanji":
            self.current_labels = kanji_labels
            self.mode_label.config(text="Current mode: Kanji" if self.current_language=="en" else "現在のモード: 漢字")
        else:
            self.current_labels = kuzushiji_labels
            self.mode_label.config(text="Current mode: Kuzushiji" if self.current_language=="en" else "現在のモード: くずし字")
        self._reset_results()

    # ----------------------------
    # Drawing
    # ----------------------------
    def _start_stroke(self, event):
        self.last_x, self.last_y = event.x, event.y
        self.strokes.append([])
        self.redo_stack.clear()

    def _paint(self, event):
        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = event.x, event.y
            return
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                width=self.line_width, fill="black",
                                capstyle=tk.ROUND, smooth=True)
        self.draw.line([self.last_x, self.last_y, event.x, event.y], fill=0, width=self.line_width)
        self.strokes[-1].append(((self.last_x, self.last_y, event.x, event.y), self.line_width))
        self.last_x, self.last_y = event.x, event.y

    def _end_stroke(self, event):
        self.last_x, self.last_y = None, None
        if self.auto_var.get():
            self.recognize()

    def _update_brush(self, _val):
        try:
            self.line_width = int(float(self.brush_scale.get()))
        except Exception:
            pass

    def undo(self):
        if not self.strokes: return
        stroke = self.strokes.pop()
        self.redo_stack.append(stroke)
        self._redraw_from_strokes()
        self._reset_results()

    def redo(self):
        if not self.redo_stack: return
        stroke = self.redo_stack.pop()
        self.strokes.append(stroke)
        self._redraw_from_strokes()
        self._reset_results()

    def clear_canvas(self):
        self.canvas.delete("all")
        self._draw_grid()
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.strokes.clear()
        self.redo_stack.clear()
        self._reset_results()

    def _draw_grid(self):
        self.canvas.delete("grid_line")
        w = self.canvas_size
        h = self.canvas_size
        for i in range(1, 3):
            x = w * i / 3
            self.canvas.create_line(x, 0, x, h, fill=PALETTE["grid"], dash=(4, 2), width=1, tags="grid_line")
        for i in range(1, 3):
            y = h * i / 3
            self.canvas.create_line(0, y, w, y, fill=PALETTE["grid"], dash=(4, 2), width=1, tags="grid_line")

    def _redraw_from_strokes(self):
        self.canvas.delete("all")
        self._draw_grid()
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        for stroke in self.strokes:
            for seg, width in stroke:
                x1, y1, x2, y2 = seg
                self.canvas.create_line(x1, y1, x2, y2, width=width, fill="black", capstyle=tk.ROUND, smooth=True)
                self.draw.line([x1, y1, x2, y2], fill=0, width=width)

    # ----------------------------
    # Recognition
    # ----------------------------
    def _reset_results(self):
        default_text = "Please draw a character" if self.current_language=="en" else "文字を描いてください"
        self.result_label.config(text=default_text)
        self.conf_bar["value"] = 0
        for i in self.top_table.get_children():
            self.top_table.delete(i)

    def _preprocess_for_model(self):
        img = self.image.copy()
        arr = np.array(img)
        mask = arr < 255
        model_input = self.model_inputs[self.current_mode]
        if not mask.any():
            base = Image.new("L", (model_input, model_input), 255)
            return np.array(base, dtype=np.float32).reshape(1, model_input, model_input, 1) / 255.0
        ys, xs = np.where(mask)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        cropped = img.crop((x1, y1, x2 + 1, y2 + 1))
        w, h = cropped.size
        m = int(0.20 * max(w, h))  # Increased padding for better context
        canvas = Image.new("L", (w + 2*m, h + 2*m), 255)
        canvas.paste(cropped, (m, m))
        side = max(canvas.size)
        square = Image.new("L", (side, side), 255)
        square.paste(canvas, ((side-canvas.size[0])//2, (side-canvas.size[1])//2))
        resized = square.resize((model_input, model_input), Image.LANCZOS)
        inverted = ImageOps.invert(resized)
        x = np.array(inverted, dtype=np.float32) / 255.0
        # Enhance contrast for Kuzushiji
        x = np.clip(x * 1.2, 0, 1)  # Slight contrast boost
        return x.reshape(1, model_input, model_input, 1)

    def recognize(self):
        arr = np.array(self.image)
        mask = arr < 255
        if not mask.any():
            messagebox.showwarning("Warning", "Please write a character" if self.current_language == "en" else "文字を書いてください")
            return

        x = self._preprocess_for_model()
        try:
            if self.current_mode == "hiragana":
                predictions = self.hiragana_model.predict(x, verbose=0)
            elif self.current_mode == "katakana":
                predictions = self.katakana_model.predict(x, verbose=0)
            elif self.current_mode == "kanji":
                predictions = self.kanji_model.predict(x, verbose=0)
            else:
                predictions = self.kuzushiji_model.predict(x, verbose=0)

            probs = predictions[0]
            predicted_index = int(np.argmax(probs))
            confidence = float(probs[predicted_index])

            label = self.current_labels[predicted_index]
            if isinstance(label, (tuple, list)):
                label = label[1] if self.current_language=="en" else label[0]

            self.result_label.config(text=label)
            self.conf_bar["value"] = int(confidence * 100)

            for i in self.top_table.get_children():
                self.top_table.delete(i)
            top_indices = np.argsort(probs)[::-1][:5]
            for i in top_indices:
                label = self.current_labels[int(i)]
                if isinstance(label, (tuple, list)):
                    label = label[1] if self.current_language=="en" else label[0]
                self.top_table.insert("", tk.END, values=(label, f"{probs[int(i)]:.2%}"))

        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = JapaneseCharacterRecognizer(root)
    root.mainloop()