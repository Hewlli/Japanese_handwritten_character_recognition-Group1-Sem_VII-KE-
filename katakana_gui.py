import tkinter as tk
from tkinter import ttk, messagebox, Frame, LabelFrame
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
from katakanajapanese import label as katakana_labels
from hiraganajapanese import label as hiragana_labels

class JapaneseCharacterRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Japanese Character Recognizer")
        
        # Load models
        try:
            self.hiragana_model = load_model("hiragana.h5")
            self.katakana_model = load_model("katakana.h5")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
            self.root.destroy()
            return
        
        # Current mode (hiragana/katakana)
        self.current_mode = "hiragana"
        self.current_labels = hiragana_labels
        
        # Setup GUI
        self.setup_ui()
        
        # Drawing variables
        self.last_x = None
        self.last_y = None
        self.line_width = 15
        self.drawing_color = "black"
        self.bg_color = "white"
        self.image = Image.new("L", (300, 300), self.bg_color)
        self.draw = ImageDraw.Draw(self.image)
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Character Set", padding="10")
        mode_frame.pack(fill=tk.X, pady=5)
        
        self.mode_var = tk.StringVar(value="hiragana")
        ttk.Radiobutton(mode_frame, text="Hiragana", variable=self.mode_var, 
                       value="hiragana", command=self.change_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="Katakana", variable=self.mode_var, 
                       value="katakana", command=self.change_mode).pack(side=tk.LEFT, padx=10)
        
        # Drawing area and controls
        drawing_frame = ttk.Frame(main_frame)
        drawing_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Drawing canvas
        self.canvas = tk.Canvas(drawing_frame, width=300, height=300, bg="white", cursor="cross")
        self.canvas.pack(side=tk.LEFT, padx=10)
        
        # Result display
        result_frame = LabelFrame(drawing_frame, text="Recognition Results", padx=10, pady=10)
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.result_label = ttk.Label(result_frame, text="Draw a character", font=("Helvetica", 24))
        self.result_label.pack(pady=10)
        
        self.confidence_label = ttk.Label(result_frame, text="Confidence: -")
        self.confidence_label.pack()
        
        self.mode_label = ttk.Label(result_frame, text="Current mode: Hiragana", font=("Helvetica", 10))
        self.mode_label.pack(pady=10)
        
        self.top_predictions_label = ttk.Label(result_frame, text="Top predictions will appear here")
        self.top_predictions_label.pack(pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Recognize", command=self.recognize).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
    
    def change_mode(self):
        self.current_mode = self.mode_var.get()
        if self.current_mode == "hiragana":
            self.current_labels = hiragana_labels
            self.mode_label.config(text="Current mode: Hiragana")
        else:
            self.current_labels = katakana_labels
            self.mode_label.config(text="Current mode: Katakana")
        self.clear_canvas()
    
    def paint(self, event):
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, 
                                   width=self.line_width, fill=self.drawing_color,
                                   capstyle=tk.ROUND, smooth=tk.TRUE)
            
            # Draw on PIL image
            self.draw.line([self.last_x, self.last_y, event.x, event.y], 
                         fill="black", width=self.line_width)
            
        self.last_x = event.x
        self.last_y = event.y
    
    def reset(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (300, 300), self.bg_color)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a character")
        self.confidence_label.config(text="Confidence: -")
        self.top_predictions_label.config(text="Top predictions will appear here")
    
    def recognize(self):
        # Process the image to match model input
        img = self.image.resize((48, 48))  # Resize to 48x48
        img = ImageOps.invert(img)  # Invert colors (model expects white on black)
        img_array = np.array(img) / 255.0  # Normalize
        
        # Add channel and batch dimensions
        img_array = img_array.reshape(1, 48, 48, 1)
        
        try:
            # Make prediction based on current mode
            if self.current_mode == "hiragana":
                predictions = self.hiragana_model.predict(img_array)
            else:
                predictions = self.katakana_model.predict(img_array)
                
            predicted_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_index]
            
            # Display results
            self.result_label.config(text=self.current_labels[predicted_index])
            self.confidence_label.config(text=f"Confidence: {confidence:.2%}")
            
            # Show top 5 predictions
            top_indices = np.argsort(predictions[0])[::-1][:5]
            top_predictions = "\n".join(
                [f"{self.current_labels[i]}: {predictions[0][i]:.2%}" for i in top_indices]
            )
            self.top_predictions_label.config(text=top_predictions)
            
        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = JapaneseCharacterRecognizer(root)
    root.mainloop()