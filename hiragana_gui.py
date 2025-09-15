import tkinter as tk
from tkinter import Canvas, Button, Label, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tensorflow.keras.models import load_model
from hiraganajapanese import label  # Ensure this file exists with hiragana labels

class HiraganaRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Japanese Hiragana Recognizer")
        
        # Load the trained model
        try:
            self.model = load_model("hiragana.h5")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.root.destroy()
            return
        
        # Setup GUI
        self.setup_ui()
        
    def setup_ui(self):
        # Drawing canvas
        self.canvas = Canvas(self.root, width=300, height=300, bg="white", cursor="pencil")
        self.canvas.grid(row=0, column=0, padx=10, pady=10, columnspan=2)
        
        # Labels
        self.instruction = Label(self.root, text="Draw a hiragana character", 
                               font=("Helvetica", 12))
        self.instruction.grid(row=1, column=0, columnspan=2)
        
        self.result = Label(self.root, text="", font=("Helvetica", 14, "bold"),
                          fg="blue")
        self.result.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Buttons
        Button(self.root, text="Recognize", command=self.predict,
              width=10).grid(row=3, column=0, pady=5)
        Button(self.root, text="Clear", command=self.clear,
              width=10).grid(row=3, column=1, pady=5)
        
        # Drawing variables
        self.last_x = None
        self.last_y = None
        self.line_width = 20  # Thicker lines for better recognition
        self.drawing = Image.new("L", (300, 300), 255)  # White background
        self.draw = ImageDraw.Draw(self.drawing)
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
    
    def paint(self, event):
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                  width=self.line_width, fill="black",
                                  capstyle=tk.ROUND, smooth=True)
            # Draw on PIL image
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                         fill=0, width=self.line_width)
        self.last_x = event.x
        self.last_y = event.y
    
    def reset(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear(self):
        self.canvas.delete("all")
        self.drawing = Image.new("L", (300, 300), 255)
        self.draw = ImageDraw.Draw(self.drawing)
        self.result.config(text="")
        self.instruction.config(text="Draw a hiragana character")
    
    def predict(self):
        try:
            # Preprocess image
            img = self.drawing.resize((48, 48))
            img = ImageOps.invert(img)
            
            # Convert to numpy array
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 48, 48, 1)
            
            # Get prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(predictions)
            confidence = np.max(predictions)
            
            # Only show prediction if confident enough
            if confidence < 0.85:  # 85% confidence threshold
                self.result.config(text="Not recognized as hiragana", fg="red")
            else:
                char = label[predicted_idx]
                self.result.config(text=f"Predicted: {char} ({confidence:.1%})", 
                                fg="green")
                
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HiraganaRecognizer(root)
    root.mainloop()