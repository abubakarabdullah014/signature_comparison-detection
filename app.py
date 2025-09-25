import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image, ImageTk
import time

# Load the trained model
def load_model():
    model_path = 'runs/detect/signature_detection_train_improved/weights/best.pt'
    if not os.path.exists(model_path):
        messagebox.showerror("Error", f"Model file not found at {model_path}. Please place best.pt in the correct directory.")
        return None
    return YOLO(model_path)

model = load_model()

# Function to extract all signature patterns and get annotated image
def extract_signature_patterns(image_path, model, conf_threshold=0.2):
    img = cv2.imread(image_path)
    if img is None:
        messagebox.showerror("Error", f"Could not load image at {image_path}")
        return None, None, None, False
    
    results = model.predict(img, conf=conf_threshold, save=False)
    annotated_img = results[0].plot()
    
    patterns = []
    signature_detected = False
    for result in results:
        boxes = result.boxes
        print(f"Number of boxes detected: {len(boxes)} for {image_path}")
        if len(boxes) > 0:
            signature_detected = True
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                signature_crop = img[y1:y2, x1:x2]
                signature_crop = cv2.resize(signature_crop, (224, 224))
                signature_gray = cv2.cvtColor(signature_crop, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(signature_gray, 100, 200)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    patterns.append((signature_crop, contours))
    if not patterns:
        messagebox.showwarning("Warning", f"No signature patterns detected in {image_path}")
    return patterns, annotated_img, annotated_img, signature_detected

# Function to compare signature patterns
def compare_signature_patterns(patterns1, patterns2):
    if not patterns1 or not patterns2:
        messagebox.showerror("Error", "Comparison failed: One or both images do not contain detectable signature patterns.")
        return None, None
    
    min_similarity = float('inf')
    are_similar = False
    for crop1, contours1 in patterns1:
        for crop2, contours2 in patterns2:
            if contours1 and contours2:
                similarity = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I3, 0.0)
                print(f"Pattern similarity score: {similarity:.4f}")
                min_similarity = min(min_similarity, similarity)
                if similarity < 0.1:
                    are_similar = True
    return min_similarity, are_similar

# GUI Setup
root = tk.Tk()


# Load and set initial background image
bg_image_path = r"C:\Users\Abu Bakar Abdullah\Desktop\content\runs\theme.png"
bg_photo = None
last_update = 0
UPDATE_DELAY = 0.1  # Debounce delay in seconds

def update_background(event=None):
    global bg_photo, last_update
    current_time = time.time()
    if current_time - last_update < UPDATE_DELAY:
        return
    
    try:
        width = root.winfo_width()
        height = root.winfo_height()
        if width <= 1 or height <= 1:  # Skip updates when minimized
            return
        
        bg_image = Image.open(bg_image_path)
        bg_image = bg_image.resize((width, height), Image.Resampling.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)
        
        canvas.config(width=width, height=height)
        canvas.delete("all")
        if bg_photo:
            canvas.create_image(0, 0, anchor=tk.NW, image=bg_photo)
        last_update = current_time
    except Exception as e:
        messagebox.showerror("Error", f"Failed to update background image: {e}")

# Create Canvas for background
canvas = tk.Canvas(root)
canvas.grid(row=0, column=0, rowspan=3, sticky="nsew")

# Initial background update
root.update()
update_background()

# Bind resize event with debouncing
root.bind("<Configure>", update_background)

# Set minimum window size
root.minsize(600, 500)

# Style constants
BUTTON_COLOR = '#6B7280'  # Slate gray
BUTTON_HOVER = '#4B5563'  # Darker gray
SECONDARY_BUTTON_COLOR = '#F59E0B'  # Amber
SECONDARY_HOVER_COLOR = '#D97706'  # Darker amber
TEXT_COLOR = '#FFFFFF'  # White
ALT_TEXT_COLOR = '#1F2937'  # Dark gray
FRAME_BG = '#F3F4F6'  # Off-white
HEADER_BG = '#4B5563'  # Gradient start (simulated)

# Main content frame with fixed layout
content_frame = tk.Frame(root, bg=FRAME_BG, padx=20, pady=20, bd=1, relief="solid")
content_frame.grid(row=1, column=0, sticky="nsew", pady=(50, 0))  # Offset below header

# Header frame with fixed position
header_frame = tk.Frame(root, bg=HEADER_BG, pady=10, height=50)
header_frame.grid(row=0, column=0, sticky="ew")

# Input, compare, and detect frames inside content frame
input_frame = tk.Frame(content_frame, bg=FRAME_BG, pady=10)
input_frame.grid(row=0, column=0, pady=10, sticky="ew")

compare_frame = tk.Frame(content_frame, bg=FRAME_BG, pady=10)
compare_frame.grid(row=1, column=0, pady=10, sticky="ew")

detect_frame = tk.Frame(content_frame, bg=FRAME_BG, pady=10)
detect_frame.grid(row=2, column=0, pady=10, sticky="ew")

# Configure grid weights for responsiveness
root.grid_rowconfigure(0, weight=1)  # Canvas expands
root.grid_rowconfigure(1, weight=0)  # Content frame fixed
root.grid_columnconfigure(0, weight=1)

content_frame.grid_rowconfigure(0, weight=0)
content_frame.grid_rowconfigure(1, weight=0)
content_frame.grid_rowconfigure(2, weight=0)
content_frame.grid_columnconfigure(0, weight=1)

# Header
tk.Label(header_frame, text="Signature Comparison App", font=("Helvetica", 25, "bold"), fg=TEXT_COLOR, bg=HEADER_BG).pack(expand=True)

# Variables
image1_path = tk.StringVar()
image2_path = tk.StringVar()
result_var = tk.StringVar()
image1_display = tk.StringVar()
image2_display = tk.StringVar()
detected1_var = tk.StringVar(value="Not Detected")
detected2_var = tk.StringVar(value="Not Detected")

# Functions for buttons
def select_image1():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if path:
        image1_path.set(path)
        image1_display.set(path)

def select_image2():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if path:
        image2_path.set(path)
        image2_display.set(path)

def compare_images():
    if not model:
        return
    path1 = image1_path.get()
    path2 = image2_path.get()
    if not path1 or not path2:
        messagebox.showerror("Error", "Please select both images.")
        return
    
    patterns1, annotated_img1, _, detected1 = extract_signature_patterns(path1, model)
    patterns2, annotated_img2, _, detected2 = extract_signature_patterns(path2, model)
    
    detected1_var.set("Detected" if detected1 else "Not Detected")
    detected2_var.set("Detected" if detected2 else "Not Detected")
    
    if annotated_img1 is not None and annotated_img2 is not None:
        img1 = Image.fromarray(cv2.cvtColor(annotated_img1, cv2.COLOR_BGR2RGB))
        img1 = img1.resize((300, 300), Image.Resampling.LANCZOS)
        photo1 = ImageTk.PhotoImage(img1)
        image_label1.config(image=photo1)
        image_label1.image = photo1

        img2 = Image.fromarray(cv2.cvtColor(annotated_img2, cv2.COLOR_BGR2RGB))
        img2 = img2.resize((300, 300), Image.Resampling.LANCZOS)
        photo2 = ImageTk.PhotoImage(img2)
        image_label2.config(image=photo2)
        image_label2.image = photo2
    
    if patterns1 and patterns2:
        similarity, are_similar = compare_signature_patterns(patterns1, patterns2)
        if similarity is not None:
            result_var.set(f"Similarity Score: {similarity:.4f}\nResult: {'Same' if are_similar else 'Not Same'}")
        else:
            result_var.set("No similar patterns detected.")
    else:
        result_var.set("One or both images have no detectable signature patterns.")

def view_detected_signatures():
    if not model:
        return
    path1 = image1_path.get()
    path2 = image2_path.get()
    if not path1 or not path2:
        messagebox.showerror("Error", "Please select both images.")
        return
    
    patterns1, _, _, detected1 = extract_signature_patterns(path1, model)
    patterns2, _, _, detected2 = extract_signature_patterns(path2, model)
    
    detect_window = tk.Toplevel(root)
    detect_window.title("Signature Detection Result")
    detect_window.geometry("400x200")
    detect_window.configure(bg=FRAME_BG)
    
    result_text = tk.Text(detect_window, height=5, width=50, font=("Helvetica", 12), bg='white', fg=ALT_TEXT_COLOR)
    result_text.pack(pady=20)
    
    result_text.insert(tk.END, f"Image 1: {'Sign detected' if detected1 else 'No sign detected'}\n")
    result_text.insert(tk.END, f"Image 2: {'Sign detected' if detected2 else 'No sign detected'}\n")
    
    if detected1 and detected2 and patterns1 and patterns2:
        similarity, are_similar = compare_signature_patterns(patterns1, patterns2)
        if similarity is not None:
            result_text.insert(tk.END, f"The signs are {'same' if are_similar else 'different'} (Similarity Score: {similarity:.4f})\n")
    
    result_text.config(state='disabled')

# GUI Elements
tk.Label(input_frame, text="Select First Image:", font=("Helvetica", 14), fg=ALT_TEXT_COLOR, bg=FRAME_BG).grid(row=0, column=0, pady=5, sticky="w")
tk.Entry(input_frame, textvariable=image1_path, width=50, font=("Helvetica", 12), bg='white', fg=ALT_TEXT_COLOR).grid(row=1, column=0, pady=5, sticky="ew")
tk.Button(input_frame, text="Browse", command=select_image1, bg=BUTTON_COLOR, fg=TEXT_COLOR, activebackground=BUTTON_HOVER, font=("Helvetica", 12, "bold"), padx=15, pady=5, highlightthickness=2, highlightbackground=BUTTON_COLOR).grid(row=1, column=1, pady=5, padx=5)

tk.Label(input_frame, text="Select Second Image:", font=("Helvetica", 14), fg=ALT_TEXT_COLOR, bg=FRAME_BG).grid(row=2, column=0, pady=5, sticky="w")
tk.Entry(input_frame, textvariable=image2_path, width=50, font=("Helvetica", 12), bg='white', fg=ALT_TEXT_COLOR).grid(row=3, column=0, pady=5, sticky="ew")
tk.Button(input_frame, text="Browse", command=select_image2, bg=BUTTON_COLOR, fg=TEXT_COLOR, activebackground=BUTTON_HOVER, font=("Helvetica", 12, "bold"), padx=15, pady=5, highlightthickness=2, highlightbackground=BUTTON_COLOR).grid(row=3, column=1, pady=5, padx=5)

tk.Button(compare_frame, text="Compare Signatures", command=compare_images, bg=BUTTON_COLOR, fg=TEXT_COLOR, activebackground=BUTTON_HOVER, font=("Helvetica", 14, "bold"), padx=20, pady=10, highlightthickness=2, highlightbackground=BUTTON_COLOR).grid(row=0, column=0, pady=10, padx=5)
tk.Button(compare_frame, text="View Detected Signatures", command=view_detected_signatures, bg=SECONDARY_BUTTON_COLOR, fg=TEXT_COLOR, activebackground=SECONDARY_HOVER_COLOR, font=("Helvetica", 14, "bold"), padx=20, pady=10, highlightthickness=2, highlightbackground=SECONDARY_BUTTON_COLOR).grid(row=0, column=1, pady=10, padx=5)

image_frame = tk.Frame(compare_frame, bg=FRAME_BG)
image_frame.grid(row=1, column=0, columnspan=2, pady=10)
image_label1 = tk.Label(image_frame, text="Annotated Image 1", bg=FRAME_BG, fg=ALT_TEXT_COLOR, font=("Helvetica", 12))
image_label1.grid(row=0, column=0, padx=10)
image_label2 = tk.Label(image_frame, text="Annotated Image 2", bg=FRAME_BG, fg=ALT_TEXT_COLOR, font=("Helvetica", 12))
image_label2.grid(row=0, column=1, padx=10)

tk.Label(detect_frame, text="Detection Status:", font=("Helvetica", 14, "bold"), fg=ALT_TEXT_COLOR, bg=FRAME_BG).grid(row=0, column=0, pady=5, sticky="w")
tk.Label(detect_frame, textvariable=detected1_var, font=("Helvetica", 12), fg='#d32f2f' if detected1_var.get() == "Not Detected" else '#2e7d32', bg=FRAME_BG).grid(row=0, column=1, pady=5, padx=10)
tk.Label(detect_frame, textvariable=detected2_var, font=("Helvetica", 12), fg='#d32f2f' if detected2_var.get() == "Not Detected" else '#2e7d32', bg=FRAME_BG).grid(row=0, column=2, pady=5, padx=10)

#tk.Label(compare_frame, text="Result:", font=("Helvetica", 14, "bold"), fg=ALT_TEXT_COLOR, bg=FRAME_BG).grid(row=2, column=0, columnspan=2, pady=5, sticky="w")
result_text = tk.Text(compare_frame, height=4, width=50, font=("Helvetica", 12), bg='white', fg=ALT_TEXT_COLOR)
#result_text.grid(row=3, column=0, columnspan=2, pady=10)

result_text.insert(tk.END, result_var.get())
result_text.config(state='disabled')

def update_result_text():
    result_text.config(state='normal')
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, result_var.get())
    result_text.config(state='disabled')

result_var.trace('w', lambda *args: update_result_text())
detected1_var.trace('w', lambda *args: detected1_var.set(detected1_var.get()))
detected2_var.trace('w', lambda *args: detected2_var.set(detected2_var.get()))

# Start the app
if __name__ == "__main__":
    root.mainloop()