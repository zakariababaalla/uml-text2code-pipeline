#!/usr/bin/env python3

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import ImageGrab, Image, ImageTk
import math
import re 
from uml_extract import preprocess_text, extract_uml_elements
from uml_to_code import generate_python_code_from_dsl, generate_dsl as generate_python_code
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException


class UMLDiagramApp:
    def __init__(self, root):
        """Initialize the graphical interface to display the UML diagram."""
        self.root = root
        self.root.title("UML Class Diagram Generator")
        self.root.geometry("1000x700")

        # Top frame with buttons
        self.top_frame = tk.Frame(self.root)

        # Load and display the logo
        self.logo_image = Image.open("logo.jpg")  
        self.logo_image = self.logo_image.resize((90, 35), Image.Resampling.LANCZOS)  
        self.logo_photo = ImageTk.PhotoImage(self.logo_image)
    
        self.top_frame = tk.Frame(self.root, padx=105)  
        self.top_frame.pack(fill=tk.X, pady=4)

        # Label for the logo (top-right corner)
        self.logo_label = tk.Label(self.root, image=self.logo_photo, bg="white")
        self.logo_label.place(x=10, y=0)  

        # Button to load text file
        self.load_button = tk.Button(self.top_frame, text="Load Text File", command=self.load_file, width=13, height=1, font=("Arial", 11, "bold"), bg="lightblue", fg="white", activebackground="tomato", activeforeground="white")
        self.load_button.pack(side=tk.LEFT, padx=2)

        # Button to analyze text
        self.analyze_button = tk.Button(self.top_frame, text="Analyze Text", command=self.analyze_text, width=13, height=1, font=("Arial", 11, "bold"), bg="lightblue", fg="white", activebackground="tomato", activeforeground="white")
        self.analyze_button.pack(side=tk.LEFT, padx=2)

        # Button to save diagram as PNG
        self.save_button = tk.Button(self.top_frame, text="Save Diagram", command=self.save_diagram, width=13, height=1, font=("Arial", 11, "bold"), bg="lightblue", fg="white", activebackground="tomato", activeforeground="white")
        self.save_button.pack(side=tk.LEFT, padx=2)

        # Button to clear the specification text and canvas
        self.clear_text_button = tk.Button(self.top_frame, text="Clear Text", command=self.clear_text, width=13, height=1, font=("Arial", 11, "bold"), bg="lightblue", fg="white", activebackground="tomato", activeforeground="white")
        self.clear_text_button.pack(side=tk.LEFT, padx=2)

        # Button to quit the application
        self.quit_button = tk.Button(self.top_frame, text="Quit", command=self.quit_application, width=6, height=1, font=("Arial", 11, "bold"), bg="tomato", fg="white", activebackground="lightblue", activeforeground="white")
        self.quit_button.pack(side=tk.RIGHT, padx=2)
        self.quit_button.place(x=1170)

        # Text entry for specifications
        self.text_entry = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=1, font=("times new roman", 13), bg="white", fg="black", padx=15, pady=10)
        self.text_entry.pack(fill=tk.BOTH, expand=True, padx=10)

        # Frame for canvas and scrollbar
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # DSL Editing and Code Display Area (2 Columns)
        self.dsl_frame = tk.LabelFrame(self.root)
        self.dsl_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=1)

        # Frame for DSL and Python code editors side-by-side
        self.dsl_editor_frame = tk.Frame(self.dsl_frame)
        self.dsl_editor_frame.pack(fill=tk.BOTH, expand=True)

        # DSL Editor
        dsl_label = tk.Label(self.dsl_editor_frame, text="Edit Diagram", font=("Arial", 10, "bold"))
        dsl_label.grid(row=0, column=0, sticky="w")

        self.save_dsl_button = tk.Button(self.dsl_editor_frame, text="Save Modification", command=self.analyze_dsl, width=16, height=1, font=("Arial", 9, "bold"), bg="orange", fg="white", activebackground="lightblue", activeforeground="white")
        self.save_dsl_button.grid(row=0, column=0, sticky="e", padx=(0, 24))  

        self.dsl_editor = scrolledtext.ScrolledText(self.dsl_editor_frame, wrap=tk.WORD, height=4, font=("Consolas", 11), bg="#f7f7f7")
        self.dsl_editor.grid(row=1, column=0, sticky="nsew", padx=(0, 5))

        # Python Code Viewer
        code_label = tk.Label(self.dsl_editor_frame, text="Generated Python Code", font=("Arial", 10, "bold"))
        code_label.grid(row=0, column=1, sticky="w")

        self.save_code_button = tk.Button(self.dsl_editor_frame, text="Save Python File", command=self.save_as_python, width=14, height=1, font=("Arial", 9, "bold"), bg="orange", fg="white", activebackground="lightblue", activeforeground="white")
        self.save_code_button.grid(row=0, column=1, sticky="e", padx=(0, 20))

        self.code_viewer = scrolledtext.ScrolledText(self.dsl_editor_frame, wrap=tk.WORD, height=4, font=("Consolas", 11), bg="#f7f7f7", fg="blue", state=tk.DISABLED)
        self.code_viewer.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        
        # Label Validation Report
        report_label = tk.Label(self.dsl_editor_frame, text="Validation Report", font=("Arial", 10, "bold"))
        report_label.grid(row=2, column=0, sticky="w", pady=(5,0))

        # Bouton Save Report
        self.save_report_button = tk.Button(self.dsl_editor_frame, text="Save Report", command=self.save_report, width=14, height=1, font=("Arial", 9, "bold"), bg="orange", fg="white", activebackground="lightblue", activeforeground="white")
        self.save_report_button.grid(row=2, column=1, sticky="e", pady=(5,0), padx=(0,20))

        # Zone Validation Report
        self.report_viewer = scrolledtext.ScrolledText(self.dsl_editor_frame, wrap=tk.WORD, height=2, font=("Consolas", 10), bg="#f7f7f7", fg="red", state=tk.DISABLED)
        self.report_viewer.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=5, pady=(0,5))


        # Configure grid weights
        self.dsl_editor_frame.columnconfigure(0, weight=1)
        self.dsl_editor_frame.columnconfigure(1, weight=1)
        self.dsl_editor_frame.rowconfigure(1, weight=1)

        # Canvas and scrollbar
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", scrollregion=(0, 0, 2000, 2000))
        self.v_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        # Pack canvas and scrollbar
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Store UML data and class positions
        self.class_positions = {}
        self.uml_data = None

    def clear_text(self):
        """Clear the specification text entry field and the UML diagram canvas."""
        if self.text_entry.get(1.0, tk.END).strip() == "":
            messagebox.showinfo("Information", "There is nothing to clear.")
            return

        confirm = messagebox.askyesno("Confirmation", "Are you sure you want to clear the text and diagram?")
        if confirm:
            self.text_entry.delete(1.0, tk.END)
            self.dsl_editor.delete(1.0, tk.END)
            self.code_viewer.config(state=tk.NORMAL)
            self.code_viewer.delete(1.0, tk.END)
            self.code_viewer.config(state=tk.DISABLED)
            self.report_viewer.config(state=tk.NORMAL)
            self.report_viewer.delete(1.0, tk.END)
            self.report_viewer.config(state=tk.DISABLED)
            self.canvas.delete("all")
            self.uml_data = None

    def quit_application(self):
        """Show a confirmation dialog before quitting the application."""
        if messagebox.askyesno("Confirm Exit", "Are you sure you want to quit?"):
            self.root.destroy()  

    def load_file(self):
        """Load a text file and place its content in the text area."""
        self.uml_data = None
        self.canvas.delete("all")
        self.text_entry.delete(1.0, tk.END)  
        self.dsl_editor.delete(1.0, tk.END)  
        self.code_viewer.config(state=tk.NORMAL)
        self.code_viewer.delete(1.0, tk.END)
        self.code_viewer.config(state=tk.DISABLED)
        self.report_viewer.config(state=tk.NORMAL)
        self.report_viewer.delete(1.0, tk.END)
        self.report_viewer.config(state=tk.DISABLED)

        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    self.text_entry.insert(tk.END, content)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read the file: {e}")

    DetectorFactory.seed = 0
    
    def analyze_text(self):
        """Analyze the input text to extract UML elements and update DSL + code view."""
        self.canvas.delete("all")
        input_text = self.text_entry.get(1.0, tk.END).strip()

        if not input_text:
            messagebox.showwarning("Warning", "No text provided for analysis.")
            return

        if not self.is_text_in_english(input_text):
            messagebox.showwarning("Warning", "The text provided is not in English. Please use English text.")
            return

        try:
            clean_text = preprocess_text(input_text)
            self.uml_data = extract_uml_elements(clean_text)
            self.draw_uml_diagram()

            dsl_code = generate_python_code(self.uml_data)
            self.dsl_editor.delete(1.0, tk.END)
            self.dsl_editor.insert(tk.END, dsl_code)

            python_code = generate_python_code_from_dsl(dsl_code)
            self.code_viewer.config(state=tk.NORMAL)
            self.code_viewer.delete(1.0, tk.END)
            self.code_viewer.insert(tk.END, python_code)
            self.code_viewer.config(fg="blue")
            self.code_viewer.config(state=tk.DISABLED)
            self.report_viewer.config(state=tk.NORMAL)
            self.report_viewer.delete(1.0, tk.END)
            self.report_viewer.insert(tk.END, "✅ No anomalies detected.")
            self.report_viewer.config(state=tk.DISABLED)
            self.report_viewer.config(fg="green")

        except Exception as e:
            messagebox.showerror("Error", f"Error during analysis: {e}")

    

    def is_text_in_english(self, text: str) -> bool:
        """Check if the text is in English using langdetect.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if the text is in English, False otherwise.
        """
        try:
            language = detect(text) 
            return language == "en"  
        except LangDetectException:
            return False


    def analyze_dsl(self):
        """Analyze the DSL editor content and regenerate UML diagram and code."""
        dsl_text = self.dsl_editor.get(1.0, tk.END).strip()
        if not dsl_text:
            messagebox.showwarning("Warning", "DSL content is empty.")
            return

        try:
            from uml_to_code import parse_dsl_to_uml
            self.uml_data = parse_dsl_to_uml(dsl_text)
            
            self.report_viewer.config(state=tk.NORMAL)
            self.report_viewer.delete(1.0, tk.END)

            if "validation_report" in self.uml_data and self.uml_data["validation_report"]:
                for entry in self.uml_data["validation_report"]:
                    self.report_viewer.insert(tk.END, f"⚠️ {entry}\n")
                self.report_viewer.config(fg="red")
            else:
                self.report_viewer.insert(tk.END, "✅ No anomalies detected.\n")
                self.report_viewer.config(fg="green")

            self.report_viewer.config(state=tk.DISABLED)

            
            self.canvas.delete("all")
            self.draw_uml_diagram()

            python_code = generate_python_code_from_dsl(dsl_text)
            self.code_viewer.config(state=tk.NORMAL)
            self.code_viewer.delete(1.0, tk.END)
            self.code_viewer.insert(tk.END, python_code)
            self.code_viewer.config(fg="blue")
            self.code_viewer.config(state=tk.DISABLED)
            

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze DSL: {e}")


    def save_report(self):
        """Save the validation report content to a .txt file."""
        content = self.report_viewer.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "The report is empty, nothing to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Report saved successfully at:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {e}")


    def draw_uml_diagram(self):
        """Draw UML classes and relationships on the canvas."""
        self.canvas.delete("all")
        class_x = 25
        class_y = 30
        class_width = 130
        class_height = 135
        class_spacing = 100

        if not self.uml_data or not self.uml_data['classes']:
            messagebox.showwarning("Warning", "No UML classes detected.")
            return

        for cls, details in self.uml_data['classes'].items():
            self.canvas.create_rectangle(class_x, class_y, class_x + class_width, class_y + class_height, fill="lightblue")
            self.canvas.create_text(class_x + class_width / 2, class_y + 20, text=cls, font=("Arial", 12, "bold"))

            attr_y = class_y + 40
            for attr in details['attributes']:
                self.canvas.create_text(class_x + 10, attr_y, anchor="w", text=f"- {attr}", font=("Arial", 10))
                attr_y += 15

            method_y = attr_y + 10
            for method in details['methods']:
                self.canvas.create_text(class_x + 10, method_y, anchor="w", text=f"+ {method}()", font=("Arial", 10))
                method_y += 15

            self.class_positions[cls] = (class_x + class_width / 2, class_y + class_height / 2)
            class_x += class_width + class_spacing
            if class_x > self.root.winfo_width() - class_width:
                class_x = 25
                class_y += class_height + class_spacing

        self.draw_relations()

    def draw_diamond(self, x: int, y: int, filled: bool = False) -> None:
        """Draw a diamond shape on the canvas."""
        size = 9 
        points = [
            x, y - size,  # Top
            x + size, y,  # Right
            x, y + size,  # Bottom
            x - size, y   # Left
        ]
        if filled:
            self.canvas.create_polygon(points, outline="black", fill="black")
        else:
            self.canvas.create_polygon(points, outline="black", fill="white")

    def draw_triangle(self, x1: int, y1: int, x2: int, y2: int):
        """Draw a triangle at the beginning of a line to represent inheritance."""
        arrow_size = 15
        angle = math.atan2(y2 - y1, x2 - x1)

        x_base1 = x1 + arrow_size * math.cos(angle - math.pi / 6)
        y_base1 = y1 + arrow_size * math.sin(angle - math.pi / 6)
        x_base2 = x1 + arrow_size * math.cos(angle + math.pi / 6)
        y_base2 = y1 + arrow_size * math.sin(angle + math.pi / 6)

        self.canvas.create_line(x1, y1, x2, y2)

        self.canvas.create_polygon(
            x1, y1, x_base1, y_base1, x_base2, y_base2,
            fill="white", outline="black"
        )


    def draw_reflexive_arc(self, x: int, y: int, width: int) -> None:
        """Draw a square-shaped reflexive loop for self-referential relationships above a class box.

        Args:
            x (int): X coordinate for the top-left corner of the class.
            y (int): Y coordinate for the top-left corner of the class.
            width (int): Width of the class box.
        """
        loop_size = 20 

        # Start from the top-left corner, draw right, up, left, down
        self.canvas.create_line(x, y, x - loop_size, y, fill="black", width=2)             # Right
        self.canvas.create_line(x - loop_size, y, x - loop_size, y - loop_size, fill="black", width=2)  # Top
        self.canvas.create_line(x - loop_size, y - loop_size, x, y - loop_size, fill="black", width=2)  # Left
        self.canvas.create_line(x, y - loop_size, x, y, fill="black", arrow=tk.LAST, width=2)          # Bottom

    def draw_relations(self):
        """Draw relations between UML classes."""
        vertical_offset = 9  # Offset to slightly adjust the Y-coordinate for visibility

        for index, relation in enumerate(self.uml_data['relations']):
            source = relation['source']
            target = relation['target']
            rel_type = relation['type']

            if source in self.class_positions and target in self.class_positions:
                source_x, source_y = self.class_positions[source]
                target_x, target_y = self.class_positions[target]
                class_width = 130
                class_height = 135

                # Reflexive relationship (self-referencing)
                if source == target:
                    # Draw the square reflexive arc above the class
                    self.draw_reflexive_arc(source_x - class_width / 2, source_y - class_height / 2, class_width)
                    continue

                # Determine line start and end points based on source-target positions
                if source_x < target_x:
                    start_x = source_x + class_width / 2
                    end_x = target_x - class_width / 2
                else:
                    start_x = source_x - class_width / 2
                    end_x = target_x + class_width / 2

                # Apply vertical offset to source_y and target_y for better visibility
                adjusted_source_y = source_y + (index * vertical_offset)
                adjusted_target_y = target_y + (index * vertical_offset)

                # Draw line or inheritance arrow based on relation type
                if rel_type == "inheritance":
                    self.draw_triangle(start_x, adjusted_source_y, end_x, adjusted_target_y)
                else:
                    self.canvas.create_line(start_x, adjusted_source_y, end_x, adjusted_target_y)

                    # Draw diamond for aggregation or composition
                    if rel_type == "aggregation":
                        self.draw_diamond(start_x, adjusted_source_y, filled=False)
                    elif rel_type == "composition":
                        self.draw_diamond(start_x, adjusted_source_y, filled=True)

    def save_diagram(self):
        """Save only the region of the canvas containing the diagram as a PNG image."""
        items = self.canvas.find_all()
        if not items:
            messagebox.showwarning("Warning", "The diagram canvas is empty. Nothing to save.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if not file_path:
            return

        try:
            x_min, y_min, x_max, y_max = float("inf"), float("inf"), float("-inf"), float("-inf")
            for item in items:
                bbox = self.canvas.bbox(item)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    x_min = min(x_min, x1)
                    y_min = min(y_min, y1)
                    x_max = max(x_max, x2)
                    y_max = max(y_max, y2)

            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max += padding
            y_max += padding

            width = int(x_max - x_min)
            height = int(y_max - y_min)
            from PIL import Image, ImageDraw
            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)

            def get_color(color_name):
                """Convert color names to RGB or return None if unavailable."""
                try:
                    self.root.update_idletasks()  # Ensure all colors are resolved
                    if color_name:
                        r, g, b = self.root.winfo_rgb(color_name)
                        return f"#{r // 256:02x}{g // 256:02x}{b // 256:02x}"
                except tk.TclError:
                    return None
                return color_name

            for item in items:
                bbox = self.canvas.bbox(item)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    x1 -= x_min
                    y1 -= y_min
                    x2 -= x_min
                    y2 -= y_min

                    element_type = self.canvas.type(item)
                    fill_color = get_color(self.canvas.itemcget(item, "fill")) if "fill" in self.canvas.itemconfig(item) else None
                    outline_color = get_color(self.canvas.itemcget(item, "outline")) if "outline" in self.canvas.itemconfig(item) else None

                    if element_type == "rectangle":
                        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color)
                    elif element_type == "line":
                        coords = self.canvas.coords(item)
                        adjusted_coords = [(x - x_min, y - y_min) for x, y in zip(coords[::2], coords[1::2])]
                        draw.line(adjusted_coords, fill=outline_color or "black", width=2)
                    elif element_type == "polygon":
                        coords = self.canvas.coords(item)
                        adjusted_coords = [(x - x_min, y - y_min) for x, y in zip(coords[::2], coords[1::2])]
                        draw.polygon(adjusted_coords, fill=fill_color, outline=outline_color)
                    elif element_type == "text":
                        text_coords = self.canvas.coords(item)
                        text = self.canvas.itemcget(item, "text")
                        font_color = fill_color or "black"
                        draw.text((text_coords[0] - x_min, text_coords[1] - y_min), text, fill=font_color)

            image.save(file_path, "PNG")
            messagebox.showinfo("Success", f"Diagram saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save diagram: {e}")



    def save_as_python(self):
        """Save UML elements as Python source code."""
        if not self.uml_data or not self.uml_data['classes']:
            messagebox.showwarning("Warning", "No UML diagram to save. Please analyze text first.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python files", "*.py")])
        if not file_path:
            return

        try:
            uml_classes = self.uml_data['classes']
            relations = self.uml_data['relations']
            code_lines = []

            def clean_name(name: str) -> str:
                """Replace spaces and special characters with underscores in names."""
                return re.sub(r'[^\w]', '_', name).lower()

            for class_name, details in uml_classes.items():
                cleaned_class_name = clean_name(class_name).capitalize()
                code_lines.append(f"class {cleaned_class_name}:")
                attributes = details.get("attributes", [])
                methods = details.get("methods", [])

                if attributes:
                    constructor_line = "    def __init__(self"
                    for attr in attributes:
                        cleaned_attr = clean_name(attr)
                        constructor_line += f", {cleaned_attr}: str = None"
                    constructor_line += "):"
                    code_lines.append(constructor_line)

                    for attr in attributes:
                        cleaned_attr = clean_name(attr)
                        code_lines.append(f"        self.{cleaned_attr} = {cleaned_attr}")
                else:
                    code_lines.append("    def __init__(self):")
                    code_lines.append("        pass")

                for method in methods:
                    cleaned_method = clean_name(method)
                    code_lines.append(f"    def {cleaned_method}(self):")
                    code_lines.append("        pass")
                code_lines.append("")

            for relation in relations:
                source = relation['source']
                target = relation['target']
                rel_type = relation['type']

                cleaned_source = clean_name(source).capitalize()
                cleaned_target = clean_name(target).capitalize()

                if rel_type == "inheritance":
                    code_lines = [
                        line.replace(f"class {cleaned_source}:", f"class {cleaned_source}({cleaned_target}):")
                        for line in code_lines
                    ]
                elif rel_type in {"composition", "aggregation"}:
                    for i, line in enumerate(code_lines):
                        if line.startswith(f"class {cleaned_source}:"):
                            insert_index = i + 1
                            relationship = (
                                f"        self.{clean_name(target)} = {cleaned_target}()  # {rel_type.capitalize()}"
                            )
                            if "def __init__" in code_lines[insert_index]:
                                code_lines.insert(insert_index + 1, relationship)
                            else:
                                code_lines.insert(
                                    insert_index,
                                    f"    def __init__(self):\n        self.{clean_name(target)} = {cleaned_target}()  # {rel_type.capitalize()}",
                                )
                            break

            with open(file_path, 'w') as file:
                file.write("\n".join(code_lines))
            messagebox.showinfo("Success", f"Python code saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save Python code: {e}")


def main():
    root = tk.Tk()
    app = UMLDiagramApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
