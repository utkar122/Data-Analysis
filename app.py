from flask import Flask, render_template, request, jsonify, redirect, url_for
from threading import Thread
import tkinter as tk
from tkinter import ttk
import webbrowser
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
import os

app = Flask(__name__)

# Define a constant for the valid API key
VALID_API_KEY = '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'

class RibbonApp:
    def __init__(self, root, api_key):
        self.root = root
        self.api_key = api_key
        self.root.title("Ribbon Interface")
        self.root.geometry('800x600')
        self.root.configure(bg="black")

        path = os.path.dirname(__file__)
        self.image_path = os.path.join(path, 'B3.PNG')

        style = ttk.Style()
        style.theme_create('custom', parent='alt', settings={
            "TNotebook": {"configure": {"background": "black", "tabmargins": [2, 5, 2, 0]}},
            "TNotebook.Tab": {
                "configure": {"padding": [10, 5], "background": "black", "foreground": "white"},
                "map": {"background": [("selected", "blue")],
                        "foreground": [("selected", "white"), ("active", "white")]}
            }
        })

        style.theme_use("custom")

        self.photos = []
        for _ in range(3):
            image = Image.open(self.image_path)
            photo = ImageTk.PhotoImage(image)
            self.photos.append(photo)

        label_text = "B3 Automation"

        header_frame = tk.Frame(root, bg="black")
        header_frame.pack(side="top", fill="x", pady=10, padx=10)

        image_label = tk.Label(header_frame, image=self.photos[0], bg="black")
        image_label.pack(side="left")

        b3_label = tk.Label(header_frame, text=label_text, font=("Verdana", 30, "bold"), fg="white", bg="black")
        b3_label.pack(side="left", padx=(750, 0))

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(side="top", fill="both", expand=1)

        self.panels = {
            "Data QA": ["QA 1", "QA 2", "QA 3"],
            "Tab": ["Tab 1", "Tab 2", "Tab 3"],
            "Scrubbing": ["Scrubbing 1", "Scrubbing 2", "Scrubbing 3"]
        }

        for panel_name, panel_options in self.panels.items():
            panel = tk.Frame(self.notebook, bg="black")
            self.notebook.add(panel, text=panel_name)

            button_frame = tk.Frame(panel, bg="black")
            button_frame.pack(side="top", fill="x", expand=0)

            for option in panel_options:
                button = tk.Button(button_frame, text=option, command=lambda opt=option: self.perform_action(opt),
                                   font=("Verdana", 12), padx=300, pady=15, bg="black", fg="white")
                button.pack(side="left", padx=15, pady=300)

        quit_button = tk.Button(root, text='Quit', command=self.confirm, bg="black", fg="white")
        quit_button.pack(expand=0)

        button = tk.Button(root, text='Open Webpage', command=lambda: webbrowser.open('http://localhost:5000'))
        button.pack()

    def perform_action(self, action):
        if self.api_key == VALID_API_KEY:
            path1 = os.path.dirname(__file__)
            QA1_Path = os.path.join(path1, 'spss compare_V09.py')
            QA2_Path = os.path.join(path1, 'Spss merge multiple.py')

            if action == "QA 1":
                subprocess.Popen(["python", QA1_Path])
            elif action == "QA 2":
                subprocess.Popen(["python", QA2_Path])
            else:
                messagebox.showinfo("Action", f"Performing {action}...")
        else:
            messagebox.showinfo("Unauthorized", "Unauthorized access!")

    def confirm(self):
        answer = messagebox.askyesno(title='Confirmation', message='Are you sure that you want to quit?')
        if answer:
            self.root.destroy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/action', methods=['POST'])
def action():
    action = request.form.get('action')
    ribbon_app.perform_action(action)
    return "Action performed: " + action

@app.route('/quit')
def quit_app():
    ribbon_app.confirm()
    return redirect(url_for('index'))

def run_flask():
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    root = tk.Tk()
    ribbon_app = RibbonApp(root, VALID_API_KEY)
    root.protocol("WM_DELETE_WINDOW", ribbon_app.confirm)  # Handle closing event
    thread = Thread(target=run_flask)
    thread.start()
    root.mainloop()
