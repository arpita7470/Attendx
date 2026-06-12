import tkinter as tk
from tkinter import *
import pandas as pd
import os


def subjectchoose(text_to_speech):
    """Open window to choose subject and view attendance."""

    def view():
        subject = subject_var.get().strip()
        if not subject:
            tk.messagebox.showwarning("Warning", "Please enter subject name!")
            return

        attendance_file = f"./Attendance/{subject}.csv"
        if not os.path.exists(attendance_file):
            tk.messagebox.showinfo("Not Found", f"No attendance sheet found for '{subject}'")
            return

        df = pd.read_csv(attendance_file)
        show_win.destroy()
        showTable(df, subject)

    def listSubjects():
        attendance_dir = "./Attendance"
        if not os.path.exists(attendance_dir):
            subjects_list.config(text="No attendance records found.")
            return
        files = [f.replace(".csv", "") for f in os.listdir(attendance_dir) if f.endswith(".csv")]
        if files:
            subjects_list.config(text="Subjects: " + ", ".join(files))
        else:
            subjects_list.config(text="No subjects found.")

    show_win = tk.Tk()
    show_win.title("View Attendance")
    show_win.geometry("600x320")
    show_win.configure(background="#1c1c1c")
    show_win.resizable(0, 0)

    tk.Label(
        show_win, text="View Attendance Sheet",
        bg="#333333", fg="yellow", font=("Verdana", 20, "bold"),
        width=30, height=2
    ).pack(fill=X)

    tk.Label(
        show_win, text="Subject Name:", bg="#1c1c1c", fg="yellow",
        font=("Verdana", 14)
    ).place(x=60, y=90)

    subject_var = tk.StringVar()
    subject_entry = tk.Entry(
        show_win, width=20, bd=5, bg="#333333", fg="yellow",
        relief=RIDGE, font=("Verdana", 16, "bold"), textvariable=subject_var
    )
    subject_entry.place(x=250, y=90)

    tk.Button(
        show_win, text="View Attendance", command=view,
        bd=5, font=("Verdana", 14, "bold"), bg="#333333", fg="yellow",
        height=1, width=14, relief=RIDGE
    ).place(x=200, y=160)

    tk.Button(
        show_win, text="List All Subjects", command=listSubjects,
        bd=5, font=("Verdana", 12, "bold"), bg="#333333", fg="yellow",
        height=1, width=14, relief=RIDGE
    ).place(x=200, y=220)

    subjects_list = tk.Label(
        show_win, text="", bg="#1c1c1c", fg="cyan", font=("Verdana", 11)
    )
    subjects_list.place(x=60, y=275)

    show_win.mainloop()


def showTable(df, subject):
    """Display attendance data in a scrollable table."""
    table_win = tk.Tk()
    table_win.title(f"Attendance of {subject}")
    table_win.configure(background="#1c1c1c")

    tk.Label(
        table_win, text=f"Attendance Sheet: {subject}",
        bg="#1c1c1c", fg="yellow", font=("Verdana", 18, "bold")
    ).pack(pady=10)

    # Scrollable frame
    container = tk.Frame(table_win, bg="#1c1c1c")
    container.pack(fill=BOTH, expand=True, padx=10, pady=10)

    canvas = tk.Canvas(container, bg="#1c1c1c")
    scrollbar_y = tk.Scrollbar(container, orient=VERTICAL, command=canvas.yview)
    scrollbar_x = tk.Scrollbar(table_win, orient=HORIZONTAL, command=canvas.xview)

    scrollable_frame = tk.Frame(canvas, bg="#1c1c1c")
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    scrollbar_x.pack(side=BOTTOM, fill=X)
    scrollbar_y.pack(side=RIGHT, fill=Y)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)

    # Table headers
    cols = list(df.columns)
    for j, col in enumerate(cols):
        tk.Label(
            scrollable_frame, text=col, bg="#333333", fg="yellow",
            font=("Verdana", 11, "bold"), bd=2, relief=RIDGE, width=14, height=2
        ).grid(row=0, column=j, sticky="nsew")

    # Table rows
    for i, row in df.iterrows():
        bg_color = "#1c1c1c" if i % 2 == 0 else "#2a2a2a"
        for j, val in enumerate(row):
            tk.Label(
                scrollable_frame, text=str(val), bg=bg_color, fg="white",
                font=("Verdana", 10), bd=1, relief=RIDGE, width=14, height=1
            ).grid(row=i+1, column=j, sticky="nsew")

    table_win.geometry("900x500")
    table_win.mainloop()
