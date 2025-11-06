from tkinter import *
from tkinter import ttk, messagebox, filedialog
import os
import csv
import glob
import pandas as pd
from datetime import datetime
import numpy as np
from time import strftime
from PIL import Image, ImageTk

class Attendance:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance Records")
        self.root.configure(bg="white")
        self.root.state('zoomed')  # Start in maximized mode

        # Main container frame
        main_frame = Frame(self.root, bg="white")
        main_frame.pack(fill=BOTH, expand=True)
       
        # Background setup
        self.bg_image = None
        self.bg_photo = None
        self.bg_label = Label(main_frame)
        self.bg_label.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.load_background_image()
       
        # Bind resize event
        self.root.bind("<Configure>", self.on_window_resize)
       
        # Attendance Records Frame (Full Screen)
        records_frame = LabelFrame(main_frame, bd=2, bg="white", relief=RIDGE,
                               text="Attendance Records", font=("Arial", 15, "bold"))
        records_frame.place(relx=0.01, rely=0.01, relwidth=0.98, relheight=0.98)
       
        # Search Panel
        search_frame = Frame(records_frame, bg="white")
        search_frame.pack(fill=X, padx=10, pady=10)
       
        # Exit Button
        exit_btn = Button(search_frame, text="Exit", command=self.exit,
                         font=("Arial", 12, "bold"), bg="red", fg="white")
        exit_btn.grid(row=0, column=0, padx=5)
       
        # Date Selector
        ttk.Label(search_frame, text="Select Date:").grid(row=0, column=1, padx=5)
        self.date_var = StringVar()
        self.date_combo = ttk.Combobox(search_frame, textvariable=self.date_var, width=15, state="readonly")
        self.date_combo.grid(row=0, column=2, padx=5)
        ttk.Button(search_frame, text="Load", command=self.load_date_data).grid(row=0, column=3, padx=5)
       
        # Student Search
        ttk.Label(search_frame, text="Search Student ID:").grid(row=0, column=4, padx=5)
        self.search_var = StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=15)
        search_entry.grid(row=0, column=5, padx=5)
        ttk.Button(search_frame, text="Search", command=self.search_student).grid(row=0, column=6, padx=5)
       
        # Export Button
        export_btn = ttk.Button(search_frame, text="Export to Excel", command=self.export_report)
        export_btn.grid(row=0, column=7, padx=5)
       
        # Treeview for records
        tree_frame = Frame(records_frame, bg="white")
        tree_frame.pack(fill=BOTH, expand=True, padx=10, pady=(0,10))
       
        scroll_x = ttk.Scrollbar(tree_frame, orient=HORIZONTAL)
        scroll_y = ttk.Scrollbar(tree_frame, orient=VERTICAL)
       
        # Updated columns to include sleep data
        columns = ("id", "name", "roll", "batch", "date", "time", "attendance", "awake", "asleep")
        self.attendance_table = ttk.Treeview(
            tree_frame, columns=columns, show="headings",
            xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set
        )
       
        scroll_x.pack(side=BOTTOM, fill=X)
        scroll_y.pack(side=RIGHT, fill=Y)
        scroll_x.config(command=self.attendance_table.xview)
        scroll_y.config(command=self.attendance_table.yview)
       
        # Configure headings and columns
        headings = ["ID", "Name", "Roll No", "Batch", "Date", "Time", "Status", "Awake Time", "Asleep Time"]
        col_widths = [70, 150, 80, 120, 100, 80, 80, 80, 80]
        for col, heading, width in zip(columns, headings, col_widths):
            self.attendance_table.heading(col, text=heading)
            self.attendance_table.column(col, width=width, anchor=CENTER)
       
        self.attendance_table.pack(fill=BOTH, expand=True)
       
        # Status label
        self.status_var = StringVar()
        status_label = Label(records_frame, textvariable=self.status_var, bg="white",
                            font=("Arial", 10), fg="blue")
        status_label.pack(side=BOTTOM, fill=X, padx=10, pady=5)
       
        # Load available dates
        self.update_date_list()
   
    def load_background_image(self):
        try:
            self.original_bg = Image.open("img1.jpg")
            self.update_background()
        except:
            self.bg_label.config(bg="lightblue")
   
    def update_background(self):
        if hasattr(self, 'original_bg'):
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            if width > 1 and height > 1:
                resized_img = self.original_bg.resize((width, height), Image.LANCZOS)
                self.bg_photo = ImageTk.PhotoImage(resized_img)
                self.bg_label.config(image=self.bg_photo)
   
    def on_window_resize(self, event):
        if event.widget == self.root:
            self.update_background()
   
    def update_date_list(self):
        dates = []
        if os.path.exists("attendance"):
            for file in os.listdir("attendance"):
                if file.endswith('.csv'):
                    dates.append(file.split('.')[0])
        self.date_combo['values'] = sorted(dates, reverse=True)
        if dates:
            self.date_combo.current(0)
   
    def load_date_data(self):
        date = self.date_var.get()
        if not date:
            messagebox.showwarning("Input Error", "Please select a date")
            return
           
        file_path = os.path.join("attendance", f"{date}.csv")
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"No attendance data for {date}")
            return
           
        try:
            df = pd.read_csv(file_path)
            self.display_data(df)
            self.status_var.set(f"Showing attendance for {date} - {len(df)} records")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
   
    def search_student(self):
        student_id = self.search_var.get().strip()
        if not student_id:
            messagebox.showwarning("Input Error", "Please enter a student ID")
            return
           
        all_files = glob.glob(os.path.join("attendance", "*.csv"))
        if not all_files:
            messagebox.showinfo("Info", "No attendance records found")
            return
           
        all_data = []
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path)
                if "ID" in df.columns and student_id in df["ID"].astype(str).values:
                    student_data = df[df["ID"].astype(str) == student_id]
                    all_data.append(student_data)
            except:
                continue
       
        if not all_data:
            self.status_var.set(f"No records found for ID: {student_id}")
            return
           
        full_df = pd.concat(all_data)
        self.display_data(full_df)
       
        # Calculate attendance percentage
        total_days = len(all_files)
        present_days = len([d for d in all_data if d['Status'].iloc[0] == "Present"])
        percentage = (present_days / total_days) * 100 if total_days > 0 else 0
        self.status_var.set(f"Student {student_id}: Present {present_days}/{total_days} days ({percentage:.1f}%)")
   
    def display_data(self, df):
        # Clear existing data
        for item in self.attendance_table.get_children():
            self.attendance_table.delete(item)
           
        # Insert data with all 9 columns
        for _, row in df.iterrows():
            self.attendance_table.insert("", "end", values=(
                str(row.get("ID", "")),
                str(row.get("Name", "")),
                str(row.get("Roll", "")),
                str(row.get("Batch", "")),
                str(row.get("Date", "")),
                str(row.get("Time", "")),
                str(row.get("Status", "")),
                str(row.get("Awake Time", "")),  # Add awake time
                str(row.get("Asleep Time", ""))  # Add asleep time
            ))
   
    def export_report(self):
        items = self.attendance_table.get_children()
        if not items:
            messagebox.showwarning("Export Error", "No data to export")
            return
           
        data = []
        columns = ("id", "name", "roll", "batch", "date", "time", "attendance", "awake", "asleep")
        for item in items:
            data.append(self.attendance_table.item(item)['values'])
       
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
       
        # Save dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if not file_path:
            return
           
        try:
            # Add summary statistics
            present_count = len(df[df['attendance'] == 'Present'])
            absent_count = len(df[df['attendance'] == 'Absent'])
            total_count = len(df)
           
            summary = pd.DataFrame({
                'Total Students': [total_count],
                'Present': [present_count],
                'Absent': [absent_count],
                'Attendance Percentage': [f"{(present_count/total_count)*100:.1f}%"]
            })
           
            with pd.ExcelWriter(file_path) as writer:
                df.to_excel(writer, sheet_name="Attendance Records", index=False)
                summary.to_excel(writer, sheet_name="Summary", index=False)
               
            messagebox.showinfo("Success", f"Report exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
   
    def exit(self):
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    obj = Attendance(root)
    root.mainloop()