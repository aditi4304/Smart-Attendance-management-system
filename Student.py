from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import os
import cv2
from tkinter import simpledialog
import numpy as np
import urllib.request
import imgaug.augmenters as iaa

class Student:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Management")
        self.root.configure(bg="white")
        self.root.state('zoomed')  # Start maximized
       
        # Create Data directory if not exists
        if not os.path.exists("Data"):
            os.makedirs("Data")
           
        self.var_batch = StringVar()
        self.var_Id = StringVar()
        self.var_name = StringVar()
        self.var_roll = StringVar()
        self.var_gender = StringVar()
        self.var_DOB = StringVar()
        self.var_Email = StringVar()
        self.var_Phone = StringVar()
        self.var_radio = StringVar(value="Yes")  # Default to Yes for mandatory image capture
       
        # Main container frame
        main_frame = Frame(self.root, bg="white")
        main_frame.pack(fill=BOTH, expand=True)
       
        # Background image setup
        self.bg_image = None
        self.bg_photo = None
        self.bg_label = Label(main_frame)
        self.bg_label.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.load_background_image()
       
        # Bind resize event
        self.root.bind("<Configure>", self.on_window_resize)
       
        title = Label(main_frame, text="Student Management",
                     font=("Arial", 30, "bold"), bg="darkred", fg="white")
        title.pack(side=TOP, fill=X, pady=(0, 20))

        # Main content frame
        content_frame = Frame(main_frame, bg="white", bd=2, relief=RIDGE)
        content_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)

        # Left and right frames
        left_frame = LabelFrame(content_frame, bd=2, bg="white", relief=RIDGE,
                              text="Student Details", font=("Arial", 12, "bold"))
        left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)
       
        right_frame = LabelFrame(content_frame, bd=2, bg="white", relief=RIDGE,
                               text="Student Directory", font=("Arial", 12, "bold"))
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)
       
        # Configure grid weights for left frame
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
       
        # Batch Information Section
        batch_frame = LabelFrame(left_frame, bd=2, bg="white", relief=RIDGE,
                               text="Batch Information", font=("Arial", 12, "bold"))
        batch_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)
       
        # Batch Selection
        batch_label = Label(batch_frame, text="Select Batch:", font=("Arial", 12), bg="white")
        batch_label.grid(row=0, column=0, padx=5, pady=5, sticky=W)
       
        self.batch_combo = ttk.Combobox(batch_frame, textvariable=self.var_batch,
                                      font=("Arial", 12), width=22, state="readonly")
        self.update_batch_list()
        self.batch_combo.grid(row=0, column=1, padx=5, pady=5)
       
        # Create New Batch
        new_batch_btn = Button(batch_frame, text="Create Batch", command=self.create_batch,
                              font=("Arial", 12), bg="blue", fg="white")
        new_batch_btn.grid(row=0, column=2, padx=10, pady=5)
       
        # Exit Button
        exit_btn = Button(left_frame, text="Exit", command=self.exit,
                         font=("Arial", 12), bg="red", fg="white")
        exit_btn.grid(row=0, column=1, padx=10, pady=10, sticky='ne')
       
        # Student Information Section
        student_frame = LabelFrame(left_frame, bd=2, bg="white", relief=RIDGE,
                                 text="Student Information", font=("Arial", 12, "bold"))
        student_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)
       
        # Configure student frame grid
        for i in range(8):
            student_frame.rowconfigure(i, weight=1)
        student_frame.columnconfigure(0, weight=1)
        student_frame.columnconfigure(1, weight=2)
       
        # Student ID
        Id_label = Label(student_frame, text="Student ID:", font=("Arial", 12), bg="white")
        Id_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        Id_entry = ttk.Entry(student_frame, textvariable=self.var_Id, font=("Arial", 12))
        Id_entry.grid(row=0, column=1, padx=10, pady=5, sticky='ew')
       
        # Name
        name_label = Label(student_frame, text="Student Name:", font=("Arial", 12), bg="white")
        name_label.grid(row=1, column=0, padx=10, pady=5, sticky='w')
        name_entry = ttk.Entry(student_frame, textvariable=self.var_name, font=("Arial", 12))
        name_entry.grid(row=1, column=1, padx=10, pady=5, sticky='ew')
       
        # Roll No (alphanumeric)
        roll_label = Label(student_frame, text="Roll No:", font=("Arial", 12), bg="white")
        roll_label.grid(row=2, column=0, padx=10, pady=5, sticky='w')
        roll_entry = ttk.Entry(student_frame, textvariable=self.var_roll, font=("Arial", 12))
        roll_entry.grid(row=2, column=1, padx=10, pady=5, sticky='ew')
       
        # Gender
        gender_label = Label(student_frame, text="Gender:", font=("Arial", 12), bg="white")
        gender_label.grid(row=3, column=0, padx=10, pady=5, sticky='w')
        gender_combo = ttk.Combobox(student_frame, textvariable=self.var_gender,
                                  font=("Arial", 12), state="readonly")
        gender_combo["values"] = ("Male", "Female", "Other")
        gender_combo.grid(row=3, column=1, padx=10, pady=5, sticky='ew')
       
        # DOB
        dob_label = Label(student_frame, text="Date of Birth:", font=("Arial", 12), bg="white")
        dob_label.grid(row=4, column=0, padx=10, pady=5, sticky='w')
        dob_entry = ttk.Entry(student_frame, textvariable=self.var_DOB, font=("Arial", 12))
        dob_entry.grid(row=4, column=1, padx=10, pady=5, sticky='ew')
       
        # Email
        email_label = Label(student_frame, text="Email:", font=("Arial", 12), bg="white")
        email_label.grid(row=5, column=0, padx=10, pady=5, sticky='w')
        email_entry = ttk.Entry(student_frame, textvariable=self.var_Email, font=("Arial", 12))
        email_entry.grid(row=5, column=1, padx=10, pady=5, sticky='ew')
       
        # Phone
        phone_label = Label(student_frame, text="Phone:", font=("Arial", 12), bg="white")
        phone_label.grid(row=6, column=0, padx=10, pady=5, sticky='w')
        phone_entry = ttk.Entry(student_frame, textvariable=self.var_Phone, font=("Arial", 12))
        phone_entry.grid(row=6, column=1, padx=10, pady=5, sticky='ew')
       
        # Image Capture Notice
        photo_notice = Label(student_frame, text="Image capture is mandatory for attendance",
                            font=("Arial", 12), bg="white", fg="red")
        photo_notice.grid(row=7, column=0, columnspan=2, pady=10)
       
        # Buttons
        btn_frame = Frame(left_frame, bg="white")
        btn_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=10)
       
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)
        btn_frame.columnconfigure(3, weight=1)
       
        save_btn = Button(btn_frame, text="Save & Capture Images", command=self.add_data,
                         font=("Arial", 12), bg="green", fg="white")
        save_btn.grid(row=0, column=0, padx=5, sticky='ew')
       
        update_btn = Button(btn_frame, text="Update", command=self.update_data,
                           font=("Arial", 12), bg="green", fg="white")
        update_btn.grid(row=0, column=1, padx=5, sticky='ew')
       
        delete_btn = Button(btn_frame, text="Delete", command=self.delete_data,
                           font=("Arial", 12), bg="red", fg="white")
        delete_btn.grid(row=0, column=2, padx=5, sticky='ew')
       
        reset_btn = Button(btn_frame, text="Reset", command=self.reset,
                          font=("Arial", 12), bg="gray", fg="white")
        reset_btn.grid(row=0, column=3, padx=5, sticky='ew')
       
        # Right Frame - Student Directory
        table_frame = Frame(right_frame, bg="white")
        table_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
       
        scroll_x = ttk.Scrollbar(table_frame, orient=HORIZONTAL)
        scroll_y = ttk.Scrollbar(table_frame, orient=VERTICAL)
       
        self.student_table = ttk.Treeview(table_frame, columns=(
            "id", "name", "roll", "batch", "gender", "dob", "email", "phone"),
            xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
       
        scroll_x.pack(side=BOTTOM, fill=X)
        scroll_y.pack(side=RIGHT, fill=Y)
        scroll_x.config(command=self.student_table.xview)
        scroll_y.config(command=self.student_table.yview)
       
        self.student_table.heading("id", text="ID")
        self.student_table.heading("name", text="Name")
        self.student_table.heading("roll", text="Roll No")
        self.student_table.heading("batch", text="Batch")
        self.student_table.heading("gender", text="Gender")
        self.student_table.heading("dob", text="DOB")
        self.student_table.heading("email", text="Email")
        self.student_table.heading("phone", text="Phone")
       
        self.student_table["show"] = "headings"
       
        self.student_table.column("id", width=50)
        self.student_table.column("name", width=100)
        self.student_table.column("roll", width=70)
        self.student_table.column("batch", width=100)
        self.student_table.column("gender", width=70)
        self.student_table.column("dob", width=80)
        self.student_table.column("email", width=120)
        self.student_table.column("phone", width=80)
       
        self.student_table.pack(fill=BOTH, expand=1)
        self.student_table.bind("<ButtonRelease>", self.get_cursor)
       
        self.load_student_data()
       
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
   
    def update_batch_list(self):
        if os.path.exists("Data"):
            batches = [d for d in os.listdir("Data") if os.path.isdir(os.path.join("Data", d))]
            self.batch_combo["values"] = batches
            if batches:
                self.batch_combo.current(0)
   
    def create_batch(self):
        batch_name = simpledialog.askstring("New Batch", "Enter batch name (e.g., Btech_AI):", parent=self.root)
        if batch_name:
            batch_path = os.path.join("Data", batch_name)
            if not os.path.exists(batch_path):
                os.makedirs(batch_path)
                messagebox.showinfo("Success", f"Batch '{batch_name}' created successfully")
                self.update_batch_list()
            else:
                messagebox.showerror("Error", "Batch already exists")
   
    def load_student_data(self):
        # Clear table
        for i in self.student_table.get_children():
            self.student_table.delete(i)
           
        # Load from batches
        if os.path.exists("Data"):
            for batch in os.listdir("Data"):
                batch_path = os.path.join("Data", batch)
                if os.path.isdir(batch_path):
                    for student_folder in os.listdir(batch_path):
                        student_path = os.path.join(batch_path, student_folder)
                        if os.path.isdir(student_path):
                            try:
                                # Split folder name to get ID and name
                                parts = student_folder.split('_')
                                student_id = parts[0]
                                student_name = ' '.join(parts[1:])
                               
                                # Check for info file
                                info_file = os.path.join(student_path, "info.txt")
                                if os.path.exists(info_file):
                                    with open(info_file, "r") as f:
                                        info = f.read().splitlines()
                                        if len(info) >= 6:
                                            roll = info[0]
                                            gender = info[1]
                                            dob = info[2]
                                            email = info[3]
                                            phone = info[4]
                                            batch_name = info[5] if len(info) > 5 else batch
                                else:
                                    roll = ""
                                    gender = ""
                                    dob = ""
                                    email = ""
                                    phone = ""
                                    batch_name = batch
                               
                                self.student_table.insert("", END, values=(
                                    student_id, student_name, roll, batch_name,
                                    gender, dob, email, phone
                                ))
                            except Exception as e:
                                print(f"Error loading student {student_folder}: {str(e)}")
   
    def add_data(self):
        if not all([self.var_Id.get(), self.var_name.get(), self.var_batch.get()]):
            messagebox.showerror("Error", "ID, Name, and Batch are required")
            return
           
        # Allow alphanumeric IDs
        student_folder = f"{self.var_Id.get()}_{self.var_name.get()}"
        batch_path = os.path.join("Data", self.var_batch.get())
        student_path = os.path.join(batch_path, student_folder)
       
        if not os.path.exists(batch_path):
            messagebox.showerror("Error", "Selected batch doesn't exist")
            return
           
        if os.path.exists(student_path):
            messagebox.showerror("Error", "Student with this ID already exists in the batch")
            return
           
        # Create student folder
        os.makedirs(student_path)
       
        # Save info
        info_file = os.path.join(student_path, "info.txt")
        with open(info_file, "w") as f:
            f.write(f"{self.var_roll.get()}\n")       # Alphanumeric roll number
            f.write(f"{self.var_gender.get()}\n")
            f.write(f"{self.var_DOB.get()}\n")
            f.write(f"{self.var_Email.get()}\n")
            f.write(f"{self.var_Phone.get()}\n")
            f.write(f"{self.var_batch.get()}\n")
       
        # Immediately capture images
        self.generate_dataset(student_path)
       
        messagebox.showinfo("Success", "Student added and images captured successfully")
        self.load_student_data()
        self.reset()

    def generate_dataset(self, student_path):
        """Automatically capture images with augmentation"""
        # Load DNN face detector
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
       
        proto_path = "deploy.prototxt"
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
       
        if not os.path.exists(proto_path):
            urllib.request.urlretrieve(proto_url, proto_path)
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(model_url, model_path)
       
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
       
        # Create augmentation sequence
        augmenter = iaa.Sequential([
            iaa.Fliplr(0.5),  # Horizontal flip
            iaa.Affine(
                rotate=(-20, 20),  # Rotation
                scale=(0.8, 1.2),  # Scaling
                translate_percent=(-0.1, 0.1)  # Translation
            ),
            iaa.GaussianBlur(sigma=(0, 1.0)),  # Blur
            iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Noise
            iaa.LinearContrast((0.8, 1.2)),  # Contrast
            iaa.AddToHueAndSaturation((-20, 20))  # Hue/Saturation
        ])

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
       
        img_id = 0
        captured_count = 0
        required_images = 1  # More images for better training
        frame_count = 0
        min_face_size = 60
       
        cv2.namedWindow("Capturing Face Images", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Capturing Face Images", 1024, 768)
       
        while captured_count < required_images:
            ret, frame = cap.read()
            if not ret:
                break
               
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (400, 400)), 1.0,
                                        (400, 400), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
           
            face_detected = False
           
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
               
                if confidence > 0.4:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                   
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w - 1, endX), min(h - 1, endY)
                   
                    face_width = endX - startX
                    face_height = endY - startY
                   
                    if face_width > min_face_size and face_height > min_face_size:
                        face_roi = frame[startY:endY, startX:endX]
                       
                        if face_roi.size > 0:
                            # Apply augmentations
                            augmented = augmenter(image=face_roi)
                           
                            # Convert to grayscale and save
                            gray = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
                            resized_face = cv2.resize(gray, (160, 160))
                           
                            # Save image
                            file_path = os.path.join(student_path, f"image_{img_id}.jpg")
                            cv2.imwrite(file_path, resized_face)
                           
                            # Display augmentation preview
                            cv2.imshow("Augmented", augmented)
                           
                            captured_count += 1
                            img_id += 1
                            face_detected = True

                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cv2.putText(frame, f"Capturing: {captured_count}/{required_images}",
                                    (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2)
           
            if not face_detected:
                cv2.putText(frame, "Move closer to the camera", (10, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
           
            cv2.putText(frame, f"Captured: {captured_count}/{required_images}", (10, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Student: {self.var_name.get()}", (10, 90),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press ESC to exit", (10, 120),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Capturing Face Images", frame)
            frame_count += 1
           
            if cv2.waitKey(1) == 27:  # ESC key to exit
                break
               
        cap.release()
        cv2.destroyAllWindows()
       
        if captured_count < required_images:
            messagebox.showwarning("Incomplete Capture",
                                  f"Only {captured_count} images captured. Minimum {required_images} recommended.")

    def update_data(self):
        selected = self.student_table.focus()
        if not selected:
            messagebox.showerror("Error", "No student selected")
            return
           
        student_id = self.var_Id.get()
        student_name = self.var_name.get()
        batch = self.var_batch.get()
       
        if not all([student_id, student_name, batch]):
            messagebox.showerror("Error", "ID, Name, and Batch are required")
            return
           
        # Find existing folder
        for batch_folder in os.listdir("Data"):
            batch_path = os.path.join("Data", batch_folder)
            if os.path.isdir(batch_path):
                for student_folder in os.listdir(batch_path):
                    if student_folder.startswith(f"{student_id}_"):
                        old_path = os.path.join(batch_path, student_folder)
                        new_folder = f"{student_id}_{student_name}"
                       
                        # Rename if name changed
                        if student_folder != new_folder:
                            new_path = os.path.join(batch_path, new_folder)
                            os.rename(old_path, new_path)
                        else:
                            new_path = old_path  # Use existing path if name didn't change
                       
                        # Update info file
                        info_file = os.path.join(new_path, "info.txt")
                        with open(info_file, "w") as f:
                            f.write(f"{self.var_roll.get()}\n")
                            f.write(f"{self.var_gender.get()}\n")
                            f.write(f"{self.var_DOB.get()}\n")
                            f.write(f"{self.var_Email.get()}\n")
                            f.write(f"{self.var_Phone.get()}\n")
                            f.write(f"{batch}\n")
                       
                        messagebox.showinfo("Success", "Student updated successfully")
                        self.load_student_data()
                        return
       
        messagebox.showerror("Error", "Student not found")
   
    def delete_data(self):
        selected = self.student_table.focus()
        if not selected:
            messagebox.showerror("Error", "No student selected")
            return
           
        student_id = self.var_Id.get()
       
        if not student_id:
            messagebox.showerror("Error", "No student selected")
            return
           
        # Find and delete folder
        for batch_folder in os.listdir("Data"):
            batch_path = os.path.join("Data", batch_folder)
            if os.path.isdir(batch_path):
                for student_folder in os.listdir(batch_path):
                    if student_folder.startswith(f"{student_id}_"):
                        student_path = os.path.join(batch_path, student_folder)
                        # Delete folder and contents
                        for file in os.listdir(student_path):
                            os.remove(os.path.join(student_path, file))
                        os.rmdir(student_path)
                        messagebox.showinfo("Success", "Student deleted successfully")
                        self.load_student_data()
                        self.reset()
                        return
       
        messagebox.showerror("Error", "Student not found")
   
    def get_cursor(self, event):
        cursor_row = self.student_table.focus()
        contents = self.student_table.item(cursor_row)
        values = contents["values"]
       
        if values:
            self.var_Id.set(values[0])
            self.var_name.set(values[1])
            self.var_roll.set(values[2])
            self.var_batch.set(values[3])
            self.var_gender.set(values[4])
            self.var_DOB.set(values[5])
            self.var_Email.set(values[6])
            self.var_Phone.set(values[7])
   
    def reset(self):
        self.var_Id.set("")
        self.var_name.set("")
        self.var_roll.set("")
        self.var_gender.set("")
        self.var_DOB.set("")
        self.var_Email.set("")
        self.var_Phone.set("")
        self.update_batch_list()
   
    def exit(self):
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    obj = Student(root)
    root.mainloop()