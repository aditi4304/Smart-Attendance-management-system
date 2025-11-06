from tkinter import *
from PIL import Image, ImageTk
from Student import Student
from tkinter import messagebox
import numpy as np
from time import strftime
from datetime import datetime
from face_detect_attempt1 import *
import recognition_module as rg
import cv2
import os
from Attendance import Attendance
from chatbot_ui import ChatbotUI
import csv
import time
from collections import defaultdict

# ==============================================================================
# MAIN GUI CLASS WITH INTEGRATED MODEL
# ==============================================================================
class Face_Recognition_System:
    RECOGNITION_CONFIDENCE = 0.30  # Confidence threshold for face recognition

    def __init__(self, root):
        self.root = root
        self.root.title("Smart Face Recognition Attendance System")
        self.root.configure(bg="black")
        self.root.state('zoomed')  # Start maximized

        # Create data directories if not exist
        if not os.path.exists("Data"):
            os.makedirs("Data")
        if not os.path.exists("attendance"):
            os.makedirs("attendance")
        if not os.path.exists("models"):
            os.makedirs("models")

        # Initialize model variables
        self.label_info = {}
        self.support_transform = None
        self.model_loaded = False

        # For temporal smoothing in recognition
        self.prev_predictions = defaultdict(list)

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI interface"""
        # Main frame with proportional sizing
        self.main_frame = Frame(self.root, bg="white")
        self.main_frame.pack(fill=BOTH, expand=True)

        # Configure grid weights for responsiveness
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.columnconfigure(2, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=1)

        # Background image setup
        self.bg_image = None
        self.bg_photo = None
        self.bg_label = Label(self.main_frame)
        self.bg_label.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.load_background_image()

        # Bind resize event
        self.root.bind("<Configure>", self.on_window_resize)

        # Time display function
        def time():
            time_string = strftime("%H:%M:%S %p")
            tim.config(text=time_string)
            tim.after(1000, time)

        # Title bar
        title_bar = Frame(self.main_frame, bg="Darkred")
        title_bar.grid(row=0, column=0, columnspan=3, sticky='ew')
        title = Label(title_bar, text="Advanced Face Recognition Attendance System",
                     font=("Arial", 28), bg="Darkred", fg="white")
        title.pack(side=LEFT, padx=20, pady=10)
        tim = Label(title_bar, font=("Arial", 14), bg="darkred", fg="white")
        tim.pack(side=RIGHT, padx=20, pady=10)
        time()

        # Button style configuration
        btn_font = ("Arial", 16, "bold")
        btn_bg = "black"
        btn_fg = "white"

        # Buttons with proportional positioning
        buttons = [
            ("Student Detail", self.student_buttons, 1, 0, "Student Management"),
            ("Take Attendance", self.face_recognize, 1, 1, "Live Face Recognition"),
            ("Attendance", self.Attendance_buttons, 1, 2, "Attendance Records"),
            ("Exit", self.exit, 2, 0, "Exit Application"),
            # ("Photos", self.open_img, 2, 1, "View Photos"),
            ("Model Status", self.chatbot_ui_button, 2, 2, "Generate Reports")
        ]

        for text, command, row, col, label_text in buttons:
            # Create a container frame for each button
            btn_frame = Frame(self.main_frame, bg="black", padx=10, pady=10)
            btn_frame.grid(row=row, column=col, padx=20, pady=20, sticky='nsew')
            btn_frame.columnconfigure(0, weight=1)
            btn_frame.rowconfigure(0, weight=1)
            
            # Create the button
            btn = Button(btn_frame, text=text, font=btn_font,
                        bg=btn_bg, fg=btn_fg, command=command)
            btn.grid(sticky='nsew')
            
            # Add label below button
            label = Label(self.main_frame, text=label_text, font=("Arial", 12),
                         bg="black", fg="white")
            label.grid(row=row+1, column=col, sticky='n')

    def load_background_image(self):
        """Load background image"""
        try:
            # Load the original image
            self.original_bg = Image.open("img1.jpg")
            self.update_background()
        except:
            # Fallback to solid color
            self.bg_label.config(bg="lightblue")

    def update_background(self):
        """Update background image on window resize"""
        if hasattr(self, 'original_bg'):
            # Get window size
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            
            # Only resize if window has valid dimensions
            if width > 1 and height > 1:
                # Resize the image to fit the window
                resized_img = self.original_bg.resize((width, height), Image.LANCZOS)
                self.bg_photo = ImageTk.PhotoImage(resized_img)
                self.bg_label.config(image=self.bg_photo)

    def on_window_resize(self, event):
        """Handle window resize event"""
        if event.widget == self.root:
            self.update_background()

    def student_buttons(self):
        """Open student management window"""
        self.new_window = Toplevel(self.root)
        self.app = Student(self.new_window)

    def Attendance_buttons(self):
        """Open attendance records window"""
        self.new_window = Toplevel(self.root)
        self.app = Attendance(self.new_window)

    def chatbot_ui_button(self):
        """Open Chatbot window"""
        self.new_window = Toplevel(self.root)
        self.app = ChatbotUI(self.new_window)

    def open_img(self):
        """Open data directory"""
        if os.path.exists("Data"):
            os.startfile("Data")
        else:
            messagebox.showinfo("Info", "Data directory not found")

    def mark_attendance(self, student_id, name, roll, batch, status="Present"):
        """Mark attendance for a student"""
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"attendance/{today}.csv"
        os.makedirs("attendance", exist_ok=True)

        # Create file if missing
        if not os.path.exists(filename):
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Name", "Roll", "Batch", "Time", "Date", "Status"])

        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")

        # Read existing data
        rows = []
        student_exists = False
        if os.path.exists(filename):
            with open(filename, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row and row[0] == str(student_id):
                        # Update existing record
                        row[4] = time_str  # Update time
                        row[6] = status    # Update status
                        student_exists = True
                    rows.append(row)

        # Add new record if student doesn't exist
        if not student_exists:
            rows.append([student_id, name, roll, batch, time_str, today, status])

        # Write updated data
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(rows)

        print(f"✓ Attendance marked for {name} ({student_id}) - Roll: {roll}")

    def mark_all_absent(self):
        """Mark all enrolled students as absent for today"""
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"attendance/{today}.csv"

        # Get all enrolled students
        all_students = []
        if os.path.exists("Data"):
            for batch in os.listdir("Data"):
                batch_path = os.path.join("Data", batch)
                if os.path.isdir(batch_path):
                    for student_folder in os.listdir(batch_path):
                        student_path = os.path.join(batch_path, student_folder)
                        if os.path.isdir(student_path):
                            try:
                                parts = student_folder.split('_')
                                student_id = parts[0]
                                student_name = ' '.join(parts[1:])

                                # Get roll number from info.txt
                                roll = "Unknown"
                                info_file = os.path.join(student_path, "info.txt")
                                if os.path.exists(info_file):
                                    with open(info_file, 'r') as f:
                                        lines = f.read().splitlines()
                                        if len(lines) >= 1:
                                            roll = lines[0]

                                all_students.append({
                                    'id': student_id,
                                    'name': student_name,
                                    'roll': roll,
                                    'batch': batch
                                })
                            except:
                                continue

        # Mark all as absent
        for student in all_students:
            self.mark_attendance(
                student['id'],
                student['name'],
                student['roll'],
                student['batch'],
                "Absent"
            )

        print(f"Marked {len(all_students)} students as absent")

    def face_recognize(self):
        """Main face recognition function with live camera feed"""
        print("Starting live face recognition...")
        
        # First, mark all students as absent
        self.mark_all_absent()

        # Initialize face detection
        try:
            system = ObjectDetectionSystem()
            use_yolo = True
            print("✓ YOLO face detector loaded")
        except:
            # Fallback to Haar cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            use_yolo = False
            print("⚠️ DNN detector not found, using Haar cascade")

        # Start video capture
        video_cap = cv2.VideoCapture(0)
        if not video_cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return

        # Set optimized resolution for performance
        video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        data_folder = "Data/Btech AI"
        label_info = {}
        student_folders = [f for f in os.listdir(data_folder) 
                    if os.path.isdir(os.path.join(data_folder, f))]
        
        for student_folder in student_folders:
            info_path = os.path.join(data_folder, student_folder)
            for txt in os.listdir(info_path):
                if txt.endswith(".txt"):
                    with open(os.path.join(info_path, txt)) as text:
                        label_info[student_folder] = {
                                        'name': student_folder[12:],
                                        'id': student_folder[:11],
                                        'roll': text.read()[:4]
                                    }

        # Create window
        cv2.namedWindow("Advanced Face Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Advanced Face Recognition", 1024, 576)

        # Tracking variables
        marked_today = set()
        session_start = time.time()
        frame_count = 0
        FRAME_SKIP = 2  # Process every 3rd frame for performance

        # Initialize elapsed time variables to prevent UnboundLocalError
        elapsed_min = 0
        elapsed_sec = 0

        print("Press 's' to take picture and get started. Press 'q' to exit.")

        try:
            while True:
                ret, frame = video_cap.read()
                if not ret:
                    print("Error reading frame from camera")
                    break

                frame_count += 1
                if frame_count % FRAME_SKIP != 0:
                    continue

                current_time = time.time()
                h, w = frame.shape[:2]

                # Calculate elapsed time (moved here to ensure it's always calculated)
                elapsed_time = current_time - session_start
                elapsed_min, elapsed_sec = divmod(int(elapsed_time), 60)

                # Display session statistics
                cv2.putText(frame, f"Session: {elapsed_min:02d}:{elapsed_sec:02d}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Present: {len(marked_today)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to Exit", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Press 's' to Mark Attendance", (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Advanced Face Recognition", frame)

                # Face detection
                if use_yolo and cv2.waitKey(50) & 0xFF == ord('s'):
                    enhanced_frame = system.preprocessor.enhance_frame(frame=frame)
                    results = system.model(enhanced_frame, verbose=False, conf=0.3, iou=0.5)
                    detections = system.process_detections(results, frame.shape)

                    for i in detections:
                        confidence = int(i["confidence"])
                        # if confidence < 0.5:
                        #     continue

                        startX, startY, endX, endY = i["bbox"]
                        startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)

                        # Ensure valid dimensions
                        startX, startY = max(0, startX), max(0, startY)
                        endX, endY = min(w, endX), min(h, endY)

                        # if endX - startX < 50 or endY - startY < 50:
                        #     continue

                        # Extract face
                        face_roi = frame[startY:endY, startX:endX]
                        # if face_roi.size == 0:
                        #     continue

                        # Predict identity
                        predicted_class, pred_confidence = rg.recognise(face_roi)

                        if pred_confidence > self.RECOGNITION_CONFIDENCE and predicted_class in label_info:
                            # Known student
                            info = label_info[predicted_class]
                            student_id = info['id']
                            student_name = info['name']
                            student_roll = info['roll']

                            # Mark attendance if not already marked
                            if student_id not in marked_today:
                                self.mark_attendance(
                                    student_id,
                                    student_name,
                                    student_roll,
                                    info.get('batch', 'Unknown'),
                                    "Present"
                                )
                                marked_today.add(student_id)
                                print(f"✓ Marked attendance for {student_name} (conf: {pred_confidence:.3f})")

                            # Draw bounding box for known student
                            color = (0, 255, 0)  # Green
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                            # Draw student info
                            text_y = startY - 10 if startY > 30 else endY + 20
                            cv2.putText(frame, f"{student_name}", (startX, text_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, f"Conf: {pred_confidence:.2f}", (startX, text_y + 25),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                            cv2.putText(frame, f"Roll: {student_roll}", (startX, text_y + 45),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            # Unknown face
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
                            cv2.putText(frame, f"Unknown ({pred_confidence:.2f})", (startX, startY - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                elif cv2.waitKey(50) & 0xFF == ord('q'):
                    print("📋 Attendance session ended by user")
                    break

        except Exception as e:
            print(f"Error during face recognition: {e}")
            # Calculate final elapsed time in case of error
            final_time = time.time() - session_start
            elapsed_min, elapsed_sec = divmod(int(final_time), 60)

        finally:
            # Ensure camera and windows are properly closed
            video_cap.release()
            cv2.destroyAllWindows()

            # Calculate final elapsed time if not already calculated
            if 'final_time' not in locals():
                final_time = time.time() - session_start
                elapsed_min, elapsed_sec = divmod(int(final_time), 60)

            messagebox.showinfo("Attendance Complete",
                              f"✅ Session completed successfully!\n\n"
                              f"Students marked present: {len(marked_today)}\n"
                              f"Session duration: {elapsed_min:02d}:{elapsed_sec:02d}")

    def exit(self):
        """Exit the application"""
        self.root.destroy()


if __name__ == "__main__":
    print("🚀 Starting Advanced Face Recognition Attendance System...")
    print("=" * 70)
    root = Tk()
    obj = Face_Recognition_System(root)
    root.mainloop()