import tkinter as tk
from tkinter import ttk, scrolledtext
import os
from datetime import datetime
import markdown
from tkhtmlview import HTMLLabel
import threading
from CSV_RAG_backend import CSVRagChatbot
from CSV_reader import DataGetter
import time

class ChatbotUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_widgets()
        self.setup_layout()
        self.bind_events()
        self.DATA_PATH = "/Users/shreyassawant/mydrive/Shreyus_workspace/Semester_VII/CV/project/attendance"
        try:
            self.init_database()
        
        except Exception as e:
            print("Could not initiate Database due to:")
            print(e)

    def setup_window(self):
        """Configure the main window"""
        self.root.title("AI Chatbot")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)

        # Modern color scheme
        self.bg_color = "#2C2C2C"
        self.chat_bg = "#383838"
        self.user_msg_color = "#0084FF"
        self.bot_msg_color = "#E4E6EA"
        self.text_color = "#FFFFFF"
        self.input_bg = "#404040"

        self.root.configure(bg=self.bg_color)

    def create_widgets(self):
        """Create all UI widgets"""
        # Main container frame
        self.main_frame = ttk.Frame(self.root)

        # Header frame
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_label = tk.Label(
            self.header_frame,
            text="🤖 AI Chatbot Assistant",
            font=("Arial", 16, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
            pady=10
        )

        # Chat display area with scrollbar
        self.chat_frame = ttk.Frame(self.main_frame)

        # Using scrolledtext for better scrolling experience
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            width=70,
            height=25,
            font=("Arial", 11),
            bg=self.chat_bg,
            fg=self.text_color,
            state=tk.DISABLED,
            cursor="arrow",
            selectbackground="#0084FF",
            selectforeground="white",
            borderwidth=0,
            highlightthickness=0
        )

        # Input frame
        self.input_frame = ttk.Frame(self.main_frame)

        # Message input field
        self.message_var = tk.StringVar()
        self.message_entry = tk.Entry(
            self.input_frame,
            textvariable=self.message_var,
            font=("Arial", 12),
            bg=self.input_bg,
            fg=self.text_color,
            insertbackground=self.text_color,
            borderwidth=2,
            relief="flat",
            highlightthickness=1,
            highlightcolor="#0084FF"
        )

        # Send button
        self.send_button = tk.Button(
            self.input_frame,
            text="Send",
            command=self.send_message,
            font=("Arial", 12, "bold"),
            bg=self.user_msg_color,
            fg="white",
            borderwidth=0,
            padx=20,
            pady=8,
            cursor="hand2",
            activebackground="#0066CC",
            activeforeground="white"
        )

        # Status bar
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_label = tk.Label(
            self.status_frame,
            text="Ready to chat...",
            font=("Arial", 9),
            bg=self.bg_color,
            fg="#999999",
            anchor="w"
        )

    def setup_layout(self):
        """Arrange widgets using grid layout"""
        # Configure grid weights for responsive design
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Header
        self.header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.header_frame.grid_columnconfigure(0, weight=1)
        self.header_label.grid(row=0, column=0, sticky="ew")

        # Chat area
        self.chat_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_display.grid(row=0, column=0, sticky="nsew")

        # Input area
        self.input_frame.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.message_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.send_button.grid(row=0, column=1, sticky="e")

        # Status bar
        self.status_frame.grid(row=3, column=0, sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)
        self.status_label.grid(row=0, column=0, sticky="ew")

    def bind_events(self):
        """Bind keyboard and other events"""
        # Enter key to send message
        self.message_entry.bind("<Return>", lambda event: self.send_message())

        # Focus on entry field when window is clicked
        self.root.bind("<Button-1>", lambda event: self.message_entry.focus_set())

        # Auto-focus on entry field at startup
        self.root.after(100, lambda: self.message_entry.focus_set())

    def add_message(self, message, sender="user", timestamp=True):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)

        # Add timestamp if requested
        if timestamp:
            time_str = datetime.now().strftime("%H:%M")

        # Format message based on sender
        if sender == "user":
            if timestamp:
                self.chat_display.insert(tk.END, f"You ({time_str})\n", "timestamp")
            self.chat_display.insert(tk.END, f"{message}\n\n", "user_msg")
        else:  # bot message
            if timestamp:
                self.chat_display.insert(tk.END, f"Bot ({time_str})\n", "timestamp")
            html = markdown.markdown(message)
            # Create HTML label for formatted output
            html_label = HTMLLabel(
                self.chat_display,
                html=html,
                background=self.chat_bg,
                foreground=self.text_color,
                width=70,
                padx=5,
                pady=3
            )

            self.chat_display.window_create(tk.END, window=html_label)
            self.chat_display.insert(tk.END, "\n\n")  # Add spacing

        # Configure text tags for styling
        self.chat_display.tag_configure("user_msg", 
                                       foreground="#0084FF", 
                                       font=("Arial", 11, "normal"))
        self.chat_display.tag_configure("bot_msg", 
                                       foreground="#E4E6EA", 
                                       font=("Arial", 11, "normal"))
        self.chat_display.tag_configure("timestamp", 
                                       foreground="#999999", 
                                       font=("Arial", 9, "italic"))

        # Auto-scroll to bottom
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def send_message(self):
        """Handle sending a message"""
        message = self.message_var.get().strip()

        if not message:
            return

        # Add user message to chat
        self.add_message(message, "user")

        # Clear input field
        self.message_var.set("")

        # Update status
        self.status_label.config(text="Bot is typing...")

        # Simulate bot response (replace with actual chatbot logic)
        threading.Thread(target=self.bot_response, args=(message,), daemon=True).start()

    def init_database(self):
        self.csv_files = []
        for file in os.listdir(self.DATA_PATH):
            if file.endswith(".csv"):
                self.csv_files.append(os.path.join(self.DATA_PATH, file))

    def bot_response(self, user_message):
        dg = DataGetter()
        dg.get_data(filepaths=self.csv_files, type="multi")
        query = user_message
        docs = dg.database.similarity_search(query=query, k=5)
        context = "\n\n".join([d.page_content for d in docs])

        bot = CSVRagChatbot()
        response = bot.get_response(text=context, query=query)

        # Add bot response to chat (must be done in main thread)
        self.root.after(0, lambda: self.add_message(response.content, "bot"))
        self.root.after(0, lambda: self.status_label.config(text="Ready to chat..."))

    def add_welcome_message(self):
        """Add welcome message when app starts"""
        welcome_msg = "Welcome! I\'m your AI assistant. Feel free to ask me anything or just say hello!"
        self.add_message(welcome_msg, "bot", timestamp=False)

    def run(self):
        """Start the application"""
        self.add_welcome_message()
        self.root.mainloop()

# Create and run the chatbot UI
if __name__ == "__main__":
    chatbot = ChatbotUI()
    chatbot.run()
