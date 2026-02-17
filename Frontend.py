"""
FocusGuard Frontend Application

A modern GUI for workspace monitoring with:
- Welcome screen with session configuration
- Live camera feed with status indicators
- Real-time notifications
- Focus timer
- Visual status icons for posture, gaze, and lighting
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import time
from datetime import timedelta
from typing import Optional, Dict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.base_detector import DetectionResult
from modules.lighting.lighting_detector import LightingDetector
import config


class FocusGuardApp:
    """Main application class for FocusGuard frontend."""
    
    def __init__(self, root):
        """Initialize the application."""
        self.root = root
        self.root.title("FocusGuard - Workspace Monitor")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a2e")
        
        # Session settings
        self.focus_duration = None  # Minutes
        self.break_duration = None  # Minutes
        self.session_start_time = None
        self.focus_elapsed = 0  # Seconds
        
        # Camera
        self.camera = None
        self.camera_running = False
        
        # Detectors (initialize with available modules)
        self.detectors = {
            'lighting': LightingDetector(),
            'gaze': None,      # Placeholder for teammate's module
            'posture': None,   # Placeholder for teammate's module
        }
        
        # Status tracking
        self.current_status = {
            'lighting': True,
            'gaze': True,
            'posture': True,
        }
        
        # Notifications queue (most recent 5)
        self.notifications = []
        self.max_notifications = 5
        
        # Show welcome screen
        self.show_welcome_screen()
    
    def show_welcome_screen(self):
        """Display the welcome/start screen."""
        # Clear root
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Main container
        container = tk.Frame(self.root, bg="#1a1a2e")
        container.pack(fill=tk.BOTH, expand=True)
        
        # Background (you can add an image here later)
        canvas = tk.Canvas(container, bg="#1a1a2e", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create a gradient-like effect with rectangles
        for i in range(100):
            color_val = int(26 + i * 0.3)
            b = int(min(46 + i * 0.5, 80))
            color = f"#{color_val:02x}{color_val:02x}{b:02x}"
            canvas.create_rectangle(0, i*8, 1200, (i+1)*8, fill=color, outline="")
        
        # Center frame
        center_frame = tk.Frame(canvas, bg="#16213e", padx=60, pady=50)
        center_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Title
        title_label = tk.Label(
            center_frame,
            text="FocusGuard",
            font=("Helvetica", 48, "bold"),
            fg="#00d4ff",
            bg="#16213e"
        )
        title_label.pack(pady=(0, 10))
        
        # Subtitle
        subtitle = tk.Label(
            center_frame,
            text="Your AI-Powered Workspace Assistant",
            font=("Helvetica", 16),
            fg="#e0e0e0",
            bg="#16213e"
        )
        subtitle.pack(pady=(0, 30))
        
        # Welcome message
        welcome_text = (
            "Hello, welcome to your new focus session!\n\n"
            "Please input the desired length in minutes of the focus session\n"
            "and break length between sessions."
        )
        welcome_label = tk.Label(
            center_frame,
            text=welcome_text,
            font=("Helvetica", 14),
            fg="#ffffff",
            bg="#16213e",
            justify=tk.CENTER
        )
        welcome_label.pack(pady=(0, 30))
        
        # Input frame
        input_frame = tk.Frame(center_frame, bg="#16213e")
        input_frame.pack(pady=(0, 30))
        
        # Focus duration input
        focus_frame = tk.Frame(input_frame, bg="#16213e")
        focus_frame.pack(pady=10)
        
        tk.Label(
            focus_frame,
            text="Focus Duration (minutes):",
            font=("Helvetica", 12),
            fg="#ffffff",
            bg="#16213e"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.focus_entry = tk.Entry(
            focus_frame,
            font=("Helvetica", 12),
            bg="#0f3460",
            fg="#ffffff",
            insertbackground="#ffffff",
            width=10,
            relief=tk.FLAT,
            borderwidth=2
        )
        self.focus_entry.pack(side=tk.LEFT)
        self.focus_entry.insert(0, "25")  # Default Pomodoro
        
        # Break duration input
        break_frame = tk.Frame(input_frame, bg="#16213e")
        break_frame.pack(pady=10)
        
        tk.Label(
            break_frame,
            text="Break Duration (minutes):",
            font=("Helvetica", 12),
            fg="#ffffff",
            bg="#16213e"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.break_entry = tk.Entry(
            break_frame,
            font=("Helvetica", 12),
            bg="#0f3460",
            fg="#ffffff",
            insertbackground="#ffffff",
            width=10,
            relief=tk.FLAT,
            borderwidth=2
        )
        self.break_entry.pack(side=tk.LEFT)
        self.break_entry.insert(0, "5")  # Default Pomodoro
        
        # Buttons frame
        button_frame = tk.Frame(center_frame, bg="#16213e")
        button_frame.pack(pady=(0, 0))
        
        # Continue button
        self.continue_btn = tk.Button(
            button_frame,
            text="Continue",
            font=("Helvetica", 14, "bold"),
            bg="#00d4ff",
            fg="#1a1a2e",
            activebackground="#00b8e6",
            activeforeground="#1a1a2e",
            relief=tk.FLAT,
            padx=40,
            pady=15,
            cursor="hand2",
            command=self.start_monitoring
        )
        self.continue_btn.pack(side=tk.LEFT, padx=10)
        
        # Skip button
        skip_btn = tk.Button(
            button_frame,
            text="Continue Without Time Limits",
            font=("Helvetica", 12),
            bg="#533483",
            fg="#ffffff",
            activebackground="#442870",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=30,
            pady=15,
            cursor="hand2",
            command=self.start_monitoring_no_limits
        )
        skip_btn.pack(side=tk.LEFT, padx=10)
        
        # Bind entry change to update button text
        self.focus_entry.bind('<KeyRelease>', self.update_continue_button)
        self.break_entry.bind('<KeyRelease>', self.update_continue_button)
    
    def update_continue_button(self, event=None):
        """Update continue button text based on input."""
        focus = self.focus_entry.get().strip()
        if focus:
            self.continue_btn.config(text="Continue")
        else:
            self.continue_btn.config(text="Continue Without Time Limits")
    
    def start_monitoring_no_limits(self):
        """Start monitoring without time limits."""
        self.focus_duration = None
        self.break_duration = None
        self.show_monitoring_screen()
    
    def start_monitoring(self):
        """Start monitoring with configured time limits."""
        try:
            focus = self.focus_entry.get().strip()
            break_time = self.break_entry.get().strip()
            
            if focus:
                self.focus_duration = int(focus)
                if self.focus_duration <= 0:
                    raise ValueError("Focus duration must be positive")
            else:
                self.focus_duration = None
            
            if break_time:
                self.break_duration = int(break_time)
                if self.break_duration <= 0:
                    raise ValueError("Break duration must be positive")
            else:
                self.break_duration = None
            
            self.show_monitoring_screen()
            
        except ValueError as e:
            messagebox.showerror(
                "Invalid Input",
                "Please enter valid positive numbers for duration."
            )
    
    def show_monitoring_screen(self):
        """Display the main monitoring interface."""
        # Clear root
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Main container
        main_container = tk.Frame(self.root, bg="#1a1a2e")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Top bar
        self.create_top_bar(main_container)
        
        # Content area (camera + notifications)
        content_frame = tk.Frame(main_container, bg="#1a1a2e")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left side - Camera feed
        camera_container = tk.Frame(content_frame, bg="#16213e", relief=tk.SOLID, borderwidth=2)
        camera_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Camera label
        self.camera_label = tk.Label(
            camera_container,
            bg="#000000",
            text="Initializing camera...",
            fg="#ffffff",
            font=("Helvetica", 14)
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right side - Notifications
        notif_container = tk.Frame(content_frame, bg="#16213e", relief=tk.SOLID, borderwidth=2, width=350)
        notif_container.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        notif_container.pack_propagate(False)
        
        # Notifications header
        notif_header = tk.Label(
            notif_container,
            text="Notifications",
            font=("Helvetica", 16, "bold"),
            fg="#00d4ff",
            bg="#16213e"
        )
        notif_header.pack(pady=(10, 5))
        
        # Notifications area (scrollable)
        self.notif_canvas = tk.Canvas(
            notif_container,
            bg="#0f3460",
            highlightthickness=0
        )
        self.notif_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.notif_frame = tk.Frame(self.notif_canvas, bg="#0f3460")
        self.notif_canvas.create_window((0, 0), window=self.notif_frame, anchor=tk.NW)
        
        # Bottom bar - Timer and status icons
        self.create_bottom_bar(main_container)
        
        # Start camera and monitoring
        self.session_start_time = time.time()
        self.start_camera()
        self.update_camera_feed()
        self.update_timer()
    
    def create_top_bar(self, parent):
        """Create the top navigation bar."""
        top_bar = tk.Frame(parent, bg="#16213e", height=60)
        top_bar.pack(fill=tk.X, padx=20, pady=(20, 10))
        top_bar.pack_propagate(False)
        
        # Logo/Title
        title = tk.Label(
            top_bar,
            text="FocusGuard",
            font=("Helvetica", 24, "bold"),
            fg="#00d4ff",
            bg="#16213e"
        )
        title.pack(side=tk.LEFT, padx=20)
        
        # Session info
        session_info = ""
        if self.focus_duration:
            session_info = f"Focus: {self.focus_duration}min"
            if self.break_duration:
                session_info += f" | Break: {self.break_duration}min"
        else:
            session_info = "No Time Limits"
        
        info_label = tk.Label(
            top_bar,
            text=session_info,
            font=("Helvetica", 12),
            fg="#e0e0e0",
            bg="#16213e"
        )
        info_label.pack(side=tk.LEFT, padx=20)
        
        # Stop button
        stop_btn = tk.Button(
            top_bar,
            text="Stop Session",
            font=("Helvetica", 11, "bold"),
            bg="#e94560",
            fg="#ffffff",
            activebackground="#d63850",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2",
            command=self.stop_session
        )
        stop_btn.pack(side=tk.RIGHT, padx=20)
    
    def create_bottom_bar(self, parent):
        """Create the bottom status bar with timer and icons."""
        bottom_bar = tk.Frame(parent, bg="#16213e", height=80)
        bottom_bar.pack(fill=tk.X, padx=20, pady=(10, 20))
        bottom_bar.pack_propagate(False)
        
        # Left side - Timer
        timer_frame = tk.Frame(bottom_bar, bg="#16213e")
        timer_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        tk.Label(
            timer_frame,
            text="Focus Time:",
            font=("Helvetica", 12),
            fg="#e0e0e0",
            bg="#16213e"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.timer_label = tk.Label(
            timer_frame,
            text="00:00:00",
            font=("Helvetica", 24, "bold"),
            fg="#00d4ff",
            bg="#16213e"
        )
        self.timer_label.pack(side=tk.LEFT)
        
        # Right side - Status icons
        icons_frame = tk.Frame(bottom_bar, bg="#16213e")
        icons_frame.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Create status indicators
        self.status_indicators = {}
        
        # Posture indicator
        posture_frame = self.create_status_indicator(
            icons_frame, "Posture", "posture"
        )
        posture_frame.pack(side=tk.LEFT, padx=15)
        
        # Gaze indicator
        gaze_frame = self.create_status_indicator(
            icons_frame, "Focus", "gaze"
        )
        gaze_frame.pack(side=tk.LEFT, padx=15)
        
        # Lighting indicator
        lighting_frame = self.create_status_indicator(
            icons_frame, "Lighting", "lighting"
        )
        lighting_frame.pack(side=tk.LEFT, padx=15)
    
    def create_status_indicator(self, parent, label_text, status_key):
        """Create a status indicator with icon and label."""
        container = tk.Frame(parent, bg="#16213e")
        
        # Status circle (will change color)
        canvas = tk.Canvas(
            container,
            width=50,
            height=50,
            bg="#16213e",
            highlightthickness=0
        )
        canvas.pack()
        
        # Draw circle (green by default)
        circle = canvas.create_oval(
            5, 5, 45, 45,
            fill="#4ecca3",
            outline="#4ecca3",
            width=3
        )
        
        # Store references
        self.status_indicators[status_key] = {
            'canvas': canvas,
            'circle': circle,
            'status': True
        }
        
        # Label
        label = tk.Label(
            container,
            text=label_text,
            font=("Helvetica", 10),
            fg="#e0e0e0",
            bg="#16213e"
        )
        label.pack(pady=(5, 0))
        
        return container
    
    def update_status_indicator(self, status_key, is_ok):
        """Update a status indicator's appearance."""
        if status_key not in self.status_indicators:
            return
        
        indicator = self.status_indicators[status_key]
        canvas = indicator['canvas']
        circle = indicator['circle']
        
        # Update color based on status
        color = "#4ecca3" if is_ok else "#e94560"  # Green or Red
        canvas.itemconfig(circle, fill=color, outline=color)
        
        indicator['status'] = is_ok
    
    def add_notification(self, message, module_name, is_warning=True):
        """Add a notification to the notifications panel."""
        # Add to queue
        timestamp = time.strftime("%H:%M:%S")
        self.notifications.insert(0, {
            'message': message,
            'module': module_name,
            'time': timestamp,
            'warning': is_warning
        })
        
        # Keep only recent notifications
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[:self.max_notifications]
        
        # Update UI
        self.refresh_notifications()
    
    def refresh_notifications(self):
        """Refresh the notifications display."""
        # Clear existing
        for widget in self.notif_frame.winfo_children():
            widget.destroy()
        
        # Add notifications
        for notif in self.notifications:
            self.create_notification_card(
                self.notif_frame,
                notif['message'],
                notif['module'],
                notif['time'],
                notif['warning']
            )
        
        # Update scroll region
        self.notif_frame.update_idletasks()
        self.notif_canvas.config(scrollregion=self.notif_canvas.bbox("all"))
    
    def create_notification_card(self, parent, message, module, timestamp, is_warning):
        """Create a notification card."""
        # Card container
        bg_color = "#533483" if is_warning else "#0f3460"
        card = tk.Frame(parent, bg=bg_color, relief=tk.SOLID, borderwidth=1)
        card.pack(fill=tk.X, padx=5, pady=5)
        
        # Header with module and time
        header = tk.Frame(card, bg=bg_color)
        header.pack(fill=tk.X, padx=10, pady=(8, 5))
        
        tk.Label(
            header,
            text=module.upper(),
            font=("Helvetica", 10, "bold"),
            fg="#00d4ff",
            bg=bg_color
        ).pack(side=tk.LEFT)
        
        tk.Label(
            header,
            text=timestamp,
            font=("Helvetica", 9),
            fg="#a0a0a0",
            bg=bg_color
        ).pack(side=tk.RIGHT)
        
        # Message
        msg_label = tk.Label(
            card,
            text=message,
            font=("Helvetica", 10),
            fg="#ffffff",
            bg=bg_color,
            wraplength=300,
            justify=tk.LEFT
        )
        msg_label.pack(fill=tk.X, padx=10, pady=(0, 8))
    
    def start_camera(self):
        """Initialize and start the camera."""
        try:
            self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
            
            self.camera_running = True
            
        except Exception as e:
            messagebox.showerror(
                "Camera Error",
                f"Failed to start camera: {str(e)}\n\nCheck that your camera is connected and not in use."
            )
            self.camera_running = False
    
    def update_camera_feed(self):
        """Update the camera feed and run detections."""
        if not self.camera_running or not self.camera:
            return
        
        ret, frame = self.camera.read()
        
        if ret:
            # Run detections
            self.run_detections(frame)
            
            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            
            # Convert to PhotoImage
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk, text="")
        
        # Schedule next update
        self.root.after(33, self.update_camera_feed)  # ~30 FPS
    
    def run_detections(self, frame):
        """Run all available detectors on the frame."""
        # Lighting detection
        if self.detectors['lighting']:
            try:
                result = self.detectors['lighting'].analyze(frame)
                self.process_detection_result('lighting', result)
            except Exception as e:
                print(f"Lighting detection error: {e}")
        
        # Gaze detection (when available)
        if self.detectors['gaze']:
            try:
                result = self.detectors['gaze'].analyze(frame)
                self.process_detection_result('gaze', result)
            except Exception as e:
                print(f"Gaze detection error: {e}")
        
        # Posture detection (when available)
        if self.detectors['posture']:
            try:
                result = self.detectors['posture'].analyze(frame)
                self.process_detection_result('posture', result)
            except Exception as e:
                print(f"Posture detection error: {e}")
    
    def process_detection_result(self, module_name, result: DetectionResult):
        """Process a detection result and update UI."""
        # Update status indicator
        self.update_status_indicator(module_name, result.is_ok)
        
        # Add notification if warning
        if not result.is_ok and result.warning_message:
            # Only add if status changed (to avoid spam)
            if self.current_status.get(module_name, True):
                self.add_notification(
                    result.warning_message,
                    module_name,
                    is_warning=True
                )
        
        # Update current status
        self.current_status[module_name] = result.is_ok
    
    def update_timer(self):
        """Update the focus timer display."""
        if self.session_start_time:
            elapsed = time.time() - self.session_start_time
            
            # Format time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.timer_label.config(text=time_str)
            
            # Check if focus session is complete
            if self.focus_duration:
                if elapsed >= (self.focus_duration * 60):
                    self.show_break_reminder()
                    return
        
        # Schedule next update
        self.root.after(1000, self.update_timer)
    
    def show_break_reminder(self):
        """Show break reminder when focus session is complete."""
        response = messagebox.askyesno(
            "Focus Session Complete!",
            f"Great work! You've completed {self.focus_duration} minutes of focused work.\n\n"
            f"Would you like to take a {self.break_duration}-minute break?"
        )
        
        if response:
            # TODO: Implement break timer
            messagebox.showinfo("Break Time", "Enjoy your break!")
            self.stop_session()
        else:
            # Reset timer for another session
            self.session_start_time = time.time()
            self.update_timer()
    
    def stop_session(self):
        """Stop the monitoring session and return to welcome screen."""
        if self.camera:
            self.camera.release()
        self.camera_running = False
        
        # Show summary
        if self.session_start_time:
            elapsed = time.time() - self.session_start_time
            minutes = int(elapsed // 60)
            messagebox.showinfo(
                "Session Complete",
                f"Session ended after {minutes} minutes.\n\nGreat work!"
            )
        
        self.show_welcome_screen()
    
    def on_closing(self):
        """Handle window close event."""
        if self.camera:
            self.camera.release()
        self.root.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = FocusGuardApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()