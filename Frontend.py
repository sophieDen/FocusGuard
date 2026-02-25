"""
FocusGuard Frontend Application — Sokcho Editorial Design

Soft pastel aesthetic inspired by Korean editorial print design:
- Blush × sky-blue gradient background with translucent orb blobs
- Text floats directly on gradient (NO background boxes on welcome screen)
- Frosted-glass style cards only for the monitoring interface
- Refined serif + light sans-serif typography
- Minimal, airy layout with decorative ripple lines
- Muted teal/rose status indicators
"""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import cv2
import time
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.base_detector import DetectionResult
from modules.lighting.lighting_detector import LightingDetector

# from modules.gaze.gaze_detector import GazeDetector
# from modules.posture.posture_detector import PostureDetector
import config


# Palette 
C_BG_TOP    = (9, 31, 91)
C_BG_MID    = (237, 240, 245)
C_BG_BOT    = (208, 228, 255)
C_ORB1      = (245, 208, 204)
C_ORB2      = (200, 223, 245)
C_DEEP      = "#3a4a6b"
C_MUTED     = "#8a9ab8"
C_ACCENT    = "#9b8aa8"
C_OK        = "#7abfaa"
C_WARN      = "#c97a7a"
C_WHITE     = "#fafcff"
C_BORDER    = "#c8d4e8"


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def make_gradient_image(w, h):
    """Create a soft 3-stop vertical gradient with blush × sky tones."""
    img = Image.new("RGB", (w, h))
    pixels = img.load()
    for y in range(h):
        t = y / h
        if t < 0.5:
            c = lerp_color(C_BG_TOP, C_BG_MID, t * 2)
        else:
            c = lerp_color(C_BG_MID, C_BG_BOT, (t - 0.5) * 2)
        for x in range(w):
            pixels[x, y] = c
    return img


def draw_orbs(img):
    """Paint soft translucent orbs onto the background image."""
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def soft_orb(cx, cy, r, color_rgb, alpha_max=130):
        for step in range(12):
            frac = step / 12
            current_r = int(r * (1 - frac * 0.6))
            alpha = int(alpha_max * (1 - frac) * (1 - frac))
            rgba = (*color_rgb, alpha)
            draw.ellipse(
                [cx - current_r, cy - current_r, cx + current_r, cy + current_r],
                fill=rgba
            )

    # Large rose orb — top-left
    soft_orb(-40, -40, 340, C_ORB1, alpha_max=120)
    soft_orb(-40, -40, 160, (255, 248, 246), alpha_max=80)

    # Large sky orb — bottom-right
    soft_orb(w + 30, h + 30, 360, C_ORB2, alpha_max=120)
    soft_orb(w + 30, h + 30, 170, (240, 250, 255), alpha_max=80)

    # Small accent orb — right-center
    soft_orb(int(w * 0.82), int(h * 0.42), 160, (230, 210, 240), alpha_max=80)

    blurred = overlay.filter(ImageFilter.GaussianBlur(radius=28))
    base = img.convert("RGBA")
    base.alpha_composite(blurred)
    return base.convert("RGB")


def make_bg(w, h):
    img = make_gradient_image(w, h)
    img = draw_orbs(img)
    return img


# Main App 
class FocusGuardApp:
    """FocusGuard with Sokcho editorial aesthetic."""

    def __init__(self, root):
        self.root = root
        self.root.title("FocusGuard")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        self.root.configure(bg=C_DEEP)

        self.focus_duration = None
        self.break_duration = None
        self.session_start_time = None

        self.camera = None
        self.camera_running = False

        self.detectors = {
            'lighting': LightingDetector(),
            'gaze': None,      # placeholders
            'posture': None,   # placeholders
        }
        self.current_status = {'lighting': True, 'gaze': True, 'posture': True}
        self.notifications = []
        self.max_notifications = 5

        self.show_welcome_screen()

    # helpers 

    def _clear(self):
        for w in self.root.winfo_children():
            w.destroy()

    def _bg_canvas(self, parent):
        """Return a canvas pre-filled with the gradient+orb background."""
        canvas = tk.Canvas(parent, highlightthickness=0, bd=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        def _repaint(event=None):
            cw, ch = canvas.winfo_width(), canvas.winfo_height()
            if cw < 2 or ch < 2:
                return
            img = make_bg(cw, ch)
            photo = ImageTk.PhotoImage(img)
            canvas._bg_photo = photo
            canvas.delete("bg")
            canvas.create_image(0, 0, anchor=tk.NW, image=photo, tags="bg")
            canvas.tag_lower("bg")

        canvas.bind("<Configure>", _repaint)
        canvas.after(50, _repaint)
        return canvas

    def _entry(self, parent, default="", width=6):
        e = tk.Entry(
            parent,
            font=("Georgia", 26),
            fg=C_DEEP,
            bg="#f4f7fd",
            insertbackground=C_DEEP,
            relief=tk.FLAT,
            width=width,
            bd=0,
            highlightthickness=0,
            justify=tk.CENTER,
        )
        e.insert(0, default)
        return e

    def _btn_primary(self, parent, text, cmd, width=22):
        return tk.Button(
            parent,
            text=text,
            command=cmd,
            font=("Helvetica", 10, "bold"),
            fg=C_WHITE,
            bg=C_DEEP,

            activebackground=C_DEEP,   # IMPORTANT
            activeforeground=C_WHITE,

            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,     
            highlightbackground=C_DEEP, 
            highlightcolor=C_DEEP,      

            width=width,
            pady=12,
            cursor="hand2"
        )

    def _btn_ghost(self, parent, text, cmd, width=22):
        return tk.Button(
            parent, text=text, command=cmd,
            font=("Helvetica", 9),
            fg=C_MUTED, bg="#dde5f0",
            activebackground="#c8d4e8", activeforeground=C_DEEP,
            relief=tk.FLAT, bd=0,
            width=width, pady=11,
            cursor="hand2",
        )

    # Welcome Screen

    def show_welcome_screen(self):
        self._clear()

        outer = tk.Frame(self.root, bg="#d8e4f2")
        outer.pack(fill=tk.BOTH, expand=True)

        canvas = self._bg_canvas(outer)

        # Left text: editorial branding
        
        canvas.create_text(310, 280, text="Focus", 
                          font=("Georgia", 52, "italic"), fill=C_DEEP)
        
        canvas.create_text(310, 335, text="Guard", 
                          font=("Georgia", 52), fill=C_DEEP)
        
        
        # Divider line
        canvas.create_line(290, 400, 330, 400, fill=C_BORDER, width=1)
        
        # Description (multi-line)
        desc_lines = [
            "Welcome to your focus session.",
            "",
            "FocusGuard monitors your posture,",
            "gaze, and lighting — gently keeping",
            "you at your best."
        ]
        y = 435
        for line in desc_lines:
            canvas.create_text(310, y, text=line, 
                             font=("Helvetica", 14), fill=C_MUTED)
            y += 22

        # Right panel: config card
        right = tk.Frame(canvas, bg="#eef2fa", bd=0,
                         highlightthickness=1, highlightbackground=C_BORDER)
        canvas.create_window(860, 400, window=right, width=360, height=320, anchor=tk.CENTER)

        tk.Label(right, text="Begin a session",
                 font=("Georgia", 20), fg=C_DEEP, bg="#eef2fa").pack(pady=(34, 2))
        tk.Label(right, text="CONFIGURE YOUR FOCUS CYCLE",
                 font=("Helvetica", 8), fg=C_MUTED, bg="#eef2fa").pack(pady=(0, 24))

        # Fields grid
        fields = tk.Frame(right, bg="#eef2fa")
        fields.pack(padx=40, pady=(0, 24))

        # Focus field
        focus_col = tk.Frame(fields, bg="#eef2fa")
        focus_col.grid(row=0, column=0, padx=16)
        tk.Label(focus_col, text="FOCUS", font=("Helvetica", 8),
                 fg=C_MUTED, bg="#eef2fa").pack()
        self.focus_entry = self._entry(focus_col, "25", width=4)
        self.focus_entry.pack(pady=6)
        tk.Frame(focus_col, bg=C_BORDER, height=1).pack(fill=tk.X)
        tk.Label(focus_col, text="min", font=("Helvetica", 9),
                 fg=C_MUTED, bg="#eef2fa").pack(pady=(4, 0))

        # Separator dot
        tk.Label(fields, text="·", font=("Helvetica", 24),
                 fg=C_BORDER, bg="#eef2fa").grid(row=0, column=1, padx=8, pady=18)

        # Break field
        break_col = tk.Frame(fields, bg="#eef2fa")
        break_col.grid(row=0, column=2, padx=16)
        tk.Label(break_col, text="BREAK", font=("Helvetica", 8),
                 fg=C_MUTED, bg="#eef2fa").pack()
        self.break_entry = self._entry(break_col, "5", width=4)
        self.break_entry.pack(pady=6)
        tk.Frame(break_col, bg=C_BORDER, height=1).pack(fill=tk.X)
        tk.Label(break_col, text="min", font=("Helvetica", 9),
                 fg=C_MUTED, bg="#eef2fa").pack(pady=(4, 0))

        # Buttons
        btn_area = tk.Frame(right, bg="#eef2fa")
        btn_area.pack(padx=36, fill=tk.X)

        self._btn_ghost(btn_area, "BEGIN FOCUS SESSION", self.start_monitoring, width=28).pack(fill=tk.X, pady=(0, 8))

    # Session start logic

    def start_monitoring(self):
        try:
            focus = self.focus_entry.get().strip()
            brk = self.break_entry.get().strip()
            self.focus_duration = int(focus) if focus else None
            self.break_duration = int(brk) if brk else None
            if self.focus_duration and self.focus_duration <= 0:
                raise ValueError
            if self.break_duration and self.break_duration <= 0:
                raise ValueError
            self.show_monitoring_screen()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid positive numbers.")

    # Monitoring Screen

    def show_monitoring_screen(self):
        self._clear()

        outer = tk.Frame(self.root, bg="#d8e4f2")
        outer.pack(fill=tk.BOTH, expand=True)

        # Top bar
        top = tk.Frame(outer, bg="#eef2fa", height=56,
                       highlightthickness=1, highlightbackground=C_BORDER)
        top.pack(fill=tk.X, padx=0, pady=0)
        top.pack_propagate(False)

        tk.Label(top, text="Focus", font=("Georgia", 18, "italic"),
                 fg=C_ACCENT, bg="#eef2fa").pack(side=tk.LEFT, padx=(24, 0), pady=12)
        tk.Label(top, text="Guard", font=("Georgia", 18),
                 fg=C_DEEP, bg="#eef2fa").pack(side=tk.LEFT, padx=(2, 16), pady=12)

        if self.focus_duration:
            info = f"Focus  {self.focus_duration} min"
            if self.break_duration:
                info += f"   ·   Break  {self.break_duration} min"
        else:
            info = "No Time Limits"
        tk.Label(top, text=info, font=("Helvetica", 9), fg=C_MUTED, bg="#eef2fa").pack(side=tk.LEFT)

        stop_btn = tk.Button(
            top, text="END SESSION",
            font=("Helvetica", 8, "bold"),
            fg=C_WHITE, bg=C_WARN,
            activebackground="#b06060", activeforeground=C_WHITE,
            relief=tk.FLAT, padx=18, pady=8, cursor="hand2",
            command=self.stop_session
        )
        stop_btn.pack(side=tk.RIGHT, padx=20, pady=10)

        # Content area
        content = tk.Frame(outer, bg="#dce8f4")
        content.pack(fill=tk.BOTH, expand=True, padx=18, pady=10)

        # Camera pane
        cam_wrap = tk.Frame(content, bg="#eef2fa",
                            highlightthickness=1, highlightbackground=C_BORDER)
        cam_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 9))

        cam_header = tk.Frame(cam_wrap, bg="#eef2fa", height=36)
        cam_header.pack(fill=tk.X)
        cam_header.pack_propagate(False)
        tk.Label(cam_header, text="LIVE VIEW", font=("Helvetica", 8),
                 fg=C_MUTED, bg="#eef2fa").pack(side=tk.LEFT, padx=14, pady=10)

        self.camera_label = tk.Label(cam_wrap, bg="#cdd8eb",
                                     text="Initializing camera…",
                                     fg=C_MUTED, font=("Helvetica", 11))
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Notifications pane
        notif_wrap = tk.Frame(content, bg="#eef2fa", width=300,
                              highlightthickness=1, highlightbackground=C_BORDER)
        notif_wrap.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(9, 0))
        notif_wrap.pack_propagate(False)

        tk.Label(notif_wrap, text="NOTIFICATIONS", font=("Helvetica", 8),
                 fg=C_MUTED, bg="#eef2fa").pack(pady=(14, 4))
        tk.Frame(notif_wrap, bg=C_BORDER, height=1).pack(fill=tk.X, padx=14)

        self.notif_canvas = tk.Canvas(notif_wrap, bg="#eef2fa", highlightthickness=0)
        self.notif_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.notif_frame = tk.Frame(self.notif_canvas, bg="#eef2fa")
        self.notif_canvas.create_window((0, 0), window=self.notif_frame, anchor=tk.NW)

        # Bottom bar
        bot = tk.Frame(outer, bg="#eef2fa", height=72,
                       highlightthickness=1, highlightbackground=C_BORDER)
        bot.pack(fill=tk.X, padx=0, pady=0, side=tk.BOTTOM)
        bot.pack_propagate(False)

        # Timer
        timer_f = tk.Frame(bot, bg="#eef2fa")
        timer_f.pack(side=tk.LEFT, padx=24, pady=12)
        tk.Label(timer_f, text="FOCUS TIME", font=("Helvetica", 8),
                 fg=C_MUTED, bg="#eef2fa").pack(anchor=tk.W)
        self.timer_label = tk.Label(timer_f, text="00:00:00",
                                    font=("Georgia", 26),
                                    fg=C_DEEP, bg="#eef2fa")
        self.timer_label.pack()

        # Status indicators
        status_f = tk.Frame(bot, bg="#eef2fa")
        status_f.pack(side=tk.RIGHT, padx=28, pady=14)
        self.status_indicators = {}
        for key, lbl in [("posture", "Posture"), ("gaze", "Focus"), ("lighting", "Lighting")]:
            self._make_status_pill(status_f, lbl, key).pack(side=tk.LEFT, padx=12)

        # Start systems
        self.session_start_time = time.time()
        self.start_camera()
        self.update_camera_feed()
        self.update_timer()

    def _make_status_pill(self, parent, label_text, key):
        """Create an elegant pill-shaped status indicator."""
        frame = tk.Frame(parent, bg="#eef2fa")

        dot_canvas = tk.Canvas(frame, width=44, height=44,
                               bg="#eef2fa", highlightthickness=0)
        dot_canvas.pack()

        # Outer ring
        ring = dot_canvas.create_oval(3, 3, 41, 41,
                                      outline=C_BORDER, width=1, fill="#dde5f0")
        # Inner filled dot
        dot = dot_canvas.create_oval(11, 11, 33, 33,
                                     fill=C_OK, outline="", width=0)

        self.status_indicators[key] = {
            'canvas': dot_canvas, 'dot': dot, 'ring': ring, 'status': True
        }

        tk.Label(frame, text=label_text.upper(), font=("Helvetica", 8),
                 fg=C_MUTED, bg="#eef2fa").pack(pady=(3, 0))
        return frame

    def update_status_indicator(self, key, is_ok):
        if key not in self.status_indicators:
            return
        ind = self.status_indicators[key]
        color = C_OK if is_ok else C_WARN
        ind['canvas'].itemconfig(ind['dot'], fill=color)
        ind['status'] = is_ok

    # Notifications 

    def add_notification(self, message, module_name, is_warning=True):
        ts = time.strftime("%H:%M:%S")
        self.notifications.insert(0, {
            'message': message, 'module': module_name,
            'time': ts, 'warning': is_warning
        })
        self.notifications = self.notifications[:self.max_notifications]
        self.refresh_notifications()

    def refresh_notifications(self):
        for w in self.notif_frame.winfo_children():
            w.destroy()
        for n in self.notifications:
            self._notif_card(self.notif_frame, n['message'],
                             n['module'], n['time'], n['warning'])
        self.notif_frame.update_idletasks()
        self.notif_canvas.config(scrollregion=self.notif_canvas.bbox("all"))

    def _notif_card(self, parent, message, module, ts, is_warning):
        bg = "#f0e8ec" if is_warning else "#e8eef8"
        border = C_WARN if is_warning else C_BORDER

        card = tk.Frame(parent, bg=bg, highlightthickness=1,
                        highlightbackground=border)
        card.pack(fill=tk.X, padx=6, pady=5)

        hdr = tk.Frame(card, bg=bg)
        hdr.pack(fill=tk.X, padx=10, pady=(8, 4))
        tk.Label(hdr, text=module.upper(), font=("Helvetica", 8, "bold"),
                 fg=C_ACCENT, bg=bg).pack(side=tk.LEFT)
        tk.Label(hdr, text=ts, font=("Helvetica", 8),
                 fg=C_MUTED, bg=bg).pack(side=tk.RIGHT)

        tk.Label(card, text=message, font=("Helvetica", 10), fg=C_DEEP,
                 bg=bg, wraplength=258, justify=tk.LEFT).pack(
            padx=10, pady=(0, 8), anchor=tk.W)

    # Camera ───────────────────────────────────────────────────────────────

    def start_camera(self):
        try:
            self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
            self.camera_running = True
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            self.camera_running = False

    def update_camera_feed(self):
        if not self.camera_running or not self.camera:
            return
        ret, frame = self.camera.read()
        if ret:
            self.run_detections(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk, text="")
        self.root.after(33, self.update_camera_feed)

    def run_detections(self, frame):
        for key in ('lighting', 'gaze', 'posture'):
            if self.detectors[key]:
                try:
                    result = self.detectors[key].analyze(frame)
                    self.process_detection_result(key, result)
                except Exception as e:
                    print(f"{key} error: {e}")

    def process_detection_result(self, module_name, result: DetectionResult):
        self.update_status_indicator(module_name, result.is_ok)
        if not result.is_ok and result.warning_message:
            if self.current_status.get(module_name, True):
                self.add_notification(result.warning_message, module_name, True)
        self.current_status[module_name] = result.is_ok

    # Timer

    def update_timer(self):
        if self.session_start_time:
            elapsed = time.time() - self.session_start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.timer_label.config(text=f"{h:02d}:{m:02d}:{s:02d}")
            if self.focus_duration and elapsed >= self.focus_duration * 60:
                self.show_break_reminder()
                return
        self.root.after(1000, self.update_timer)

    def show_break_reminder(self):
        resp = messagebox.askyesno(
            "Focus Session Complete",
            f"You've completed {self.focus_duration} minutes.\n\n"
            f"Take a {self.break_duration}-minute break?"
        )
        if resp:
            messagebox.showinfo("Break Time", "Enjoy your break!")
            self.stop_session()
        else:
            self.session_start_time = time.time()
            self.update_timer()

    def stop_session(self):
        if self.camera:
            self.camera.release()
        self.camera_running = False
        if self.session_start_time:
            mins = int((time.time() - self.session_start_time) // 60)
            messagebox.showinfo("Session Complete",
                                f"Session ended after {mins} minutes. Great work!")
        self.show_welcome_screen()

    def on_closing(self):
        if self.camera:
            self.camera.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FocusGuardApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()