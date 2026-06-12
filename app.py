import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import torch

class RasterizerApp:
    def __init__(self, root, engine, input_manager):
        self.root = root
        self.engine = engine
        self.input_manager = input_manager
        
        self.root.title("3D PyTorch Rasterizer")
        self.setup_ui()
        self.render_frame()
        self.update_loop()

    def setup_ui(self):
        top_row_frame = tk.Frame(self.root)
        top_row_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=10)

        self.image_label = tk.Label(top_row_frame)
        self.image_label.pack(side=tk.LEFT)

        info_panel = tk.Frame(top_row_frame, padx=20)
        info_panel.pack(side=tk.LEFT, fill=tk.Y, anchor="n")

        self.lbl_fov = tk.Label(info_panel, justify="left", anchor="w")
        self.lbl_fov.pack(anchor="w", pady=6)
        self.lbl_cam_pos = tk.Label(info_panel, justify="left", anchor="w")
        self.lbl_cam_pos.pack(anchor="w", pady=6)
        self.lbl_cam_rot = tk.Label(info_panel, justify="left", anchor="w")
        self.lbl_cam_rot.pack(anchor="w", pady=6)
        self.lbl_light_source_rot = tk.Label(info_panel, justify="left", anchor="w")
        self.lbl_light_source_rot.pack(anchor="w", pady=6)

        # Build Input Grid for Surface Insertion
        grid_frame = tk.Frame(info_panel)
        grid_frame.pack(pady=(130, 5))

        self.entries = []
        labels = ["x", "y", "z"]
        for r in range(3):
            row_frame = tk.Frame(grid_frame)
            row_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
            row_entries = []
            for c in range(3):
                entry = tk.Entry(row_frame, width=8, justify="center")
                entry.pack(side=tk.LEFT, padx=2)
                self.add_placeholder(entry, labels[c])
                row_entries.append(entry)
            self.entries.append(row_entries)

        tk.Button(info_panel, text="Add Surface", command=self.add_surface).pack(pady=5, fill=tk.X)
        tk.Button(info_panel, text="Cancel", command=self.cancel_writing).pack(pady=5, fill=tk.X)

    def add_placeholder(self, entry, placeholder_text):
        entry.insert(0, placeholder_text)
        entry.config(fg="gray")
        entry.is_placeholder = True
        entry.bind("<FocusIn>", lambda e: self.on_focus_in(entry))
        entry.bind("<FocusOut>", lambda e: self.on_focus_out(entry, placeholder_text))

    def on_focus_in(self, entry):
        if entry.is_placeholder:
            entry.delete(0, tk.END)
            entry.config(fg="black")
            entry.is_placeholder = False

    def on_focus_out(self, entry, text):
        if not entry.get().strip():
            entry.insert(0, text)
            entry.config(fg="gray")
            entry.is_placeholder = True

    def add_surface(self):
        self.root.focus_set()
        try:
            matrix_data = [[float(self.entries[r][c].get().strip()) for c in range(3)] for r in range(3)]
            self.engine.surfaces[self.engine.surfaces_count, :, :] = torch.tensor(matrix_data, dtype=torch.float32)
            self.engine.surfaces_count += 1
            self.render_frame()
            self.cancel_writing()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please ensure all fields contain valid numbers.")

    def cancel_writing(self):
        self.root.focus_set()
        labels = ["x", "y", "z"]
        for r in range(3):
            for c in range(3):
                entry = self.entries[r][c]
                entry.delete(0, tk.END)
                entry.insert(0, labels[c])
                entry.config(fg="gray")
                entry.is_placeholder = True

    def render_frame(self):
        self.engine.update_logic(self.input_manager)
        grid_4d = self.engine.grid.unsqueeze(0).unsqueeze(0)
        upscaled_grid = torch.nn.functional.interpolate(
            grid_4d, size=(self.engine.screen_size * 2, self.engine.screen_size * 2), mode='nearest'
        ).squeeze()
        
        shade_pixels = (upscaled_grid * 255).to(torch.uint8).cpu().numpy()
        new_img = ImageTk.PhotoImage(image=Image.fromarray(shade_pixels, mode='L'))
        self.image_label.config(image=new_img) 
        self.image_label.image = new_img

        c_pos = self.engine.camera_point.cpu().tolist()
        c_rot = self.engine.camera_angle.cpu().tolist()
        s_rot = self.engine.light_source_angle.cpu().tolist()

        self.lbl_fov.config(text=f"FOV: {self.engine.fov:.1f}")
        self.lbl_cam_pos.config(text=f"camera position:\nx: {c_pos[0]:.2f}, y: {c_pos[1]:.2f}, z: {c_pos[2]:.2f}")
        self.lbl_cam_rot.config(text=f"camera angle:\nlat: {c_rot[0]:.1f}°, lon: {c_rot[1]:.1f}°")
        self.lbl_light_source_rot.config(text=f"light source angle:\nlat: {s_rot[0]:.1f}°, lon: {s_rot[1]:.1f}°")

    def update_loop(self):
        if len(self.input_manager.pressed_keys) > 0:
            self.render_frame()
        self.root.after(15, self.update_loop)
