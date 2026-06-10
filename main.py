import torch
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from pynput import keyboard

EPSILON = 1e-8
VELOCITY = 0.075
ANGULAR_VELOCITY = 7.5
MIN_FOV = 0
MAX_FOV = 175
SCREEN_SIZE = 250
CLIPPING_DISTANCE = 0.01

def get_cosine(a, b):
    if a.dim() == 1: a = a.unsqueeze(0)
    if b.dim() == 1: b = b.unsqueeze(0)
    a_magnitudes = torch.linalg.vector_norm(a, dim=1)
    b_magnitudes = torch.linalg.vector_norm(b, dim=1)
    dot_ab = torch.linalg.vecdot(a, b, dim=1)
    return torch.clamp(dot_ab / (a_magnitudes * b_magnitudes), -1 + EPSILON, 1 - EPSILON)

def get_angle(a, b):
    return torch.acos(get_cosine(a, b))

def get_intersection_points(p1, p2, a, b, c, d): # p1, p2 expects two tensors of shape Nx3
    n = torch.stack([a, b, c]) # shape 3
    v = p2 - p1 # shape Nx3
    denominator = (n * v).sum(dim=1)
    if torch.any(denominator.abs() < EPSILON): 
        raise RuntimeError("Line is parallell to near clipping plane. How can this be possible since we take the intersection between two points on opposite sides of the near clipping plane?")
    t = -((n * p1).sum(dim=1) + d) / denominator
    return p1 + t.unsqueeze(1) * v # will return Nx3

def get_normals(surfaces):
    ab = surfaces[:, 1, :] - surfaces[:, 0, :]
    ac = surfaces[:, 2, :] - surfaces[:, 0, :]
    return torch.linalg.cross(ab, ac, dim=1) # will return Nx3

def get_depth_planes(screen_coordinates):
    screen_normals = get_normals(screen_coordinates)
    a, b, c = screen_normals[:, 0], screen_normals[:, 1], screen_normals[:, 2]
    d = -a * screen_coordinates[:, 0, 0] -b * screen_coordinates[:, 0, 1] -c * screen_coordinates[:, 0, 2]
    return torch.stack((a, b, c, d), dim=1)

def get_general_equations(screen_coordinates):
    x = screen_coordinates[:, :, 0]
    y = screen_coordinates[:, :, 1]
    # a = A.y - B.y. A = [0, 1, 2]. B = [1, 2, 0]. C = [2, 0, 1]
    a = y[:, [0, 1, 2]] - y[:, [1, 2, 0]] # Shape Nx3
    b = x[:, [1, 2, 0]] - x[:, [0, 1, 2]] # Shape Nx3
    c = -a * x[:, [0, 1, 2]] -b * y[:, [0, 1, 2]] # Shape Nx3

    double_corners = ((x[:, [0, 1, 2]] - x[:, [1, 2, 0]]) ** 2 + (y[:, [0, 1, 2]] - y[:, [1, 2, 0]]) ** 2 < EPSILON).any(dim=1)
    substitution = a * x[:, [2, 0, 1]] + b * y[:, [2, 0, 1]] + c # shape Nx3
    flat = (torch.abs(substitution) < EPSILON).any(dim=1) # shape N
    degenerate = torch.logical_or(double_corners, flat)
    abc = torch.stack([a, b, c], dim=2) # Shape Nx3x3
    return torch.where((substitution > 0).unsqueeze(2), abc, -abc), degenerate

def get_clipped_mask_2(surfaces, a, b, c, d): # expects Nx3x3, first corner is behind near clipping plane
    new_corners_a1 = get_intersection_points(surfaces[:, 0, :], surfaces[:, 1, :], a, b, c, d)
    new_corners_a2 = get_intersection_points(surfaces[:, 0, :], surfaces[:, 2, :], a, b, c, d)
    new_surfaces_1 = torch.stack((new_corners_a1, surfaces[:, 1, :], surfaces[:, 2, :]), dim=1)
    new_surfaces_2 = torch.stack((new_corners_a1, new_corners_a2, surfaces[:, 2, :]), dim=1)
    return torch.cat((new_surfaces_1, new_surfaces_2), dim=0)
    # will return Mx3x3 where M>N
        
def get_clipped_mask_1(surfaces, a, b, c, d): # expects Nx3x3, last corner is in front of clipping plane
    new_corners_a = get_intersection_points(surfaces[:, 0, :], surfaces[:, 2, :], a, b, c, d) # Nx3
    new_corners_b = get_intersection_points(surfaces[:, 1, :], surfaces[:, 2, :], a, b, c, d) # Nx3
    return torch.stack((new_corners_a, new_corners_b, surfaces[:, 2, :]), dim=1)
    # will returgetn Nx3x3


class Engine:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise RuntimeError("No GPU detected! This engine requires a Mac (MPS) or PC (CUDA) GPU to run.")
       
        self.fov = torch.tensor(125.0, device=self.device)
        self.camera_point = torch.zeros(3, device=self.device)
        self.camera_angle = torch.zeros(2, device=self.device)
        self.light_source_angle = torch.tensor([60.0, 0.0], device=self.device)

        self.surfaces = torch.zeros((500, 3, 3), device=self.device)
        
        self.surfaces_count = 0


        self.create_sphere()

        self.grid = torch.rand((SCREEN_SIZE, SCREEN_SIZE), device=self.device)
        # dim=0 moves vertically down the rows (Height)
        # dim=1 moves horizontally across the columns (Width)

        # ±(0.5 / SCREEN_SIZE) ensures center of each pixel
        #x goes from -1 to 1
        #y goes from 1 to -1
        self.values_dim0 = torch.linspace(-1 + (0.5 / SCREEN_SIZE), 1 - (0.5 / SCREEN_SIZE), SCREEN_SIZE, device=self.device)
        self.values_dim1 = torch.linspace(1 - (0.5 / SCREEN_SIZE), -1 + (0.5 / SCREEN_SIZE), SCREEN_SIZE, device=self.device)
        self.grid_dim0, self.grid_dim1 = torch.meshgrid(self.values_dim0, self.values_dim1, indexing='xy')
        self.grid_values = torch.stack([self.grid_dim0, self.grid_dim1], dim=2)
        self.grid_values = self.grid_values #KxKx2 where K = screen_size
        
        self.update_logic()

    def create_sphere(self):
        lat_range = torch.arange(start=-3, end=3, step=1, device=self.device) # shape 6
        lon_range = torch.arange(start=0, end=12, step=1, device=self.device) # shape 12

        lat_grid, lon_grid = torch.meshgrid(lat_range, lon_range, indexing='xy') # both will have shape 6x12
        lat_lon_grid = torch.stack((lat_grid, lon_grid), dim=2) # shape 6x12x2

        grid_a = lat_lon_grid.reshape(lat_range.shape[0] * lon_range.shape[0], 2) # reshapes into 72x2
        grid_b = grid_a.clone()
        grid_b[:, 0] = grid_a[:, 0] + 1 # 72x2
        grid_c = grid_a.clone()
        grid_c[:, 1] = grid_a[:, 1] + 1
        grid_d = grid_a.clone()
        grid_d[:, 0] = grid_a[:, 0] + 1
        grid_d[:, 1] = grid_a[:, 1] + 1

        first_stacked_grid = torch.stack((grid_a, grid_b, grid_c), dim=1) #72x3x2
        second_stacked_grid = torch.stack((grid_d, grid_b, grid_c), dim=1) #72x3x2
        stacked_grid = torch.cat((first_stacked_grid, second_stacked_grid), dim=0) #144x3x2
        grid_degrees = stacked_grid * 30
        grid_rad = torch.deg2rad(grid_degrees)

        x_values = torch.cos(grid_rad[:, :, 0]) * torch.cos(grid_rad[:, :, 1]) #144x3
        y_values = torch.cos(grid_rad[:, :, 0]) * torch.sin(grid_rad[:, :, 1])
        z_values = torch.sin(grid_rad[:, :, 0])

        sphere_xyz = torch.stack((x_values, y_values, z_values), dim=2)
        self.surfaces_count = sphere_xyz.shape[0]
        self.surfaces[:self.surfaces_count] = sphere_xyz


    def get_near_clipping_plane(self):
        rad_camera_angle = torch.deg2rad(self.camera_angle)
        close_point = torch.stack([self.camera_point[0] + torch.cos(rad_camera_angle[0]) * torch.cos(rad_camera_angle[1]) * CLIPPING_DISTANCE,
                                    self.camera_point[1] + torch.cos(rad_camera_angle[0]) * torch.sin(rad_camera_angle[1]) * CLIPPING_DISTANCE,
                                    self.camera_point[2] + torch.sin(rad_camera_angle[0]) * CLIPPING_DISTANCE])
        a = torch.cos(rad_camera_angle[1]) * torch.cos(rad_camera_angle[0])
        b = torch.sin(rad_camera_angle[1]) * torch.cos(rad_camera_angle[0])
        c = torch.sin(rad_camera_angle[0])
        d = -a * close_point[0] -b * close_point[1] -c * close_point[2]
        return a, b, c, d
        
    def get_relevant_surfaces(self):
        rad_camera_angle = torch.deg2rad(self.camera_angle)
        active_surfaces = self.surfaces[:self.surfaces_count]
        double_corners = ((active_surfaces[:, [0, 1, 2], 0] - active_surfaces[:, [1, 2, 0], 0]) ** 2 + (active_surfaces[:, [0, 1, 2], 1] - active_surfaces[:, [1, 2, 0], 1]) ** 2 + (active_surfaces[:, [0, 1, 2], 2] - active_surfaces[:, [1, 2, 0], 2]) ** 2 < EPSILON).any(dim=1)
        filtered_surfaces = active_surfaces[~double_corners]

        a, b, c, d = self.get_near_clipping_plane()

        local_depths = torch.sin(rad_camera_angle[0]) * (filtered_surfaces[:, :, 2] - self.camera_point[2]) + torch.cos(rad_camera_angle[0]) * (torch.sin(rad_camera_angle[1]) * (filtered_surfaces[:, :, 1] - self.camera_point[1]) + torch.cos(rad_camera_angle[1]) * (filtered_surfaces[:, :, 0] - self.camera_point[0]))
        # Now we'll sort the points in every triangle by depth which will make it easier to clip triangles that are only partially in frame
        # Find the indices that would sort the depths, lowest depth first
        sort_indices = torch.argsort(local_depths, dim=1)
        # Create a helper array for dimension N
        batch_indices = torch.arange(filtered_surfaces.size(0), device=self.device).unsqueeze(1)
        # Apply the sorted indices to the surfaces and depths
        filtered_surfaces = filtered_surfaces[batch_indices, sort_indices, :]
        local_depths = local_depths[batch_indices, sort_indices]

        corners_in_front_count = (local_depths > CLIPPING_DISTANCE).sum(dim=1)

        mask_3_surfaces = filtered_surfaces[corners_in_front_count == 3]
        mask_2_surfaces = filtered_surfaces[corners_in_front_count == 2]
        mask_1_surfaces = filtered_surfaces[corners_in_front_count == 1]
        clipped_mask_2_surfaces = get_clipped_mask_2(mask_2_surfaces, a, b, c, d)
        clipped_mask_1_surfaces = get_clipped_mask_1(mask_1_surfaces, a, b, c, d)

        return torch.cat((mask_3_surfaces, clipped_mask_2_surfaces, clipped_mask_1_surfaces), dim=0)
    
    def get_shades(self, surfaces): # expects Nx3x3
        rad_light_source_angle = torch.deg2rad(self.light_source_angle)
        initial_normals = get_normals(surfaces)
        light_source_vector = torch.stack((torch.cos(rad_light_source_angle[0]) * torch.cos(rad_light_source_angle[1]), torch.cos(rad_light_source_angle[0]) * torch.sin(rad_light_source_angle[1]), torch.sin(rad_light_source_angle[0])), dim=0)
        dot_normal_light_source = torch.mv(initial_normals, light_source_vector) # dot_normal_light_source has shape N
        normals = torch.where(dot_normal_light_source.unsqueeze(1) < 0, -initial_normals, initial_normals)
        # we now have normals that are always on the brightest side
        camera_vectors = self.camera_point - surfaces[:, 0, :]
        normal_camera_angles = get_angle(a=normals, b=camera_vectors)
        cos_incidence_angles = get_cosine(a=normals, b=light_source_vector)
        # cos_light_source_angle should take shape N, and will take a value between 0 and 1 since the angle won't be more than 90 degrees
        initial_shades = cos_incidence_angles
        corrected_shades = torch.where(normal_camera_angles < torch.pi / 2, initial_shades, -initial_shades) / 2 + 0.5
        # this uses Lambertian reflectance which where the cosine of the angle of incidence decides brightness
        return corrected_shades # returns a value between 0 and 1
        
    def get_screen_coordinates(self, relevant_surfaces): #expects Nx3x3
        sin_camera_lat = torch.sin(torch.deg2rad(self.camera_angle[0]))
        cos_camera_lat = torch.cos(torch.deg2rad(self.camera_angle[0]))
        sin_camera_lon = torch.sin(torch.deg2rad(self.camera_angle[1]))
        cos_camera_lon = torch.cos(torch.deg2rad(self.camera_angle[1]))
        
        local_horizontals = sin_camera_lon * (relevant_surfaces[:, :, 0] - self.camera_point[0]) - cos_camera_lon * (relevant_surfaces[:, :, 1] - self.camera_point[1])
        local_verticals = cos_camera_lat * (relevant_surfaces[:, :, 2] - self.camera_point[2]) - sin_camera_lat * (sin_camera_lon * (relevant_surfaces[:, :, 1] - self.camera_point[1]) + cos_camera_lon * (relevant_surfaces[:, :, 0] - self.camera_point[0]))
        local_depths = sin_camera_lat * (relevant_surfaces[:, :, 2] - self.camera_point[2]) + cos_camera_lat * (sin_camera_lon * (relevant_surfaces[:, :, 1] - self.camera_point[1]) + cos_camera_lon * (relevant_surfaces[:, :, 0] - self.camera_point[0]))

        f = 0.5 / torch.tan(torch.deg2rad(self.fov) / 2)
        x = f * local_horizontals / local_depths # value between -1 and 1
        y = f * local_verticals / local_depths # value between -1 and 1
        shades = self.get_shades(relevant_surfaces)
        # shades = torch.sigmoid(local_depths[:, 0] * 8)
        # 1 / local_depths because this varies linearly across the screen, as opposed to depth itself, which does not
        return torch.stack((x, y, 1 / local_depths), dim=2), shades # Nx3x3, N


    def get_screen_triangles(self, general_equations, degenerate, depth_planes, shades):
        if general_equations.size(0) == 0:
            return torch.zeros((SCREEN_SIZE, SCREEN_SIZE), device=self.device)
    
        gen_eq_a = general_equations[:, :, 0].unsqueeze(2).unsqueeze(3) 
        gen_eq_b = general_equations[:, :, 1].unsqueeze(2).unsqueeze(3)
        gen_eq_c = general_equations[:, :, 2].unsqueeze(2).unsqueeze(3)
        
        within_triangle = (gen_eq_a * self.grid_values[:, :, 0] + gen_eq_b * self.grid_values[:, :, 1] + gen_eq_c >= 0).all(dim=1) 
        within_triangle[degenerate] = False

        # Allocate ONLY flat 1D screen buffers (Size: K * K) instead of 3D tensors
        num_pixels = SCREEN_SIZE * SCREEN_SIZE
        final_depth_buffer = torch.full((num_pixels,), -float('inf'), device=self.device)
        final_shade_buffer = torch.zeros((num_pixels,), device=self.device)

        if within_triangle.any():
            # 1. Extract 1D indices for active fragments across all triangles
            n_idx, y_idx, x_idx = torch.where(within_triangle)
            
            # 2. Gather parameters and compute depth values purely in 1D
            a = depth_planes[n_idx, 0]
            b = depth_planes[n_idx, 1]
            c = depth_planes[n_idx, 2]
            d = depth_planes[n_idx, 3]

            gx = self.grid_values[y_idx, x_idx, 0]
            gy = self.grid_values[y_idx, x_idx, 1]

            depth_values_1d = (-a * gx - b * gy - d) / c
            shades_1d = shades[n_idx].view(-1)

            # 3. Map 2D (y, x) screen coordinates to flat 1D pixel indices
            flat_pixel_indices = y_idx * SCREEN_SIZE + x_idx

            # 4. Perform a 1D Max Pooling operation directly into the depth buffer
            final_depth_buffer.scatter_reduce_(0, flat_pixel_indices, depth_values_1d, reduce="max", include_self=True)
            
            # 5. Check which fragments matched the winning max depth (with a tiny floating point tolerance)
            winning_mask = torch.abs(depth_values_1d - final_depth_buffer[flat_pixel_indices]) < EPSILON
            
            # 6. Safely pass only the winning shades to the 1D frame buffer
            final_shade_buffer[flat_pixel_indices[winning_mask]] = shades_1d[winning_mask]
        
        # Reshape the flat 1D screen array back to a 2D matrix (KxK) for presentation
        return final_shade_buffer.view(SCREEN_SIZE, SCREEN_SIZE)

    def key_handling(self):
        rad_camera_angle = torch.deg2rad(self.camera_angle)
        if input_manager.is_held('w'):
            self.camera_point[0] += torch.cos(rad_camera_angle[0]) * torch.cos(rad_camera_angle[1]) * VELOCITY
            self.camera_point[1] += torch.cos(rad_camera_angle[0]) * torch.sin(rad_camera_angle[1]) * VELOCITY
            self.camera_point[2] += torch.sin(rad_camera_angle[0]) * VELOCITY
        if input_manager.is_held('s'):
            self.camera_point[0] -= torch.cos(rad_camera_angle[0]) * torch.cos(rad_camera_angle[1]) * VELOCITY
            self.camera_point[1] -= torch.cos(rad_camera_angle[0]) * torch.sin(rad_camera_angle[1]) * VELOCITY
            self.camera_point[2] -= torch.sin(rad_camera_angle[0]) * VELOCITY
        if input_manager.is_held('a'):
            self.camera_point[0] -= torch.sin(rad_camera_angle[1]) * VELOCITY
            self.camera_point[1] += torch.cos(rad_camera_angle[1]) * VELOCITY
        if input_manager.is_held('d'):
            self.camera_point[0] += torch.sin(rad_camera_angle[1]) * VELOCITY
            self.camera_point[1] -= torch.cos(rad_camera_angle[1]) * VELOCITY

        if input_manager.is_held('i'):
            self.light_source_angle[0] += ANGULAR_VELOCITY
        if input_manager.is_held('k'):
            self.light_source_angle[0] -= ANGULAR_VELOCITY
        if input_manager.is_held('j'):
            self.light_source_angle[1] += ANGULAR_VELOCITY
        if input_manager.is_held('l'):
            self.light_source_angle[1] -= ANGULAR_VELOCITY
        self.light_source_angle[0] = torch.clamp(self.light_source_angle[0], -90, 90)

        if input_manager.is_held('up'):
            self.camera_angle[0] += ANGULAR_VELOCITY
        if input_manager.is_held('down'):
            self.camera_angle[0] -= ANGULAR_VELOCITY
        if input_manager.is_held('left'):
            self.camera_angle[1] += ANGULAR_VELOCITY
        if input_manager.is_held('right'):
            self.camera_angle[1] -= ANGULAR_VELOCITY
        self.camera_angle[0] = torch.clamp(self.camera_angle[0], -90, 90)

        if input_manager.is_held('+'):
            self.fov += 5
        if input_manager.is_held('-'):
            self.fov -= 5
        self.fov = torch.clamp(self.fov, MIN_FOV, MAX_FOV)
        
    def update_logic(self):
        self.key_handling()
        relevant_surfaces = self.get_relevant_surfaces()
        screen_coordinates, shades = self.get_screen_coordinates(relevant_surfaces)
        depth_planes = get_depth_planes(screen_coordinates) # Nx4
        general_equations, degenerate = get_general_equations(screen_coordinates)
        pixel_shades = self.get_screen_triangles(general_equations, degenerate, depth_planes, shades)
        self.grid = pixel_shades
    
def get_image():
    engine.update_logic()
    grid_4d = engine.grid.unsqueeze(0).unsqueeze(0)
    upscaled_grid = torch.nn.functional.interpolate(
        grid_4d, 
        size=(SCREEN_SIZE * 2, SCREEN_SIZE * 2), 
        mode='nearest'
    ).squeeze() # Make it a 2D tensor again (500x500)
    shade_pixels = (upscaled_grid * 255).to(torch.uint8).cpu().numpy()
    
    new_img = ImageTk.PhotoImage(image=Image.fromarray(shade_pixels, mode='L'))
    label.config(image=new_img) 
    label.image = new_img

    # --- Update the Metric Labels Panel ---
    c_pos = engine.camera_point.cpu().tolist()
    c_rot = engine.camera_angle.cpu().tolist()
    s_rot = engine.light_source_angle.cpu().tolist()


    lbl_fov.config(text=f"Fov: {engine.fov}")
    lbl_cam_pos.config(text=f"Camera Position:\nX: {c_pos[0]:.2f}, Y: {c_pos[1]:.2f}, Z: {c_pos[2]:.2f}")
    lbl_cam_rot.config(text=f"Camera Angle:\nLat: {c_rot[0]:.1f}°, Lon: {c_rot[1]:.1f}°")
    lbl_light_source_rot.config(text=f"Light Source Angle:\nLat: {s_rot[0]:.1f}°, Lon: {s_rot[1]:.1f}°")

def update_loop():
    if len(input_manager.pressed_keys) > 0:
        get_image()
    root.after(15, update_loop)

class InputManager:
    def __init__(self):
        self.pressed_keys = set()
        self.listener = keyboard.Listener(
            on_press=self._on_press, 
            on_release=self._on_release
        )
        self.listener.start()
    def _on_press(self, key):
        # Handle regular character keys
        if hasattr(key, 'char') and key.char:
            self.pressed_keys.add(key.char)
        # Handle special keys (arrows)
        else:
            self.pressed_keys.add(key)
    def _on_release(self, key):
        if hasattr(key, 'char') and key.char:
            self.pressed_keys.discard(key.char)
        else:
            self.pressed_keys.discard(key)
    def is_held(self, key_name):
        # Check against string character or pynput Key object
        if key_name in self.pressed_keys:
            return True
        # Map strings to special key objects
        key_map = {
            'up': keyboard.Key.up,
            'down': keyboard.Key.down,
            'left': keyboard.Key.left,
            'right': keyboard.Key.right
        }
        return key_map.get(key_name) in self.pressed_keys
    
    

input_manager = InputManager()
engine = Engine()


root = tk.Tk()
root.title("3D Engine Display")

top_row_frame = tk.Frame(root)
top_row_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=10)

label = tk.Label(top_row_frame)
label.pack(side=tk.LEFT)

info_panel = tk.Frame(top_row_frame, padx=20)
info_panel.pack(side=tk.LEFT, fill=tk.Y, anchor="n")

lbl_fov = tk.Label(info_panel, justify="left", anchor="w")
lbl_fov.pack(anchor="w", pady=6)

lbl_cam_pos = tk.Label(info_panel, justify="left", anchor="w")
lbl_cam_pos.pack(anchor="w", pady=6)

lbl_cam_rot = tk.Label(info_panel, justify="left", anchor="w")
lbl_cam_rot.pack(anchor="w", pady=6)

lbl_light_source_rot = tk.Label(info_panel, justify="left", anchor="w")
lbl_light_source_rot.pack(anchor="w", pady=6)

lbl_surf_idx = tk.Label(info_panel, justify="left", anchor="w")
lbl_surf_idx.pack(anchor="w", pady=6)

get_image()

grid_frame = tk.Frame(root)
grid_frame.pack(pady=10)

def add_placeholder(entry, placeholder_text):
    default_fg = entry.cget("fg")
    entry.insert(0, placeholder_text)
    entry.config(fg="gray")
    entry.is_placeholder = True

    def on_focus_in(event):
        if entry.is_placeholder:
            entry.delete(0, tk.END)
            entry.config(fg=default_fg)
            entry.is_placeholder = False

    def on_focus_out(event):
        if not entry.get().strip():
            entry.insert(0, placeholder_text)
            entry.config(fg="gray")
            entry.is_placeholder = True

    entry.bind("<FocusIn>", on_focus_in)
    entry.bind("<FocusOut>", on_focus_out)

entries = []
labels = ["x", "y", "z"]
for r in range(3):
    row_frame = tk.Frame(grid_frame)
    row_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
    row_entries = []
    for c in range(3):
        entry = tk.Entry(row_frame, width=8, justify="center")
        entry.pack(side=tk.LEFT, padx=2)
        add_placeholder(entry, labels[c])
        row_entries.append(entry)
    entries.append(row_entries)

def add_surface():
    root.focus_set()
    matrix_data = []
    try:
        for r in range(3):
            row_data = []
            for c in range(3):
                entry = entries[r][c]
                if getattr(entry, 'is_placeholder', False):
                    raise ValueError
                val = float(entry.get().strip())
                row_data.append(val)
            matrix_data.append(row_data)
            
        engine.surfaces[engine.surfaces_count, :, :] = torch.tensor(matrix_data, dtype=torch.float32)
        engine.surfaces_count += 1
        get_image()
        cancel_writing()
    except ValueError:
        messagebox.showerror("Invalid Input", "Please ensure all 9 fields contain valid numbers.")

def cancel_writing():
    root.focus_set()
    labels = ["x", "y", "z"]
    for r in range(3):
        for c in range(3):
            entry = entries[r][c]
            entry.delete(0, tk.END)
            entry.insert(0, labels[c])
            entry.config(fg="gray")
            entry.is_placeholder = True

add_button = tk.Button(root, text="Add Surface", command=add_surface)
add_button.pack(pady=5)

cancel_button = tk.Button(root, text="Cancel", command=cancel_writing)
cancel_button.pack(pady=5)

update_loop()
root.mainloop()
