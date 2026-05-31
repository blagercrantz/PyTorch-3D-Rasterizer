import torch
import tkinter as tk
from PIL import Image, ImageTk
from pynput import keyboard

def get_angle(a, b):
    # returns shape N
    if a.dim() == 1: a = a.unsqueeze(0)
    if b.dim() == 1: b = b.unsqueeze(0)
    a_magnitudes = torch.linalg.vector_norm(a, dim=1)
    b_magnitudes = torch.linalg.vector_norm(b, dim=1)
    dot_ab = torch.linalg.vecdot(a, b, dim=1)
    return torch.acos(torch.clamp(dot_ab / ((a_magnitudes * b_magnitudes)), -1 + 1e-7, 1 - 1e-7))

def get_cosine(a, b):
    # returns shape N
    if a.dim() == 1: a = a.unsqueeze(0)
    if b.dim() == 1: b = b.unsqueeze(0)
    a_magnitudes = torch.linalg.vector_norm(a, dim=1)
    b_magnitudes = torch.linalg.vector_norm(b, dim=1)
    dot_ab = torch.linalg.vecdot(a, b, dim=1)
    return torch.clamp(dot_ab / ((a_magnitudes * b_magnitudes)), -1 + 1e-7, 1 - 1e-7)

def get_intersection_points(p1, p2, a, b, c, d): # p1, p2 expects two tensors of shape Nx3
        n = torch.stack([a, b, c]) # shape 3
        v = p2 - p1 # shape Nx3
        denominator = (n * v).sum(dim=1)
        if torch.any(denominator.abs() < 1e-8): 
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
        
        substitution = a * x[:, [2, 0, 1]] + b * y[:, [2, 0, 1]] + c # shape Nx3
        degenerate = (torch.abs(substitution) < 0.000001).any(dim=1) # shape N
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
        # will return Nx3x3
    
class Engine:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise RuntimeError("No GPU detected! This engine requires a Mac (MPS) or PC (CUDA) GPU to run.")
       
        self.fov = torch.tensor(120.0, device=self.device)
        self.camera_point = torch.zeros(3, device=self.device)
        self.camera_rotation = torch.zeros(2, device=self.device)
        self.sun_rotation = torch.tensor([75.0, 54.0], device=self.device)
        
        self.surfaces = torch.zeros((1000, 3, 3), device=self.device)
        
        self.surfaces_count = 576

        lat_range = torch.arange(start=-3, end=3, step=0.5, device=self.device) # shape 6
        lon_range = torch.arange(start=0, end=12, step=0.5, device=self.device) # shape 12

        lat_grid, lon_grid = torch.meshgrid(lat_range, lon_range, indexing='xy') # both will have shape 6x12
        lat_lon_grid = torch.stack((lat_grid, lon_grid), dim=2) # shape 6x12x2

        grid_a = lat_lon_grid.reshape(lat_range.shape[0] * lon_range.shape[0], 2) # reshapes into 72x2
        grid_b = grid_a.clone()
        grid_b[:, 0] = grid_a[:, 0] + 0.5 # 72x2
        grid_c = grid_a.clone()
        grid_c[:, 1] = grid_a[:, 1] + 0.5
        grid_d = grid_a.clone()
        grid_d[:, 0] = grid_a[:, 0] + 0.5
        grid_d[:, 1] = grid_a[:, 1] + 0.5

        first_stacked_grid = torch.stack((grid_a, grid_b, grid_c), dim=1) #72x3x2
        second_stacked_grid = torch.stack((grid_d, grid_b, grid_c), dim=1) #72x3x2
        stacked_grid = torch.cat((first_stacked_grid, second_stacked_grid), dim=0) #144x3x2
        grid_degrees = stacked_grid * 30
        grid_rad = torch.deg2rad(grid_degrees)

        x_values = torch.cos(grid_rad[:, :, 0]) * torch.cos(grid_rad[:, :, 1]) #144x3
        y_values = torch.cos(grid_rad[:, :, 0]) * torch.sin(grid_rad[:, :, 1])
        z_values = torch.sin(grid_rad[:, :, 0])

        sphere_xyz = torch.stack((x_values, y_values, z_values), dim=2)
        self.surfaces[:576] = sphere_xyz


        self.screen_size = 200
        self.grid = torch.rand((self.screen_size, self.screen_size), device=self.device)
        # dim=0 moves vertically down the rows (Height)
        # dim=1 moves horizontally across the columns (Width)
        

    def get_near_clipping_plane(self):
        rad_camera_rotation = torch.deg2rad(self.camera_rotation)
        close_point = torch.stack([self.camera_point[0] + torch.cos(rad_camera_rotation[0]) * torch.cos(rad_camera_rotation[1]) * 0.005,
                                    self.camera_point[1] + torch.cos(rad_camera_rotation[0]) * torch.sin(rad_camera_rotation[1]) * 0.005,
                                    self.camera_point[2] + torch.sin(rad_camera_rotation[0]) * 0.001])
        a = torch.cos(rad_camera_rotation[1]) * torch.cos(rad_camera_rotation[0])
        b = torch.sin(rad_camera_rotation[1]) * torch.cos(rad_camera_rotation[0])
        c = torch.sin(rad_camera_rotation[0])
        d = -a * close_point[0] -b * close_point[1] -c * close_point[2]
        return a, b, c, d
        
    def get_relevant_surfaces(self):
        rad_camera_rotation = torch.deg2rad(self.camera_rotation)
        active_surfaces = self.surfaces[:self.surfaces_count]
        a, b, c, d = self.get_near_clipping_plane()

        local_depths = torch.sin(rad_camera_rotation[0]) * (active_surfaces[:, :, 2] - self.camera_point[2]) + torch.cos(rad_camera_rotation[0]) * (torch.sin(rad_camera_rotation[1]) * (active_surfaces[:, :, 1] - self.camera_point[1]) + torch.cos(rad_camera_rotation[1]) * (active_surfaces[:, :, 0] - self.camera_point[0]))
        # Now we'll sort the points in every triangle by depth which will make it easier to clip triangles that are only partially in frame
        # Find the indices that would sort the depths, lowest depth first
        sort_indices = torch.argsort(local_depths, dim=1)
        # Create a helper array for dimension N
        batch_indices = torch.arange(self.surfaces_count, device=self.device).unsqueeze(1)
        # Apply the sorted indices to the surfaces and depths
        active_surfaces = active_surfaces[batch_indices, sort_indices, :]
        local_depths = local_depths[batch_indices, sort_indices]

        corners_in_front_count = (local_depths > 0.005).sum(dim=1)

        mask_3_surfaces = active_surfaces[corners_in_front_count == 3]
        mask_2_surfaces = active_surfaces[corners_in_front_count == 2]
        mask_1_surfaces = active_surfaces[corners_in_front_count == 1]
        clipped_mask_2_surfaces = get_clipped_mask_2(mask_2_surfaces, a, b, c, d)
        clipped_mask_1_surfaces = get_clipped_mask_1(mask_1_surfaces, a, b, c, d)
        return torch.cat((mask_3_surfaces, clipped_mask_2_surfaces, clipped_mask_1_surfaces), dim=0)
    
    def get_shades(self, surfaces): # expects Nx3x3
        rad_sun_rotation = torch.deg2rad(self.sun_rotation)
        initial_normals = get_normals(surfaces)
        sun_vector = torch.stack((torch.cos(rad_sun_rotation[0]) * torch.cos(rad_sun_rotation[1]), torch.cos(rad_sun_rotation[0]) * torch.sin(rad_sun_rotation[1]), torch.sin(rad_sun_rotation[0])), dim=0)
        dot_normal_sun = torch.mv(initial_normals, sun_vector) # dot_normal_sun has shape N
        normals = torch.where(dot_normal_sun.unsqueeze(1) < 0, -initial_normals, initial_normals)
        # we now have normals that are always on the brightest side
        camera_vectors = self.camera_point - surfaces[:, 0, :]
        normal_camera_angles = get_angle(a=normals, b=camera_vectors)
        cos_incidence_angles = get_cosine(a=normals, b=sun_vector)
        # cos_sun_angle should take shape N, and will take a value between 0 and 1 since the angle won't be more than 90 degrees
        initial_shades = cos_incidence_angles
        # this uses Lambertian reflectance which where the cosine of the angle of incidence decides brightness
        return torch.where(normal_camera_angles < torch.pi / 2, initial_shades, -initial_shades) / 2.4 + 0.5 # returns a value between 0 and 1
        
    def get_2d_corners(self, relevant_surfaces): #expects Nx3x3
        rad_camera_rotation = torch.deg2rad(self.camera_rotation)
        
        local_horizontals = torch.sin(rad_camera_rotation[1]) * (relevant_surfaces[:, :, 0] - self.camera_point[0]) - torch.cos(rad_camera_rotation[1]) * (relevant_surfaces[:, :, 1] - self.camera_point[1])
        local_verticals = torch.cos(rad_camera_rotation[0]) * (relevant_surfaces[:, :, 2] - self.camera_point[2]) - torch.sin(rad_camera_rotation[0]) * (torch.sin(rad_camera_rotation[1]) * (relevant_surfaces[:, :, 1] - self.camera_point[1]) + torch.cos(rad_camera_rotation[1]) * (relevant_surfaces[:, :, 0] - self.camera_point[0]))
        local_depths = torch.sin(rad_camera_rotation[0]) * (relevant_surfaces[:, :, 2] - self.camera_point[2]) + torch.cos(rad_camera_rotation[0]) * (torch.sin(rad_camera_rotation[1]) * (relevant_surfaces[:, :, 1] - self.camera_point[1]) + torch.cos(rad_camera_rotation[1]) * (relevant_surfaces[:, :, 0] - self.camera_point[0]))

        f = 0.5 / torch.tan(torch.deg2rad(self.fov) / 2)
        x = f * local_horizontals / local_depths # value between -1 and 1
        y = f * local_verticals / local_depths # value between -1 and 1
        shades = self.get_shades(relevant_surfaces)
        # 1 / local_depths because this varies linearly across the screen, as opposed to depth itself, which does not
        return torch.stack((x, y, 1 / local_depths), dim=2), shades # Nx3x3, N


    def get_screen_triangles(self, general_equations, degenerate, depth_planes, shades):

        if general_equations.size(0) == 0:
            return torch.zeros((self.screen_size, self.screen_size), device=self.device)
    
        # ±(0.5 / self.screen_size) ensures center of each pixel
        values_dim0 = torch.linspace(1 - (0.5 / self.screen_size), -1 + (0.5 / self.screen_size), self.screen_size, device=self.device)
        values_dim1 = torch.linspace(-1 + (0.5 / self.screen_size), 1 - (0.5 / self.screen_size), self.screen_size, device=self.device)
        grid_dim0, grid_dim1 = torch.meshgrid(values_dim0, values_dim1, indexing='ij')
        grid_values = torch.stack([grid_dim0, grid_dim1], dim=-1) #KxKx2 where K = screen_size

        gen_eq_a = general_equations[:, :, 0].unsqueeze(2).unsqueeze(3) # Nx3x1x1, this unsqueezing is because we want to get Nx3x1x1 * KxK = Nx3xKxK
        gen_eq_b = general_equations[:, :, 1].unsqueeze(2).unsqueeze(3)
        gen_eq_c = general_equations[:, :, 2].unsqueeze(2).unsqueeze(3)
        within_triangle = (gen_eq_a * grid_values[:, :, 1] + gen_eq_b * grid_values[:, :, 0] + gen_eq_c > 0).all(dim=1) # NxKxK
        within_triangle[degenerate] = False

        depth_triangles = torch.where(
             within_triangle, 
            (-depth_planes[:, 0].unsqueeze(1).unsqueeze(2) * grid_values[:, :, 1] -depth_planes[:, 1].unsqueeze(1).unsqueeze(2) * grid_values[:, :, 0] -depth_planes[:, 3].unsqueeze(1).unsqueeze(2))/depth_planes[:, 2].unsqueeze(1).unsqueeze(2), 
            torch.tensor(-float('inf'), device=self.device))
        shade_triangles = torch.where(within_triangle, shades.unsqueeze(1).unsqueeze(2), torch.tensor(0.0, device=self.device)) #NxKxK

        # Safe max calculation. 
        max_depth, index_closest_pixel = torch.max(depth_triangles, dim=0) # Both are KxK
        
        # Gather the shades corresponding to the closest depth
        closest_triangle_shades = torch.gather(shade_triangles, dim=0, index=index_closest_pixel.unsqueeze(0)).squeeze(0) # KxK

        # Clean up empty pixels (where depth remained -inf)
        closest_triangle_shades[max_depth == -float('inf')] = 0.0
        
        return closest_triangle_shades

    def key_handling(self):
        rad_camera_rotation = torch.deg2rad(self.camera_rotation)
        if input_manager.is_held('w'):
            self.camera_point[0] += torch.cos(rad_camera_rotation[0]) * torch.cos(rad_camera_rotation[1]) * 0.075
            self.camera_point[1] += torch.cos(rad_camera_rotation[0]) * torch.sin(rad_camera_rotation[1]) * 0.075
            self.camera_point[2] += torch.sin(rad_camera_rotation[0]) * 0.075
        if input_manager.is_held('s'):
            self.camera_point[0] -= torch.cos(rad_camera_rotation[0]) * torch.cos(rad_camera_rotation[1]) * 0.075
            self.camera_point[1] -= torch.cos(rad_camera_rotation[0]) * torch.sin(rad_camera_rotation[1]) * 0.075
            self.camera_point[2] -= torch.sin(rad_camera_rotation[0]) * 0.075
        if input_manager.is_held('a'):
            self.camera_point[0] -= torch.cos(rad_camera_rotation[0]) * torch.sin(rad_camera_rotation[1]) * 0.075
            self.camera_point[1] += torch.cos(rad_camera_rotation[0]) * torch.cos(rad_camera_rotation[1]) * 0.075
        if input_manager.is_held('d'):
            self.camera_point[0] += torch.cos(rad_camera_rotation[0]) * torch.sin(rad_camera_rotation[1]) * 0.075
            self.camera_point[1] -= torch.cos(rad_camera_rotation[0]) * torch.cos(rad_camera_rotation[1]) * 0.075



        if input_manager.is_held('up'):
            self.camera_rotation[0] += 7.5
            if self.camera_rotation[0] > 90:
                self.camera_rotation[0] = 90
        if input_manager.is_held('down'):
            self.camera_rotation[0] -= 7.5
            if self.camera_rotation[0] < -90:
                self.camera_rotation[0] = -90
        if input_manager.is_held('left'):
            self.camera_rotation[1] += 7.5
        if input_manager.is_held('right'):
            self.camera_rotation[1] -= 7.5
        
    def update_logic(self):
        self.key_handling()
             
        relevant_surfaces = self.get_relevant_surfaces()
        screen_coordinates, shades = self.get_2d_corners(relevant_surfaces)
        depth_planes = get_depth_planes(screen_coordinates) # Nx4
        general_equations, degenerate = get_general_equations(screen_coordinates)
        pixel_shades = self.get_screen_triangles(general_equations, degenerate, depth_planes, shades)

        self.grid = pixel_shades

    def get_image(self):
        shade_pixels = (self.grid[:, :] * 255).to(torch.uint8).cpu().numpy()
        
        return ImageTk.PhotoImage(image=Image.fromarray(shade_pixels, mode='L'))

def update_loop():
    engine.update_logic()
    new_img = engine.get_image() 
    
    label.config(image=new_img) 
    label.image = new_img

    root.after(25, update_loop)


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
root.title("GPU Grid Viewer")

label = tk.Label(root)
label.pack()

update_loop()
root.mainloop()
