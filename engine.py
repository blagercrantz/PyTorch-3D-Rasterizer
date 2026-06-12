import torch
import math_functions

# Constants
EPSILON = 1e-8
VELOCITY = 0.075
ANGULAR_VELOCITY = 7.5
MIN_FOV, MAX_FOV = 0, 175
CLIPPING_DISTANCE = 0.005

class Engine:
    def __init__(self, screen_size):
        self.screen_size = screen_size
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
       
        self.fov = torch.tensor(110.0, device=self.device)
        self.camera_point = torch.zeros(3, device=self.device)
        self.camera_angle = torch.zeros(2, device=self.device)
        self.light_source_angle = torch.tensor([45.0, 0.0], device=self.device)

        self.surfaces = torch.zeros((500, 3, 3), device=self.device)
        self.surfaces_count = 0
        self.example_environment()

        self.grid = torch.rand((self.screen_size, self.screen_size), device=self.device)
        
        values_dim0 = torch.linspace(-1 + (0.5 / self.screen_size), 1 - (0.5 / self.screen_size), self.screen_size, device=self.device)
        values_dim1 = torch.linspace(1 - (0.5 / self.screen_size), -1 + (0.5 / self.screen_size), self.screen_size, device=self.device)
        grid_dim0, grid_dim1 = torch.meshgrid(values_dim0, values_dim1, indexing='xy')
        self.grid_values = torch.stack([grid_dim0, grid_dim1], dim=2)

    def example_environment(self):
        lat_range = torch.arange(start=-2.5, end=1, step=0.5, device=self.device) # shape A
        lon_range = torch.arange(start=0, end=12, step=0.5, device=self.device) # shape B

        lat_grid, lon_grid = torch.meshgrid(lat_range, lon_range, indexing='xy')
        lat_lon_grid = torch.stack((lat_grid, lon_grid), dim=2) # shape AxBx2

        grid_a = lat_lon_grid.reshape(lat_range.size(0) * lon_range.size(0), 2) # reshapes into (A*B)x2
        grid_b = grid_a.clone()
        grid_b[:, 0] = grid_a[:, 0] + 0.5 # (A*B)x2
        grid_c = grid_a.clone()
        grid_c[:, 1] = grid_a[:, 1] + 0.5
        grid_d = grid_a.clone()
        grid_d[:, 0] = grid_a[:, 0] + 0.5
        grid_d[:, 1] = grid_a[:, 1] + 0.5

        botton_lat_range = torch.tensor([-3.0], device=self.device)
        bottom_lat_grid, bottom_lon_grid = torch.meshgrid(botton_lat_range, lon_range, indexing='xy')
        bottom_lat_lon_grid = torch.stack((bottom_lat_grid, bottom_lon_grid), dim=2)
        bottom_grid_a = bottom_lat_lon_grid.reshape(botton_lat_range.size(0) * lon_range.size(0), 2)
        bottom_grid_b = bottom_grid_a.clone()
        bottom_grid_b[:, 0] = bottom_grid_a[:, 0] + 0.5
        bottom_grid_c = bottom_grid_a.clone()
        bottom_grid_c[:, 0] = bottom_grid_a[:, 0] + 0.5
        bottom_grid_c[:, 1] = bottom_grid_a[:, 1] + 0.5

        first_stacked_grid = torch.stack((grid_a, grid_b, grid_c), dim=1) #(A*B)x3x2
        second_stacked_grid = torch.stack((grid_d, grid_b, grid_c), dim=1) #(A*B)x3x2
        bottom_stacked_grid = torch.stack((bottom_grid_a, bottom_grid_b, bottom_grid_c), dim=1) #(A*B)x3x2

        stacked_grid = torch.cat((first_stacked_grid, second_stacked_grid, bottom_stacked_grid), dim=0) #(2AB)x3x2
        grid_degrees = stacked_grid * 30
        grid_rad = torch.deg2rad(grid_degrees)

        x_values = torch.cos(grid_rad[:, :, 0]) * torch.cos(grid_rad[:, :, 1]) #(2AB)
        y_values = torch.cos(grid_rad[:, :, 0]) * torch.sin(grid_rad[:, :, 1])
        z_values = torch.sin(grid_rad[:, :, 0])

        sphere_xyz = torch.stack((x_values, y_values, z_values), dim=2)
        self.surfaces_count = sphere_xyz.size(0)        
        self.surfaces[:self.surfaces_count] = sphere_xyz

    def get_near_clipping_plane(self):
        rad_camera_angle = torch.deg2rad(self.camera_angle)
        close_point = torch.stack([
            self.camera_point[0] + torch.cos(rad_camera_angle[0]) * torch.cos(rad_camera_angle[1]) * CLIPPING_DISTANCE,
            self.camera_point[1] + torch.cos(rad_camera_angle[0]) * torch.sin(rad_camera_angle[1]) * CLIPPING_DISTANCE,
            self.camera_point[2] + torch.sin(rad_camera_angle[0]) * CLIPPING_DISTANCE
        ])
        a = torch.cos(rad_camera_angle[1]) * torch.cos(rad_camera_angle[0])
        b = torch.sin(rad_camera_angle[1]) * torch.cos(rad_camera_angle[0])
        c = torch.sin(rad_camera_angle[0])
        d = -a * close_point[0] - b * close_point[1] - c * close_point[2]
        return a, b, c, d
        
    def get_relevant_surfaces(self):
        rad_camera_angle = torch.deg2rad(self.camera_angle)
        active_surfaces = self.surfaces[:self.surfaces_count]
        a, b, c, d = self.get_near_clipping_plane()

        local_depths = torch.sin(rad_camera_angle[0]) * (active_surfaces[:, :, 2] - self.camera_point[2]) + \
                       torch.cos(rad_camera_angle[0]) * (torch.sin(rad_camera_angle[1]) * (active_surfaces[:, :, 1] - self.camera_point[1]) + \
                       torch.cos(rad_camera_angle[1]) * (active_surfaces[:, :, 0] - self.camera_point[0]))
        
        sort_indices = torch.argsort(local_depths, dim=1)
        batch_indices = torch.arange(active_surfaces.size(0), device=self.device).unsqueeze(1)
        active_surfaces = active_surfaces[batch_indices, sort_indices, :]
        local_depths = local_depths[batch_indices, sort_indices]

        corners_in_front_count = (local_depths > CLIPPING_DISTANCE).sum(dim=1)

        mask_3_surfaces = active_surfaces[corners_in_front_count == 3]
        clipped_mask_2_surfaces = math_functions.get_clipped_mask_2(active_surfaces[corners_in_front_count == 2], a, b, c, d)
        clipped_mask_1_surfaces = math_functions.get_clipped_mask_1(active_surfaces[corners_in_front_count == 1], a, b, c, d)

        return torch.cat((mask_3_surfaces, clipped_mask_2_surfaces, clipped_mask_1_surfaces), dim=0)
    
    def get_shades(self, surfaces):
        rad_light_source_angle = torch.deg2rad(self.light_source_angle)
        initial_normals = math_functions.get_normals(surfaces)
        light_source_vector = torch.stack((
            torch.cos(rad_light_source_angle[0]) * torch.cos(rad_light_source_angle[1]), 
            torch.cos(rad_light_source_angle[0]) * torch.sin(rad_light_source_angle[1]), 
            torch.sin(rad_light_source_angle[0])
        ), dim=0)
        
        dot_normal_light_source = torch.mv(initial_normals, light_source_vector)
        normals = torch.where(dot_normal_light_source.unsqueeze(1) < 0, -initial_normals, initial_normals)
        camera_vectors = self.camera_point - surfaces[:, 0, :]
        
        normal_camera_angles = math_functions.get_angle(a=normals, b=camera_vectors)
        cos_incidence_angles = math_functions.get_cosine(a=normals, b=light_source_vector)
        
        return torch.where(normal_camera_angles < torch.pi / 2, cos_incidence_angles, -cos_incidence_angles) / 2 + 0.5
        
    def get_screen_coordinates(self, relevant_surfaces):
        sin_cam_lat, cos_cam_lat = torch.sin(torch.deg2rad(self.camera_angle[0])), torch.cos(torch.deg2rad(self.camera_angle[0]))
        sin_cam_lon, cos_cam_lon = torch.sin(torch.deg2rad(self.camera_angle[1])), torch.cos(torch.deg2rad(self.camera_angle[1]))
        
        local_horizontals = sin_cam_lon * (relevant_surfaces[:, :, 0] - self.camera_point[0]) - cos_cam_lon * (relevant_surfaces[:, :, 1] - self.camera_point[1])
        local_verticals = cos_cam_lat * (relevant_surfaces[:, :, 2] - self.camera_point[2]) - sin_cam_lat * (sin_cam_lon * (relevant_surfaces[:, :, 1] - self.camera_point[1]) + cos_cam_lon * (relevant_surfaces[:, :, 0] - self.camera_point[0]))
        local_depths = sin_cam_lat * (relevant_surfaces[:, :, 2] - self.camera_point[2]) + cos_cam_lat * (sin_cam_lon * (relevant_surfaces[:, :, 1] - self.camera_point[1]) + cos_cam_lon * (relevant_surfaces[:, :, 0] - self.camera_point[0]))

        f = 0.5 / torch.tan(torch.deg2rad(self.fov) / 2)
        x = f * local_horizontals / local_depths
        y = f * local_verticals / local_depths
        shades = self.get_shades(relevant_surfaces)
        return torch.stack((x, y, 1 / local_depths), dim=2), shades

    def get_screen_triangles(self, general_equations, degenerate, depth_planes, shades):
        if general_equations.size(0) == 0:
            return torch.zeros((self.screen_size, self.screen_size), device=self.device)
    
        gen_eq_a = general_equations[:, :, 0].unsqueeze(2).unsqueeze(3) 
        gen_eq_b = general_equations[:, :, 1].unsqueeze(2).unsqueeze(3)
        gen_eq_c = general_equations[:, :, 2].unsqueeze(2).unsqueeze(3)
        
        within_triangle = (gen_eq_a * self.grid_values[:, :, 0] + gen_eq_b * self.grid_values[:, :, 1] + gen_eq_c >= 0).all(dim=1) 
        within_triangle[degenerate] = False

        num_pixels = self.screen_size * self.screen_size
        final_depth_buffer = torch.full((num_pixels,), -float('inf'), device=self.device)
        final_shade_buffer = torch.zeros((num_pixels,), device=self.device)

        if within_triangle.any():
            n_idx, y_idx, x_idx = torch.where(within_triangle)
            a, b, c, d = depth_planes[n_idx, 0], depth_planes[n_idx, 1], depth_planes[n_idx, 2], depth_planes[n_idx, 3]

            gx = self.grid_values[y_idx, x_idx, 0]
            gy = self.grid_values[y_idx, x_idx, 1]

            depth_values_1d = (-a * gx - b * gy - d) / c
            shades_1d = shades[n_idx].view(-1)
            flat_pixel_indices = y_idx * self.screen_size + x_idx

            final_depth_buffer.scatter_reduce_(0, flat_pixel_indices, depth_values_1d, reduce="max", include_self=True)
            winning_mask = torch.abs(depth_values_1d - final_depth_buffer[flat_pixel_indices]) < EPSILON
            final_shade_buffer[flat_pixel_indices[winning_mask]] = shades_1d[winning_mask]
        
        return final_shade_buffer.view(self.screen_size, self.screen_size)

    def handle_input(self, input_manager):
        rad_camera_angle = torch.deg2rad(self.camera_angle)
        if input_manager.is_held('w'):
            self.camera_point += torch.stack([torch.cos(rad_camera_angle[0]) * torch.cos(rad_camera_angle[1]), torch.cos(rad_camera_angle[0]) * torch.sin(rad_camera_angle[1]), torch.sin(rad_camera_angle[0])]) * VELOCITY
        if input_manager.is_held('s'):
            self.camera_point -= torch.stack([torch.cos(rad_camera_angle[0]) * torch.cos(rad_camera_angle[1]), torch.cos(rad_camera_angle[0]) * torch.sin(rad_camera_angle[1]), torch.sin(rad_camera_angle[0])]) * VELOCITY
        if input_manager.is_held('a'):
            self.camera_point[0] -= torch.sin(rad_camera_angle[1]) * VELOCITY
            self.camera_point[1] += torch.cos(rad_camera_angle[1]) * VELOCITY
        if input_manager.is_held('d'):
            self.camera_point[0] += torch.sin(rad_camera_angle[1]) * VELOCITY
            self.camera_point[1] -= torch.cos(rad_camera_angle[1]) * VELOCITY

        if input_manager.is_held('i'): self.light_source_angle[0] += ANGULAR_VELOCITY
        if input_manager.is_held('k'): self.light_source_angle[0] -= ANGULAR_VELOCITY
        if input_manager.is_held('j'): self.light_source_angle[1] = (self.light_source_angle[1] + ANGULAR_VELOCITY + 180) % 360 - 180
        if input_manager.is_held('l'): self.light_source_angle[1] = (self.light_source_angle[1] - ANGULAR_VELOCITY + 180) % 360 - 180
        self.light_source_angle[0] = torch.clamp(self.light_source_angle[0], -90, 90)

        if input_manager.is_held('up'): self.camera_angle[0] += ANGULAR_VELOCITY
        if input_manager.is_held('down'): self.camera_angle[0] -= ANGULAR_VELOCITY
        if input_manager.is_held('left'): self.camera_angle[1] = (self.camera_angle[1] + ANGULAR_VELOCITY + 180) % 360 - 180
        if input_manager.is_held('right'): self.camera_angle[1] = (self.camera_angle[1] - ANGULAR_VELOCITY + 180) % 360 - 180
        self.camera_angle[0] = torch.clamp(self.camera_angle[0], -90, 90)

        if input_manager.is_held('+'): self.fov += 5
        if input_manager.is_held('-'): self.fov -= 5
        self.fov = torch.clamp(self.fov, MIN_FOV, MAX_FOV)
        
    def update_logic(self, input_manager):
        self.handle_input(input_manager)
        relevant_surfaces = self.get_relevant_surfaces()
        screen_coordinates, shades = self.get_screen_coordinates(relevant_surfaces)
        depth_planes = math_functions.get_depth_planes(screen_coordinates)
        general_equations, degenerate = math_functions.get_general_equations(screen_coordinates)
        self.grid = self.get_screen_triangles(general_equations, degenerate, depth_planes, shades)