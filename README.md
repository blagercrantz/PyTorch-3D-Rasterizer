# PyTorch-3D-Rasterizer

<img width="400" height="400" alt="Image" src="https://github.com/user-attachments/assets/9a19312b-20f1-4f7a-b6a2-2dae9879ed57" />

A 3D graphics rasterization engine built from scratch using PyTorch to calculate millions of operations in parallel using the GPU instead of using for-loops. It calculates how triangular surfaces within a 3D-environment should be displayed from the perspective of a camera with its own coordinates (x, y, z) and angle.

The shading of each surface is calculated using lambertian reflectance, and depends on light_source_angle, the angle of the surface's normal, and which side of the surface the camera is seeing.

How the engine works step-by-step:

1. In get_relevant_surfaces(), relevant_surfaces are calculated, which consists of surfaces that are entirely in front of the near clipping plane, as well as the clipped parts of surfaces that intersect with it and fall on the visible side of it.
2. In get_screen_coordinates(), the coordinates on the screen of the surfaces' corners are calculated, as well as the shading of every surface.
3. In get_depth_planes(), the depth planes (ax + by + cz + d = 0) are calculated for every surface, where x and y are the "2D coordinates" of the corners, as displayed on the screen, and where z is the reciprocal of depth. It is the reciprocal of depth, and not depth, since the reciprocal varies linearly across the screen, and depth itself does not. The depth planes will be useful for displaying intersecting surfaces where neither surface is fully in front of the other.
4. In get_general_equations(), one general equation (ax + by + c > 0) is calculated for each edge of each "relevant surface" where x and y are the "2D coordinates" of the corners, as displayed on the screen. We also calculate any triangles become degenerate when displayed on the screen, to avoid showing these.
5. In get_screen_triangles(), the set of pixels that fall within each surface is calculated, as well as the shade of the to nearest surface for each pixel. This data is then displayed on the screen.

Youtube showcase: https://youtu.be/-3ZlTCT_nZc?si=WQ_Igk9q-mjqI_Eo
