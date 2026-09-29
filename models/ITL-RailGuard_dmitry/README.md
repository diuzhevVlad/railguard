# ITL-RailGuard_dmitry

Reference: `../dimtry.jpg`. Gray blazer, pale buttoned shirt, dark trousers, glasses and black leather shoes.

A manually described, procedural approximation for simulation, not a measured
3D scan or photogrammetric reconstruction. Height is chosen as 1.80 m.
Facial likeness, body proportions and hidden surfaces are approximate.
Geometry consists of overlapping closed components.

- `meshes/person.dae`: colored COLLADA, metres, Z up, +X forward.
- `meshes/person.glb`: colored glTF binary, metres, Y up; opens in Rerun.
- `model.sdf`: matching visual and collision meshes for Gazebo.
- `preview.png`: rendered front and oblique views.

Regenerate from the repository root: `python3 tools/generate_itl_people.py --person dmitry`.
The generator requires NumPy; previews additionally require Matplotlib.
