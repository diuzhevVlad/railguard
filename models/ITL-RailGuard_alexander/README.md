# ITL-RailGuard_alexander

Simplified procedural interpretation of the supplied photo: brown cap, gray
T-shirt, beige shorts, crossbody bag, bottle, white socks and sneakers.
Walking pose is static. Facial likeness and hidden surfaces are approximate.
Height is 1.83 m, chosen for the model, not measured from the photo.
Geometry comprises overlapping closed components, not one fused surface.

- `meshes/person.dae`: COLLADA for Gazebo, metres, Z up, X forward.
- `meshes/person.glb`: colored GLB for Rerun, metres, Y up.
- `preview.png`: front and oblique views.

Open from the repository root:

```bash
rerun models/ITL-RailGuard_alexander/meshes/person.glb
```

For Gazebo, set the models resource path following
`../person_standing/README.md` and insert into your world's SDF:

```xml
<include>
  <uri>model://ITL-RailGuard_alexander</uri>
  <name>photo_person_obstacle</name>
  <pose>10 0 0 0 0 0</pose>
</include>
```

Visual and collision geometry share the mesh. No calibrated reflectance model.
XML, triangle indices, closed component edges, winding and GLB buffer bounds
were checked locally. Gazebo/Rerun loading and LiDAR returns remain untested.
