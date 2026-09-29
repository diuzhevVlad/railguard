# Donald Trump — stylized caricature

Original procedural mesh with swept blond hair, a dark suit, white shirt and red tie.
Simplified artistic likeness, not a scan or accurate anatomical reconstruction.
Approximate height 1.89 m; this is an artistic choice, not a factual measurement.
Closed overlapping body parts, not a fused surface. Static, no animation.

- `meshes/trump.dae`: colored COLLADA mesh for Gazebo; metres, Z up, X forward.
- `meshes/trump.glb`: colored glTF binary for Rerun; metres, converted to Y up.
- `preview.png`: front and oblique views.

Open in Rerun from the repository root:

```bash
rerun models/trump_stylized/meshes/trump.glb
```

For Gazebo, add the repository's `models` directory to your version's model
resource path, as described in `../person_standing/README.md`, then include:

```xml
<include>
  <uri>model://trump_stylized</uri>
  <name>trump_obstacle</name>
  <pose>10 0 0 0 0 0</pose>
</include>
```

Visual and collision geometry use the same mesh. No sensor reflectance model.
Gazebo and Rerun loading have not been tested in this environment.
