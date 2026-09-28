# Reference-inspired alien

A simplified procedural interpretation of the supplied alien photograph, not an
exact reconstruction. Hidden surfaces and proportions are artistically inferred.
Includes an elongated dome, rib cage, dorsal tubes, crouching legs, claws and a
curved segmented tail. Static mesh with overlapping closed components; no rig.
Approximate height 1.86 m. Dimensions are artistic, not inferred measurements.

- `meshes/alien.dae`: colored COLLADA for Gazebo; metres, Z up, X forward.
- `meshes/alien.glb`: colored GLB for Rerun; metres, Y up.
- `preview.png`: front and side views.

Open from the repository root:

```bash
rerun models/alien_reference/meshes/alien.glb
```

For Gazebo, configure the models resource path as described in
`../person_standing/README.md`, then add this to your world's SDF:

```xml
<include>
  <uri>model://alien_reference</uri>
  <name>alien_obstacle</name>
  <pose>10 0 0 0 0 0</pose>
</include>
```

Both visual and collision geometry use the mesh. This is a geometric obstacle,
with no calibrated reflectance model. Simulator loading and LiDAR returns still
need testing in Gazebo.

Regenerate with `python3 tools/generate_alien.py`. The mesh generator uses the
Python standard library; matplotlib optionally produces the preview.
