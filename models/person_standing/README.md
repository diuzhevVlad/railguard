# Standing person obstacle

`meshes/person.dae` is an original procedural COLLADA 1.4.1 mesh. Height: 1.75 m.
Units: metres. Z points up, X points forward; origin lies on the ground between
the feet. The model is a static, simplified clothed human with separated legs
and arms at its sides. No textures, external assets, animation or rig are needed.

Both visual and collision geometry reference the mesh, so the model provides
geometry for rendering-based and collision-based LiDAR implementations. Actual
returns depend on the sensor and rendering/physics configuration; validate these
in your simulator. This is a geometric proxy, not a calibrated human reflectance
model. The body consists of overlapping closed surfaces, not one fused surface.

## Insert into Gazebo

From the repository root, set the resource path for your Gazebo version:

```bash
# Gazebo Sim (gz)
export GZ_SIM_RESOURCE_PATH="$PWD/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
# Ignition Gazebo / Fortress (ign)
export IGN_GAZEBO_RESOURCE_PATH="$PWD/models${IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}"
# Gazebo Classic
export GAZEBO_MODEL_PATH="$PWD/models${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"
```

Add this inside the `<world>` element of your existing SDF world, then launch
that world normally. This places the person 10 m along the world's X axis:

```xml
<include>
  <uri>model://person_standing</uri>
  <name>person_01</name>
  <pose>10 0 0 0 0 0</pose>
</include>
```

Pose is `x y z roll pitch yaw`, with angles in radians. Adjust Z to the track
surface height. Use unique names for additional instances.

Geometry/XML have been checked locally; Gazebo is unavailable here, so
simulator loading and sensor returns have not been tested.

Format reference: https://gazebosim.org/api/sim/7/meshtofuel.html
