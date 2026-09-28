"""Generate an original, simplified standing-person COLLADA mesh (metres)."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/person_standing'
vertices, faces = [], []


def ellipsoid(center, radii, tilt=0, segments=24, rings=16):
    """Closed ellipsoid, optionally rotated around X, with outward triangles."""
    start = len(vertices)
    def point(x, y, z):
        c, s = math.cos(tilt), math.sin(tilt)
        vertices.append((center[0] + x, center[1] + c*y-s*z,
                         center[2] + s*y+c*z))
    point(0, 0, radii[2])
    for j in range(1, rings):
        theta = math.pi*j/rings
        for i in range(segments):
            phi = 2*math.pi*i/segments
            point(radii[0]*math.sin(theta)*math.cos(phi),
                  radii[1]*math.sin(theta)*math.sin(phi), radii[2]*math.cos(theta))
    bottom = len(vertices)
    point(0, 0, -radii[2])
    for i in range(segments):
        nxt = (i+1) % segments
        faces.append((start, start+1+i, start+1+nxt))
        for j in range(rings-2):
            a, b = start+1+j*segments+i, start+1+j*segments+nxt
            faces.extend([(a, a+segments, b), (b, a+segments, b+segments)])
        last = start+1+(rings-2)*segments
        faces.append((last+i, bottom, last+nxt))


# +X forward, +Y left, +Z up; soles at Z=0; total height 1.75 m.
ellipsoid((0, 0, 1.625), (.098, .085, .125))  # head
ellipsoid((.089, 0, 1.625), (.031, .025, .034))  # nose / forward cue
ellipsoid((0, 0, 1.485), (.058, .057, .083))  # neck
ellipsoid((0, 0, 1.285), (.135, .225, .245))  # clothed chest
ellipsoid((0, 0, 1.075), (.125, .155, .205))  # waist
ellipsoid((0, 0, .94), (.14, .177, .15))  # pelvis
for side in (-1, 1):
    ellipsoid((0, side*.245, 1.29), (.077, .078, .185), side*.16)
    ellipsoid((.012, side*.283, 1.045), (.059, .061, .165), side*.07)
    ellipsoid((.025, side*.292, .865), (.04, .048, .09))  # hand
    ellipsoid((0, side*.105, .72), (.103, .087, .255))  # thigh
    ellipsoid((.012, side*.11, .495), (.075, .075, .085))
    ellipsoid((0, side*.112, .30), (.073, .063, .225))  # calf
    ellipsoid((.05, side*.112, .075), (.15, .071, .075))  # shoe

# Independent closed, overlapping body parts, intentionally no animation rig.
ns = 'http://www.collada.org/2005/11/COLLADASchema'
ET.register_namespace('', ns)
def node(parent, tag, text=None, **attrs):
    child = ET.SubElement(parent, '{'+ns+'}'+tag, attrs)
    if text is not None:
        child.text = str(text)
    return child

doc = ET.Element('{'+ns+'}COLLADA', version='1.4.1')
asset = node(doc, 'asset')
node(asset, 'created', '2026-09-22T00:00:00Z')
node(asset, 'modified', '2026-09-22T00:00:00Z')
node(asset, 'unit', name='meter', meter='1')
node(asset, 'up_axis', 'Z_UP')
effects = node(doc, 'library_effects')
effect = node(effects, 'effect', id='body-effect')
phong = node(node(node(effect, 'profile_COMMON'), 'technique', sid='common'), 'phong')
node(node(phong, 'diffuse'), 'color', '0.25 0.45 0.65 1')
material = node(node(doc, 'library_materials'), 'material', id='body-material', name='body')
node(material, 'instance_effect', url='#body-effect')
mesh = node(node(node(doc, 'library_geometries'), 'geometry', id='person', name='person'), 'mesh')

normals = []
for a, b, c in faces:
    u = [vertices[b][i]-vertices[a][i] for i in range(3)]
    v = [vertices[c][i]-vertices[a][i] for i in range(3)]
    n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    length = math.sqrt(sum(x*x for x in n))
    assert length > 1e-12
    normals.append(tuple(x/length for x in n))
for name, data in [('positions', vertices), ('normals', normals)]:
    source = node(mesh, 'source', id=name)
    node(source, 'float_array', ' '.join(f'{x:.8f}' for row in data for x in row),
         id=name+'-array', count=str(len(data)*3))
    accessor = node(node(source, 'technique_common'), 'accessor',
                    source='#'+name+'-array', count=str(len(data)), stride='3')
    for axis in 'XYZ':
        node(accessor, 'param', name=axis, type='float')
node(node(mesh, 'vertices', id='vertices'), 'input', semantic='POSITION', source='#positions')
triangles = node(mesh, 'triangles', count=str(len(faces)), material='body')
node(triangles, 'input', semantic='VERTEX', source='#vertices', offset='0')
node(triangles, 'input', semantic='NORMAL', source='#normals', offset='1')
node(triangles, 'p', ' '.join(f'{v} {i}' for i, face in enumerate(faces) for v in face))
scene = node(node(doc, 'library_visual_scenes'), 'visual_scene', id='Scene')
instance = node(node(scene, 'node', id='Person', name='Person'), 'instance_geometry', url='#person')
node(node(node(instance, 'bind_material'), 'technique_common'), 'instance_material',
     symbol='body', target='#body-material')
node(node(doc, 'scene'), 'instance_visual_scene', url='#Scene')
(MODEL / 'meshes').mkdir(parents=True, exist_ok=True)
ET.indent(doc)
ET.ElementTree(doc).write(MODEL / 'meshes/person.dae', encoding='utf-8', xml_declaration=True)
print(f'Created person.dae: {len(vertices)} vertices, {len(faces)} triangles')

if __name__ == '__main__':
    # Optional preview dependency only; generation above uses the standard library.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        pass
    else:
        fig = plt.figure(figsize=(8, 7))
        for index, azimuth in enumerate((0, 55), 1):
            ax = fig.add_subplot(1, 2, index, projection='3d')
            ax.add_collection3d(Poly3DCollection(
                [[vertices[i] for i in face] for face in faces],
                facecolor='#4779a6', edgecolor='none', linewidth=0))
            ax.set(xlim=(-.4, .4), ylim=(-.4, .4), zlim=(0, 1.8),
                   xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
            ax.set_box_aspect((.8, .8, 1.8))
            ax.view_init(elev=8, azim=azimuth)
        fig.suptitle('Standing person • 1.75 m • +X forward')
        fig.tight_layout()
        fig.savefig(MODEL / 'preview.png', dpi=150)
