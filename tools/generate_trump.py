"""Generate an stylized Donald Trump caricature COLLADA mesh (metres)."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/trump_stylized'
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


# Stylized caricature proportions; not a measurement of the real person.
palette = {'suit': (.035,.065,.14), 'skin': (.86,.53,.32),
           'hair': (.86,.68,.30), 'shirt': (.94,.94,.91),
           'tie': (.72,.025,.035), 'shoe': (.025,.025,.03),
           'eye': (.13,.20,.28), 'mouth': (.53,.25,.20)}
face_materials = []
def part(material, center, radii, tilt=0):
    before = len(faces)
    ellipsoid(center, radii, tilt)
    face_materials.extend([material]*(len(faces)-before))

part('suit', (0,0,1.25), (.19,.255,.31))
part('suit', (.025,0,1.035), (.185,.215,.25))
part('suit', (0,0,.89), (.16,.19,.16))
part('skin', (0,0,1.535), (.072,.077,.10))
part('skin', (.006,0,1.705), (.126,.112,.163))
part('skin', (.07,0,1.62), (.092,.096,.07))
part('skin', (.128,0,1.70), (.047,.03,.043))
for side in (-1,1):
    part('skin', (0,side*.111,1.705), (.033,.022,.05))
    part('skin', (.103,side*.064,1.68), (.037,.047,.048))
    part('shirt', (.128,side*.057,1.487), (.028,.043,.062), side*.45)
    part('suit', (0,side*.27,1.29), (.09,.09,.215),side*.10)
    part('suit', (.01,side*.30,1.04), (.074,.075,.17))
    part('shirt', (.016,side*.30,.90), (.065,.066,.035))
    part('skin', (.02,side*.30,.835), (.05,.056,.086))
    part('suit', (0,side*.107,.66), (.108,.096,.28))
    part('suit', (0,side*.113,.30), (.085,.077,.235))
    part('shoe', (.055,side*.113,.07), (.155,.082,.07))
    part('eye', (.12,side*.047,1.738), (.012,.017,.009))
    part('hair', (.116,side*.045,1.763), (.014,.035,.01),side*.12)
part('shirt', (.173,0,1.365), (.026,.071,.115))
part('tie', (.202,0,1.421), (.02,.03,.038))
part('tie', (.209,0,1.255), (.022,.035,.15))
part('mouth', (.153,0,1.642), (.009,.03,.007))
part('hair', (-.023,0,1.837), (.117,.12,.055))
# Layered side-swept fringe, asymmetric to suggest the characteristic hairstyle.
part('hair', (.059,-.024,1.844), (.091,.119,.045),-.15)
part('hair', (.087,.026,1.82), (.055,.086,.03),-.20)
part('hair', (-.065,-.096,1.755), (.068,.026,.095))
part('hair', (-.066,.096,1.766), (.065,.025,.078))

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
materials = node(doc, 'library_materials')
for name, color in palette.items():
    effect = node(effects, 'effect', id=name+'-effect')
    phong = node(node(node(effect, 'profile_COMMON'), 'technique', sid='common'), 'phong')
    node(node(phong, 'diffuse'), 'color', ' '.join(map(str, (*color,1))))
    material = node(materials, 'material', id=name+'-material', name=name)
    node(material, 'instance_effect', url='#'+name+'-effect')
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
for name in palette:
    group = [(i, f) for i,f in enumerate(faces) if face_materials[i] == name]
    triangles = node(mesh, 'triangles', count=str(len(group)), material=name)
    node(triangles, 'input', semantic='VERTEX', source='#vertices', offset='0')
    node(triangles, 'input', semantic='NORMAL', source='#normals', offset='1')
    node(triangles, 'p', ' '.join(f'{v} {i}' for i, face in group for v in face))
scene = node(node(doc, 'library_visual_scenes'), 'visual_scene', id='Scene')
instance = node(node(scene, 'node', id='Person', name='Person'), 'instance_geometry', url='#person')
bindings = node(node(instance, 'bind_material'), 'technique_common')
for name in palette:
    node(bindings, 'instance_material', symbol=name, target='#'+name+'-material')
node(node(doc, 'scene'), 'instance_visual_scene', url='#Scene')
(MODEL / 'meshes').mkdir(parents=True, exist_ok=True)
ET.indent(doc)
ET.ElementTree(doc).write(MODEL / 'meshes/trump.dae', encoding='utf-8', xml_declaration=True)
print(f'Created trump.dae: {len(vertices)} vertices, {len(faces)} triangles')

# A GLB copy preserves colors when opened directly in Rerun.
import json
import struct
blob = bytearray()
views, accessors, primitives = [], [], []
def buffer(data, component, kind, count, bounds=None):
    while len(blob) % 4:
        blob.append(0)
    views.append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(data)})
    blob.extend(data)
    accessors.append({'bufferView': len(views)-1, 'componentType': component,
                      'count': count, 'type': kind, **(bounds or {})})
    return len(accessors)-1

# glTF uses Y up. Convert (X,Y,Z) to (X,Z,-Y).
positions = [(x,z,-y) for x,y,z in vertices]
flat = [x for p in positions for x in p]
position = buffer(struct.pack('<'+'f'*len(flat), *flat), 5126, 'VEC3', len(positions),
                  {'min': [min(p[i] for p in positions) for i in range(3)],
                   'max': [max(p[i] for p in positions) for i in range(3)]})
for material_index, name in enumerate(palette):
    indices = [v for i,f in enumerate(faces) if face_materials[i] == name for v in f]
    accessor = buffer(struct.pack('<'+'I'*len(indices), *indices), 5125, 'SCALAR', len(indices))
    primitives.append({'attributes': {'POSITION': position}, 'indices': accessor,
                       'material': material_index})
gltf = {'asset': {'version': '2.0', 'generator': 'generate_trump.py'},
        'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0}],
        'meshes': [{'primitives': primitives}],
        'materials': [{'name': name, 'pbrMetallicRoughness': {
            'baseColorFactor': [*color,1], 'metallicFactor': 0, 'roughnessFactor': .85}}
            for name,color in palette.items()],
        'buffers': [{'byteLength': len(blob)}], 'bufferViews': views, 'accessors': accessors}
header = json.dumps(gltf, separators=(',', ':')).encode()
header += b' ' * (-len(header) % 4)
blob.extend(b'\0' * (-len(blob) % 4))
glb = (struct.pack('<III', 0x46546C67, 2, 28+len(header)+len(blob))
       + struct.pack('<II', len(header), 0x4E4F534A) + header
       + struct.pack('<II', len(blob), 0x004E4942) + blob)
(MODEL / 'meshes/trump.glb').write_bytes(glb)

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
                facecolor=[palette[m] for m in face_materials], edgecolor='none', linewidth=0))
            ax.set(xlim=(-.4, .4), ylim=(-.4, .4), zlim=(0, 1.95),
                   xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
            ax.set_box_aspect((.8, .8, 1.95))
            ax.view_init(elev=8, azim=azimuth)
        fig.suptitle('Donald Trump — stylized procedural caricature')
        fig.tight_layout()
        fig.savefig(MODEL / 'preview.png', dpi=150)
