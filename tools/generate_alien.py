"""Generate a reference-inspired alien COLLADA mesh (metres)."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/alien_reference'
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


# Reference-inspired crouching biomechanical alien; dimensions are artistic.
palette = {'shell': (.075,.09,.10), 'ribs': (.18,.20,.19),
           'joint': (.035,.042,.045), 'claw': (.36,.38,.33),
           'teeth': (.62,.64,.57)}
face_materials = []
def part(material, center, radii, tilt=0):
    before = len(faces)
    ellipsoid(center, radii, tilt, segments=24, rings=12)
    face_materials.extend([material]*(len(faces)-before))

def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def unit(v):
    length=math.sqrt(sum(x*x for x in v))
    return tuple(x/length for x in v)

def tube(material, points, radii, sides=12):
    """Capped curved tube, with consistently outward triangle winding."""
    start=len(vertices)
    before=len(faces)
    for i,p in enumerate(points):
        tangent=unit(sub(points[min(i+1,len(points)-1)],points[max(i-1,0)]))
        ref=(0,0,1) if abs(tangent[2])<.9 else (0,1,0)
        u=unit(cross(ref,tangent)); v=cross(tangent,u)
        for j in range(sides):
            angle=2*math.pi*j/sides
            vertices.append(tuple(p[k]+radii[i]*(math.cos(angle)*u[k]+math.sin(angle)*v[k]) for k in range(3)))
    for i in range(len(points)-1):
        for j in range(sides):
            a=start+i*sides+j; b=start+i*sides+(j+1)%sides
            faces.extend([(a,b,a+sides),(b,b+sides,a+sides)])
    cap0=len(vertices); vertices.extend([points[0],points[-1]])
    last=start+(len(points)-1)*sides
    for j in range(sides):
        nxt=(j+1)%sides
        faces.extend([(cap0,start+nxt,start+j),(cap0+1,last+j,last+nxt)])
    face_materials.extend([material]*(len(faces)-before))

def bone(material,a,b,r):
    tube(material,[a,tuple((x+y)/2 for x,y in zip(a,b)),b],[r*.72,r,r*.65],16)

# Hunched torso, narrow waist and armor plates.
part('shell',(.04,0,1.24),(.18,.24,.30))
part('joint',(-.04,0,1.02),(.13,.145,.20))
part('shell',(-.10,0,.90),(.16,.19,.14))
bone('joint',(.09,0,1.43),(.23,0,1.64),.095)
# Long rearward dome: +X is the face, rear extends towards -X.
part('shell',(-.035,0,1.72),(.405,.145,.158))
part('shell',(.265,0,1.645),(.135,.112,.105))
part('joint',(.356,0,1.605),(.041,.082,.028))
part('ribs',(.31,0,1.568),(.077,.091,.025))
for y in [-.065,-.045,-.025,-.005,.015,.035,.055]:
    bone('teeth',(.393,y,1.618),(.397,y,1.596),.006)
    bone('teeth',(.38,y,1.583),(.392,y,1.600),.005)
# Longitudinal crest lines follow the dome silhouette.
for side in (-1,1):
    for offset in (.055,.105):
        pts=[]
        for i in range(19):
            t=.18+(math.pi-.36)*i/18
            pts.append((-.035+.39*math.cos(t),side*offset,1.72+.15*math.sin(t)*math.sqrt(1-(offset/.15)**2)))
        tube('ribs',pts,[.009]*len(pts),8)
# Raised rib cage sweeps around both sides, with central chest armor.
for row in range(7):
    z=1.10+row*.055
    width=.18+.045*math.sin(row/6*math.pi)
    for side in (-1,1):
        pts=[]
        for i in range(13):
            t=math.pi*i/12
            pts.append((.025+.178*math.cos(t),side*width*math.sin(t),z+.035*math.sin(t)))
        tube('ribs',pts,[.014]*len(pts),8)
bone('shell',(.216,0,1.12),(.216,0,1.42),.037)
for z in [1.02+i*.056 for i in range(9)]:
    part('ribs',(-.145,0,z),(.035,.052,.025))

for side in (-1,1):
    shoulder=(.02,side*.26,1.40)
    elbow=(.10,side*.37,1.12)
    wrist=(.34,side*.39,.83)
    part('shell',shoulder,(.12,.115,.12))
    bone('shell',shoulder,elbow,.075)
    part('joint',elbow,(.078,.078,.078))
    bone('shell',elbow,wrist,.066)
    bone('ribs',(.15,side*.414,1.09),(.34,side*.43,.85),.023)
    part('shell',(.36,side*.39,.77),(.06,.074,.094))
    for finger in range(4):
        y=side*(.337+finger*.035)
        pts=[(.38,y,.75),(.40,y+side*.018,.65),(.43,y+side*.018,.59),(.475,y,.62)]
        tube('shell',pts,[.015,.014,.010,.004],8)
        tube('claw',pts[-2:],[.010,.0015],8)
    tube('shell',[(.35,side*.33,.79),(.42,side*.30,.73),(.46,side*.30,.70)],[.02,.016,.003],10)
    hip=(-.10,side*.13,.91); knee=(.22,side*.25,.59); ankle=(-.01,side*.25,.18)
    bone('shell',hip,knee,.103)
    part('ribs',knee,(.10,.094,.09))
    bone('shell',knee,ankle,.07)
    bone('ribs',(.25,side*.29,.56),(.015,side*.29,.20),.021)
    bone('shell',ankle,(.16,side*.25,.07),.056)
    for toe in range(3):
        y=side*(.185+toe*.065)
        tube('shell',[(.08,y,.075),(.24,y,.045),(.34,y,.024)],[.030,.024,.008],10)
        tube('claw',[(.31,y,.03),(.39,y,.018)],[.016,.002],8)
    # Back-mounted swept breathing tubes.
    for j in range(2):
        y=side*(.11+j*.09)
        pts=[(-.12,y,1.32),(-.29,y,1.48),(-.38,y,1.62),(-.43,y,1.60)]
        tube('shell',pts,[.052,.047,.032,.023],12)
        part('joint',pts[-1],(.025,.026,.028))

# Long tapering tail curves behind the crouching body and out to one side.
pts=[]
for i in range(45):
    t=i/44
    pts.append((-.21-1.30*math.sin(t*math.pi*.80),.80*t*t,.88-.70*math.sin(t*math.pi*.65)))
radii=[.070*(1-i/44)+.008 for i in range(45)]
tube('shell',pts,radii,16)
for i in range(2,42,2):
    p=pts[i]; r=radii[i]
    part('ribs',p,(r*1.15,r*1.15,r*.8))
    tube('ribs',[(p[0],p[1],p[2]+r*.5),(p[0]-.025,p[1],p[2]+r+.035)],[r*.40,.002],8)
last=pts[-1]
tube('claw',[last,(last[0]+.09,last[1]+.055,last[2]+.04),(last[0]+.22,last[1]+.07,last[2]+.09)],[.026,.038,.001],12)
# Ground the lowest point of the feet.
lowest=min(v[2] for v in vertices)
vertices[:]=[(x,y,z-lowest) for x,y,z in vertices]

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
ET.ElementTree(doc).write(MODEL / 'meshes/alien.dae', encoding='utf-8', xml_declaration=True)
print(f'Created alien.dae: {len(vertices)} vertices, {len(faces)} triangles')

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
gltf = {'asset': {'version': '2.0', 'generator': 'generate_alien.py'},
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
(MODEL / 'meshes/alien.glb').write_bytes(glb)

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
        preview_colors=[]
        light=unit((.7,-.5,1))
        for material,n in zip(face_materials,normals):
            shade=.50+.50*max(0,sum(a*b for a,b in zip(n,light)))
            preview_colors.append(tuple(min(1,c*shade*1.8+.045) for c in palette[material]))
        fig = plt.figure(figsize=(12, 7))
        for index, azimuth in enumerate((0, 95), 1):
            ax = fig.add_subplot(1, 2, index, projection='3d')
            ax.add_collection3d(Poly3DCollection(
                [[vertices[i] for i in face] for face in faces],
                facecolor=preview_colors, edgecolor='none', linewidth=0))
            ax.set(xlim=(-1.6, .6), ylim=(-.6, 1.0), zlim=(0, 1.95),
                   xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
            ax.set_box_aspect((2.2, 1.6, 1.95))
            ax.view_init(elev=12, azim=azimuth)
        fig.suptitle('Reference-inspired alien • procedural obstacle mesh')
        fig.tight_layout()
        fig.savefig(MODEL / 'preview.png', dpi=150)
