"""Generate a photo-inspired casually dressed person COLLADA mesh (metres)."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/ITL-RailGuard_alexander'
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


# Outfit and pose inspired by the provided photo; scale is an artistic choice.
palette = {'shirt': (.29,.28,.31), 'shorts': (.79,.77,.67),
           'skin': (.76,.58,.43), 'cap': (.24,.13,.095),
           'hair': (.10,.065,.04), 'bag': (.025,.035,.032),
           'white': (.89,.91,.89), 'shoe': (.30,.36,.36),
           'bottle': (.12,.65,.74), 'eye': (.07,.055,.04),
           'mouth': (.44,.25,.20)}
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
    previous_u=None
    for i,p in enumerate(points):
        tangent=unit(sub(points[min(i+1,len(points)-1)],points[max(i-1,0)]))
        ref=(0,0,1) if abs(tangent[2])<.9 else (0,1,0)
        u=unit(cross(ref,tangent))
        if previous_u is not None:
            dot=sum(a*b for a,b in zip(previous_u,tangent))
            projected=tuple(previous_u[k]-dot*tangent[k] for k in range(3))
            if sum(x*x for x in projected)>1e-8:
                u=unit(projected)
        previous_u=u
        v=cross(tangent,u)
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

def loft(material, levels, sides=24):
    """Closed garment surface from elliptical horizontal cross sections."""
    start=len(vertices); before=len(faces)
    for x,y,z,rx,ry in levels:
        for i in range(sides):
            a=2*math.pi*i/sides
            vertices.append((x+rx*math.cos(a), y+ry*math.sin(a),z))
    for k in range(len(levels)-1):
        for i in range(sides):
            a=start+k*sides+i; b=start+k*sides+(i+1)%sides
            faces.extend([(a,b,a+sides),(b,b+sides,a+sides)])
    cap=len(vertices)
    vertices.extend([levels[0][:3],levels[-1][:3]])
    last=start+(len(levels)-1)*sides
    for i in range(sides):
        j=(i+1)%sides
        faces.extend([(cap,start+j,start+i),(cap+1,last+i,last+j)])
    face_materials.extend([material]*(len(faces)-before))

def strap(points,width=.045,depth=.006):
    start=len(vertices); before=len(faces)
    for i,p in enumerate(points):
        tangent=unit(sub(points[min(i+1,len(points)-1)],points[max(i-1,0)]))
        ref=(1,0,0) if abs(tangent[0])<.9 else (0,0,1)
        u=unit(cross(ref,tangent)); v=cross(tangent,u)
        for su,sv in [(1,1),(-1,1),(-1,-1),(1,-1)]:
            vertices.append(tuple(p[k]+su*width/2*u[k]+sv*depth*v[k] for k in range(3)))
    for k in range(len(points)-1):
        for i in range(4):
            a=start+4*k+i; b=start+4*k+(i+1)%4
            faces.extend([(a,b,a+4),(b,b+4,a+4)])
    faces.extend([(start,start+2,start+1),(start,start+3,start+2)])
    e=start+4*(len(points)-1)
    faces.extend([(e,e+1,e+2),(e,e+2,e+3)])
    face_materials.extend(['bag']*(len(faces)-before))

# Loose T-shirt, with a small plain chest patch.
loft('shirt',[(0,0,.89,.14,.205),(0,0,.95,.145,.215),(0,0,1.13,.135,.19),
              (0,0,1.35,.15,.215),(0,0,1.43,.13,.22),(0,0,1.49,.08,.105)])
part('white',(.153,-.032,1.345),(.006,.036,.036))
part('skin',(0,0,1.52),(.057,.061,.10))
# Bent right arm holds the strap across the torso.
tube('shirt',[(0,-.19,1.42),(.015,-.245,1.33),(.015,-.275,1.23)],[.083,.088,.079],20)
bone('skin',(.01,-.275,1.24),(.11,-.29,1.17),.06)
part('skin',(.11,-.29,1.17),(.061,.06,.06))
bone('skin',(.11,-.29,1.17),(.205,-.04,1.225),.052)
part('skin',(.205,-.04,1.225),(.035,.063,.032),-.25)
for i in range(4):
    bone('skin',(.222,-.073+i*.020,1.234),(.221,-.027+i*.019,1.242),.008)
# Left arm hangs down with a bottle.
tube('shirt',[(0,.19,1.42),(.005,.245,1.32),(.005,.265,1.22)],[.083,.088,.078],20)
bone('skin',(.005,.265,1.23),(.025,.285,1.06),.054)
part('skin',(.025,.285,1.06),(.052,.051,.053))
bone('skin',(.025,.285,1.06),(.07,.295,.88),.044)
part('skin',(.074,.295,.855),(.037,.045,.057))
tube('bottle',[(.099,.302,.765),(.099,.302,.825),(.099,.302,.89)],[.037,.038,.03],20)
tube('white',[(.099,.302,.902),(.099,.302,.914)],[.018,.018],16)
tube('white',[(.099,.302,.814),(.099,.302,.844)],[.039,.039],20)
# Crossbody bag and flat diagonal strap.
part('bag',(.13,-.15,1.02),(.068,.142,.103),-.20)
strap([(.125,-.17,1.04),(.161,-.11,1.15),(.164,-.035,1.29),
       (.154,.065,1.41),(.075,.15,1.49),(-.05,.15,1.49),
       (-.13,.10,1.37),(-.15,0,1.22),(-.13,-.13,1.05)],.047)
# Shorts with two separate, broad hems.
part('shorts',(-.005,0,.865),(.14,.177,.11))
loft('shorts',[(.045,.104,.57,.113,.108),(.024,.10,.72,.116,.108),(0,.097,.90,.13,.114)])
loft('shorts',[(-.09,-.102,.59,.105,.103),(-.045,-.10,.74,.12,.108),(0,-.097,.90,.13,.114)])
# Walking pose: forward left leg, trailing right leg.
for side,knee,ankle in [(1,(.055,.11,.53),(.125,.11,.14)),
                        (-1,(-.10,-.11,.55),(-.235,-.11,.18))]:
    part('skin',knee,(.07,.069,.095))
    bone('skin',knee,ankle,.060)
    tube('white',[(ankle[0],ankle[1],ankle[2]-.04),ankle,
                  (ankle[0]-.015,ankle[1],ankle[2]+.10)],[.05,.051,.052],20)
    foot=(ankle[0]+.055,ankle[1],.06 if side==1 else .105)
    part('shoe',foot,(.14,.066,.056))
    part('white',(foot[0],foot[1],foot[2]-.029),(.143,.069,.027))
    part('white',(foot[0]+.04,foot[1],foot[2]+.025),(.09,.054,.035))
    for j in range(4):
        x=foot[0]-.04+j*.023
        tube('white',[(x,foot[1]-.035,foot[2]+.046),(x+.01,foot[1]+.035,foot[2]+.046)],[.005,.005],8)
# Head turned slightly to the person's left; simplified face and baseball cap.
head_start=len(vertices)
part('skin',(.006,0,1.664),(.095,.082,.131))
part('skin',(.04,0,1.591),(.072,.071,.054))
part('hair',(-.037,0,1.705),(.078,.087,.087))
for side in (-1,1):
    part('skin',(0,side*.082,1.665),(.023,.017,.034))
    part('eye',(.094,side*.036,1.691),(.008,.013,.007))
    part('hair',(.09,side*.035,1.711),(.009,.025,.007),side*.12)
part('skin',(.101,0,1.668),(.032,.023,.03))
part('mouth',(.102,0,1.623),(.009,.032,.012))
part('white',(.110,0,1.627),(.004,.025,.005))
part('cap',(-.012,0,1.76),(.11,.101,.071))
part('cap',(.095,0,1.754),(.138,.104,.016))
part('white',(.074,0,1.796),(.013,.016,.007))
yaw=.33
for i in range(head_start,len(vertices)):
    x,y,z=vertices[i]
    vertices[i]=(x*math.cos(yaw)-y*math.sin(yaw),x*math.sin(yaw)+y*math.cos(yaw),z)
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
ET.ElementTree(doc).write(MODEL / 'meshes/person.dae', encoding='utf-8', xml_declaration=True)
print(f'Created person.dae: {len(vertices)} vertices, {len(faces)} triangles')

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
gltf = {'asset': {'version': '2.0', 'generator': 'generate_photo_person.py'},
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
(MODEL / 'meshes/person.glb').write_bytes(glb)

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
            preview_colors.append(tuple(min(1,c*shade*1.15+.025) for c in palette[material]))
        fig = plt.figure(figsize=(9, 8))
        for index, azimuth in enumerate((0, 55), 1):
            ax = fig.add_subplot(1, 2, index, projection='3d')
            ax.add_collection3d(Poly3DCollection(
                [[vertices[i] for i in face] for face in faces],
                facecolor=preview_colors, edgecolor='none', linewidth=0))
            ax.set(xlim=(-.4, .4), ylim=(-.4, .4), zlim=(0, 1.9),
                   xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
            ax.set_box_aspect((.8, .8, 1.9))
            ax.view_init(elev=12, azim=azimuth)
        fig.suptitle('Alexander • simplified procedural mesh')
        fig.tight_layout()
        fig.savefig(MODEL / 'preview.png', dpi=150)
