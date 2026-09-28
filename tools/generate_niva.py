"""Generate an approximate classic three-door Niva obstacle; +X front, metres."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/niva_car'
vertices,faces,face_materials=[],[],[]
palette={'paint':(.43,.51,.26),'rubber':(.035,.04,.038),'glass':(.16,.25,.28),
         'chrome':(.66,.69,.68),'lamp':(.93,.92,.76),'amber':(.96,.38,.035),
         'red':(.65,.035,.02),'plate':(.86,.88,.82),'grille':(.07,.085,.075)}
def hull(material,points):
    points=np.asarray(points);h=ConvexHull(points);start=len(vertices);vertices.extend(points.tolist())
    for triangle,equation in zip(h.simplices,h.equations):
        a,b,c=triangle
        if np.dot(np.cross(points[b]-points[a],points[c]-points[a]),equation[:3])<0:b,c=c,b
        faces.append((start+int(a),start+int(b),start+int(c)));face_materials.append(material)
def box(material,center,size):
    hull(material,[[center[a]+size[a]*sign[a]/2 for a in range(3)] for sign in __import__('itertools').product((-1,1),repeat=3)])
def cylinder(material,center,radius,width,axis=1,n=32):
    others=[i for i in range(3) if i!=axis];points=[]
    for side in [-1,1]:
        for j in range(n):
            p=list(center);p[axis]+=side*width/2;p[others[0]]+=radius*math.cos(j*2*math.pi/n);p[others[1]]+=radius*math.sin(j*2*math.pi/n);points.append(p)
    hull(material,points)
def panel(material,points,axis,thickness=.012):
    hull(material,[[v+(side*thickness/2 if i==axis else 0) for i,v in enumerate(p)] for side in [-1,1] for p in points])
# Short hood, upright three-door cabin, flat roof and high ground clearance.
box('paint',(0,0,.77),(3.52,1.57,.50))
box('paint',(1.10,0,1.04),(1.32,1.54,.12))
hull('paint',[[x,side*w,z] for side in [-1,1] for x,w,z in [(-1.73,.77,.95),(.51,.77,.99),(.09,.68,1.59),(-1.57,.69,1.59)]])
box('paint',(-.75,0,1.61),(1.75,1.40,.055))
# Window patches follow cabin surfaces; opaque for current LiDAR rendering.
for side in [-1,1]:
    panel('rubber',[(.45,side*.781,1.045),(.04,side*.702,1.55),(-.75,side*.702,1.55),(-.75,side*.781,1.045)],1,.022)
    panel('glass',[(.37,side*.797,1.09),(.01,side*.72,1.51),(-.69,side*.72,1.51),(-.69,side*.797,1.09)],1)
    panel('rubber',[(-.84,side*.781,1.045),(-.84,side*.702,1.55),(-1.49,side*.707,1.55),(-1.64,side*.781,1.045)],1,.022)
    panel('glass',[(-.89,side*.797,1.10),(-.89,side*.72,1.51),(-1.45,side*.724,1.51),(-1.58,side*.797,1.10)],1)
    box('rubber',(-.77,side*.792,.78),(.014,.015,.46))
    box('chrome',(-.57,side*.807,1.005),(.16,.027,.035))
    box('rubber',(.43,side*.87,1.12),(.20,.16,.10))
    box('rubber',(0,side*.795,.57),(2.96,.026,.045))
    box('amber',(1.51,side*.796,1.005),(.16,.022,.065))
# Front glass on the sloped windscreen.
panel('rubber',[(.50,-.70,1.04),(.50,.70,1.04),(.14,.64,1.555),(.14,-.64,1.555)],0,.02)
panel('glass',[(.468,-.65,1.09),(.468,.65,1.09),(.17,.60,1.51),(.17,-.60,1.51)],0,.025)
panel('glass',[(-1.75,-.63,1.05),(-1.75,.63,1.05),(-1.60,.60,1.53),(-1.60,-.60,1.53)],0,.025)
# Round headlights, rectangular indicators, black grille, steel bumpers.
box('grille',(1.777,0,.84),(.05,1.42,.34))
for z in [.72,.77,.82,.87,.92]:box('rubber',(1.81,0,z),(.025,1.0,.012))
for side in [-1,1]:
    cylinder('chrome',(1.818,side*.56,.835),.135,.045,0)
    cylinder('lamp',(1.844,side*.56,.835),.108,.02,0)
    box('rubber',(1.79,side*.57,1.074),(.07,.26,.11))
    box('amber',(1.831,side*.57,1.074),(.018,.20,.068))
    box('red',(-1.785,side*.64,.86),(.04,.16,.24))
box('chrome',(1.86,0,.59),(.13,1.73,.12));box('chrome',(-1.85,0,.59),(.13,1.73,.12))
box('rubber',(1.932,0,.60),(.018,1.72,.042));box('plate',(1.944,0,.59),(.02,.36,.10))
# Four wheels, silver steel rims, hub caps and recessed dark ventilation holes.
for x in [-1.09,1.12]:
 for side in [-1,1]:
    cylinder('rubber',(x,side*.76,.35),.35,.24)
    cylinder('chrome',(x,side*.889,.35),.228,.023)
    cylinder('rubber',(x,side*.905,.35),.088,.014)
    cylinder('chrome',(x,side*.917,.35),.064,.026)
    for j in range(10):
        a=2*math.pi*j/10
        cylinder('rubber',(x+.18*math.cos(a),side*.905,.35+.18*math.sin(a)),.025,.014,n=10)
    for j in range(5):
        a=2*math.pi*j/5
        cylinder('chrome',(x+.115*math.cos(a),side*.921,.35+.115*math.sin(a)),.012,.02,n=8)

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
mesh = node(node(node(doc, 'library_geometries'), 'geometry', id='niva', name='niva'), 'mesh')

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
instance = node(node(scene, 'node', id='Niva', name='Niva'), 'instance_geometry', url='#niva')
bindings = node(node(instance, 'bind_material'), 'technique_common')
for name in palette:
    node(bindings, 'instance_material', symbol=name, target='#'+name+'-material')
node(node(doc, 'scene'), 'instance_visual_scene', url='#Scene')
(MODEL / 'meshes').mkdir(parents=True, exist_ok=True)
ET.indent(doc)
ET.ElementTree(doc).write(MODEL / 'meshes/niva.dae', encoding='utf-8', xml_declaration=True)
print(f'Created niva.dae: {len(vertices)} vertices, {len(faces)} triangles')
