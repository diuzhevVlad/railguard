"""Closed colored primitives and COLLADA/GLB output for procedural people."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

def sub(a,b): return tuple(x-y for x,y in zip(a,b))

def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])

def unit(v):
    length=math.sqrt(sum(x*x for x in v))
    return tuple(x/length for x in v)

class PersonMesh:
    def __init__(self, palette):
        self.vertices, self.faces, self.face_materials = [], [], []
        self.palette = palette

    def ellipsoid(self, center, radii, tilt=0, segments=24, rings=16):
        """Closed ellipsoid, optionally rotated around X, with outward triangles."""
        start = len(self.vertices)
        def point(x, y, z):
            c, s = math.cos(tilt), math.sin(tilt)
            self.vertices.append((center[0] + x, center[1] + c*y-s*z,
                             center[2] + s*y+c*z))
        point(0, 0, radii[2])
        for j in range(1, rings):
            theta = math.pi*j/rings
            for i in range(segments):
                phi = 2*math.pi*i/segments
                point(radii[0]*math.sin(theta)*math.cos(phi),
                      radii[1]*math.sin(theta)*math.sin(phi), radii[2]*math.cos(theta))
        bottom = len(self.vertices)
        point(0, 0, -radii[2])
        for i in range(segments):
            nxt = (i+1) % segments
            self.faces.append((start, start+1+i, start+1+nxt))
            for j in range(rings-2):
                a, b = start+1+j*segments+i, start+1+j*segments+nxt
                self.faces.extend([(a, a+segments, b), (b, a+segments, b+segments)])
            last = start+1+(rings-2)*segments
            self.faces.append((last+i, bottom, last+nxt))

    def part(self, material, center, radii, tilt=0):
        before = len(self.faces)
        self.ellipsoid(center, radii, tilt, segments=24, rings=12)
        self.face_materials.extend([material]*(len(self.faces)-before))

    def tube(self, material, points, radii, sides=12):
        """Capped curved tube, with consistently outward triangle winding."""
        start=len(self.vertices)
        before=len(self.faces)
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
                self.vertices.append(tuple(p[k]+radii[i]*(math.cos(angle)*u[k]+math.sin(angle)*v[k]) for k in range(3)))
        for i in range(len(points)-1):
            for j in range(sides):
                a=start+i*sides+j; b=start+i*sides+(j+1)%sides
                self.faces.extend([(a,b,a+sides),(b,b+sides,a+sides)])
        cap0=len(self.vertices); self.vertices.extend([points[0],points[-1]])
        last=start+(len(points)-1)*sides
        for j in range(sides):
            nxt=(j+1)%sides
            self.faces.extend([(cap0,start+nxt,start+j),(cap0+1,last+j,last+nxt)])
        self.face_materials.extend([material]*(len(self.faces)-before))

    def bone(self, material,a,b,r):
        self.tube(material,[a,tuple((x+y)/2 for x,y in zip(a,b)),b],[r*.72,r,r*.65],16)

    def loft(self, material, levels, sides=24):
        """Closed garment surface from elliptical horizontal cross sections."""
        start=len(self.vertices); before=len(self.faces)
        for x,y,z,rx,ry in levels:
            for i in range(sides):
                a=2*math.pi*i/sides
                self.vertices.append((x+rx*math.cos(a), y+ry*math.sin(a),z))
        for k in range(len(levels)-1):
            for i in range(sides):
                a=start+k*sides+i; b=start+k*sides+(i+1)%sides
                self.faces.extend([(a,b,a+sides),(b,b+sides,a+sides)])
        cap=len(self.vertices)
        self.vertices.extend([levels[0][:3],levels[-1][:3]])
        last=start+(len(levels)-1)*sides
        for i in range(sides):
            j=(i+1)%sides
            self.faces.extend([(cap,start+j,start+i),(cap+1,last+i,last+j)])
        self.face_materials.extend([material]*(len(self.faces)-before))

def save_mesh(mesh, directory):
    MODEL = Path(directory)
    vertices, faces, face_materials = mesh.vertices, mesh.faces, mesh.face_materials
    palette = {k:v for k,v in mesh.palette.items() if k in set(face_materials)}
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
    gltf = {'asset': {'version': '2.0', 'generator': 'generate_itl_people.py'},
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


