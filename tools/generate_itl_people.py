"""Generate approximate colored simulation people from the named photo references.

No photogrammetry or identity estimation: clothing/poses are manually described.
All four models use a chosen height of 1.80 m; photos do not establish metric scale.
"""
from pathlib import Path
import argparse
import math
import numpy as np
from person_mesh import PersonMesh, save_mesh

ROOT=Path(__file__).resolve().parents[1]
PROFILES={
 'dmitry':dict(photo='dimtry.jpg',coat=(.24,.25,.28),shirt=(.86,.87,.88),pants=(.025,.038,.065),hair=(.055,.038,.027),shoe=(.025,.021,.019),pose='down',description='Gray blazer, pale buttoned shirt, dark trousers, glasses and black leather shoes.'),
 'leonid':dict(photo='leonid.jpg',coat=(.85,.86,.80),shirt=(.93,.92,.88),pants=(.025,.031,.038),hair=(.075,.058,.042),shoe=(.72,.70,.64),pose='crossed',description='Open white shirt with navy leaf motifs, white T-shirt, black trousers, crossed arms and pale sneakers.'),
 'vasily':dict(photo='vasily.jpg',coat=(.54,.56,.48),shirt=(.54,.56,.48),pants=(.49,.45,.31),hair=(.085,.061,.036),shoe=(.025,.025,.023),pose='pockets',description='Long hair, light gray-green buttoned shirt, beige trousers, mustard backpack and black shoes.'),
 'vladislav':dict(photo='vladislav.jpg',coat=(.025,.031,.041),shirt=(.92,.93,.93),pants=(.13,.23,.34),hair=(.045,.031,.018),shoe=(.06,.075,.079),pose='pockets',description='Black jacket, white T-shirt, blue jeans, dark sneakers and hands in jacket pockets.'),
}


def build_person(name):
    p=PROFILES[name]
    palette={k:p[k] for k in ('coat','shirt','pants','hair','shoe')}
    palette.update(skin=(.73,.54,.42),lip=(.43,.24,.20),eye=(.13,.095,.065),
        white=(.87,.88,.86),navy=(.035,.10,.23),strap=(.60,.43,.17),
        metal=(.36,.37,.38),button=(.065,.065,.068),lapel=tuple(min(1,x*1.18) for x in p['coat']))
    m=PersonMesh(palette)
    # +X is forward. Two separate trouser legs with capped hems.
    for sign in (-1,1):
        y=sign*.095
        m.loft('pants',[(0,y,.13,.063,.071),(-.012,y,.26,.066,.079),
            (.012,y,.47,.079,.084),(0,y,.57,.083,.09),(0,y,.76,.105,.102),(0,y,.91,.105,.103)])
        m.part('shoe',(.052,y,.063),(.145,.074,.063))
        if name in ('leonid','vladislav'):
            m.part('white',(.052,y,.025),(.147,.076,.018))
            for z,x in ((.092,.04),(.092,.07),(.086,.10)):
                m.bone('white',(x,y-.038,z),(x,y+.038,z),.003)
    m.part('pants',(0,0,.875),(.116,.194,.125))
    # Torso: under-shirt, then open jacket panels (or a single buttoned shirt).
    m.loft('shirt',[(0,0,.88,.12,.165),(0,0,1.00,.121,.166),(0,0,1.20,.118,.18),
        (0,0,1.39,.122,.208),(0,0,1.44,.105,.192),(0,0,1.49,.058,.078)])
    if name!='vasily':
        for sign in (-1,1):
            m.loft('coat',[(-.005,sign*.13,.87,.124,.072),(-.005,sign*.132,1.03,.13,.075),
                (-.012,sign*.142,1.23,.135,.082),(-.015,sign*.14,1.41,.134,.088),
                (-.025,sign*.09,1.48,.083,.058)])
            m.tube('lapel',[(.13,sign*.074,.94),(.138,sign*.075,1.22),(.105,sign*.092,1.41),(.075,sign*.06,1.49)],
                [.009,.011,.022,.015],8)
    if name in ('dmitry','vasily'):
        for z in np.arange(.94,1.42,.075):m.part('button',(.124,0,float(z)),(.004,.006,.006))
        for sign in (-1,1):
            m.bone('shirt',(.064,sign*.035,1.49),(.123,sign*.060,1.39),.018)
        m.loft('button',[(0,0,.87,.122,.194),(0,0,.899,.124,.192)])
        m.part('metal',(.148,0,.885),(.009,.03,.02))
        m.part('button',(.16,0,.885),(.005,.023,.013))
    # Pose-specific arms.
    for sign in (-1,1):
        shoulder=(0,sign*.202,1.405)
        if p['pose']=='down':
            elbow=(0,sign*.244,1.17);wrist=(.028,sign*.246,.93)
            m.tube('coat',[shoulder,elbow,wrist],[.083,.069,.052],20)
            m.part('shirt',wrist,(.049,.052,.031))
            m.part('skin',(.038,sign*.248,.865),(.042,.043,.078),sign*.05)
        elif p['pose']=='crossed':
            sleeve=(.03,sign*.238,1.265);elbow=(.125,sign*.23,1.16)
            m.tube('coat',[shoulder,sleeve],[.089,.083],20)
            m.bone('skin',sleeve,elbow,.065)
            wrist=(.20,-sign*.155,1.20+sign*.035)
            m.bone('skin',elbow,wrist,.057)
            m.part('skin',(.172,-sign*.18,1.245+sign*.035),(.04,.065,.027))
        else:
            elbow=(-.018,sign*.286,1.17);wrist=(.083,sign*.176,1.015 if name=='vladislav' else .93)
            m.tube('coat',[shoulder,elbow,wrist],[.086,.078,.052],20)
            if name=='vasily':m.part('skin',(.084,sign*.171,.913),(.035,.038,.047))
            else:m.bone('lapel',(.114,sign*.16,1.01),(.10,sign*.20,1.08),.007)
    # Neck, jaw and head. Features are illustrative rather than reconstructed.
    m.part('skin',(0,0,1.50),(.053,.054,.085))
    m.loft('skin',[(.014,0,1.535,.054,.052),(.012,0,1.575,.073,.069),
        (0,0,1.65,.084,.087),(-.007,0,1.725,.08,.083),(-.015,0,1.76,.06,.064)])
    for sign in (-1,1):
        m.part('skin',(-.003,sign*.087,1.654),(.023,.017,.04))
        m.part('white',(.077,sign*.034,1.675),(.010,.021,.009))
        m.part('eye',(.087,sign*.034,1.675),(.004,.008,.008))
        m.bone('hair',(.079,sign*.016,1.699),(.073,sign*.056,1.702),.005)
    m.part('skin',(.089,0,1.646),(.027,.016,.031))
    m.part('skin',(.109,0,1.630),(.014,.020,.013))
    m.part('lip',(.079,0,1.597),(.009,.031,.006))
    m.part('skin',(.071,0,1.575),(.013,.033,.016))
    # Hair cap stays behind the forehead; strands establish each silhouette.
    m.part('hair',(-.025,0,1.744),(.078,.087,.047))
    if name=='vasily':
        m.part('hair',(-.083,0,1.615),(.043,.087,.155))
        for sign in (-1,1):
            for j in range(4):
                y=sign*(.074+j*.008)
                m.tube('hair',[(-.005,y,1.746),(-.01,y*1.13,1.65),(-.01,y*1.17,1.50)], [.015,.02,.014],10)
    elif name=='vladislav':
        rng=np.random.default_rng(41)
        for _ in range(20):
            x,y=rng.uniform(-.04,.052),rng.uniform(-.07,.07)
            m.part('hair',(float(x),float(y),float(rng.uniform(1.744,1.778))),(.026,.028,.025))
    else:
        for j in range(7):
            y=-.065+j*.022
            m.tube('hair',[(-.062,y,1.75),(.00,y-.014,1.783),(.065,y-.025,1.752)], [.012,.017,.01],10)
    if name=='dmitry':
        for sign in (-1,1):
            points=[(.091,sign*.015,1.692),(.081,sign*.067,1.692),(.080,sign*.068,1.659),(.091,sign*.014,1.659),(.091,sign*.015,1.692)]
            m.tube('metal',points,[.0022]*len(points),8)
            m.bone('metal',(.082,sign*.069,1.687),(-.024,sign*.092,1.683),.002)
        m.bone('metal',(.096,-.015,1.684),(.096,.015,1.684),.002)
    if name=='vasily':
        m.part('strap',(-.155,0,1.195),(.087,.181,.245))
        for sign in (-1,1):
            m.tube('strap',[(-.12,sign*.14,1.20),(-.10,sign*.17,1.40),(0,sign*.17,1.47),(.117,sign*.15,1.30),(.12,sign*.145,1.10)], [.018,.021,.025,.023,.018],10)
    if name=='leonid':
        # Raised navy motifs approximate the shirt's visible leaf pattern.
        for sign in (-1,1):
            for z in (1.00,1.14,1.31,1.42):
                y=sign*.17
                def surface(yy,zz):
                    levels=np.array([[.87,-.005,.13,.124,.072],[1.03,-.005,.132,.13,.075],
                        [1.23,-.012,.142,.135,.082],[1.41,-.015,.14,.134,.088],
                        [1.48,-.025,.09,.083,.058]])
                    cx,cy,rx,ry=[np.interp(zz,levels[:,0],levels[:,i]) for i in range(1,5)]
                    x=cx+rx*math.sqrt(max(.01,1-((yy-sign*cy)/ry)**2))+.002
                    return (x,yy,zz)
                m.bone('navy',surface(y,z-.045),surface(y+sign*.025,z+.042),.002)
                for j in range(4):
                    zz=z-.034+j*.02
                    m.bone('navy',surface(y+sign*j*.006,zz),surface(y+sign*(j*.006+.025),zz+.02),.003)
    if name=='vladislav':
        m.bone('button',(.126,0,1.27),(.13,.04,1.267),.003)
        m.bone('lip',(.126,-.04,1.267),(.126,-.006,1.267),.003)
    points=np.asarray(m.vertices)
    points[:,2]-=points[:,2].min()
    points*=1.8/points[:,2].max()
    m.vertices=points.tolist()
    return m


def write_metadata(directory,name,profile):
    model_id=directory.name
    (directory/'model.config').write_text(f'''<?xml version="1.0"?>
<model><name>{model_id}</name><version>1.0</version><sdf version="1.6">model.sdf</sdf>
<description>Photo-inspired procedural person: {name}. Approximate 1.80 m height.</description></model>
''')
    (directory/'model.sdf').write_text(f'''<?xml version="1.0"?>
<sdf version="1.6"><model name="{model_id}"><static>true</static><link name="body">
<visual name="person_visual"><geometry><mesh><uri>model://{model_id}/meshes/person.dae</uri></mesh></geometry></visual>
<collision name="person_collision"><geometry><mesh><uri>model://{model_id}/meshes/person.dae</uri></mesh></geometry></collision>
</link></model></sdf>
''')
    (directory/'README.md').write_text(f'''# {model_id}

Reference: `../{profile['photo']}`. {profile['description']}

A manually described, procedural approximation for simulation, not a measured
3D scan or photogrammetric reconstruction. Height is chosen as 1.80 m.
Facial likeness, body proportions and hidden surfaces are approximate.
Geometry consists of overlapping closed components.

- `meshes/person.dae`: colored COLLADA, metres, Z up, +X forward.
- `meshes/person.glb`: colored glTF binary, metres, Y up; opens in Rerun.
- `model.sdf`: matching visual and collision meshes for Gazebo.
- `preview.png`: rendered front and oblique views.

Regenerate from the repository root: `python3 tools/generate_itl_people.py --person {name}`.
The generator requires NumPy; previews additionally require Matplotlib.
''')


def preview(mesh,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.faces);tri=v[f]
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
    light=np.array([.8,-.4,1]);light/=np.linalg.norm(light)
    shade=.65+.35*np.maximum(0,normals@light)
    colors=np.asarray([mesh.palette[k] for k in mesh.face_materials])*shade[:,None]
    fig=plt.figure(figsize=(9,8),facecolor='#edf0f1')
    for i,angle in enumerate((0,45),1):
        ax=fig.add_subplot(1,2,i,projection='3d',computed_zorder=False)
        ax.set_facecolor('#edf0f1');ax.add_collection3d(Poly3DCollection(tri,facecolors=colors,edgecolors='none'))
        ax.set(xlim=(-.43,.43),ylim=(-.43,.43),zlim=(-.01,1.84));ax.set_box_aspect((.86,.86,1.85))
        ax.view_init(elev=8,azim=angle);ax.set_axis_off()
    fig.suptitle(title+'\nPhoto-inspired simulation mesh',fontsize=15)
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--person',choices=list(PROFILES),action='append')
    parser.add_argument('--no-preview',action='store_true')
    args=parser.parse_args()
    for name in args.person or PROFILES:
        mesh=build_person(name);directory=ROOT/'models'/('ITL-RailGuard_'+name)
        save_mesh(mesh,directory);write_metadata(directory,name,PROFILES[name])
        if not args.no_preview:preview(mesh,directory/'preview.png',directory.name)
        print(directory,flush=True)

if __name__=='__main__':main()
