import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from rosbags.highlevel import AnyReader

from obstacle_app.core import Scenario, WorldObstacleSpec
from obstacle_app.sampling_backend import RayCastingBackend
from obstacle_app.web import create_app
from obstacle_app.world_map import file_digest, source_fingerprint, world_mesh, reduce_cloud
from tests.test_obstacle_app import make_bag, MODEL, TYPESTORE


def build_test_map(bag, topic, speed, output, progress):
    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        times = [stamp for conn, stamp, raw in reader.messages() if conn.topic == topic]
    center = np.array([[0, 0, -1.1], [0, -50, -1.1]], dtype=np.float32)
    poses = np.repeat(np.eye(4)[None], len(times), axis=0)
    poses[:, 1, 3] = -np.arange(len(times))*3
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, center=center, lengths=[0, 50], poses=poses, timestamps_ns=times)


class WorldMapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data, self.output = self.root/"data", self.root/"output"
        self.data.mkdir(); self.output.mkdir()
        self.bag = self.data/"tiny"
        self.raws = make_bag(self.bag, rails=True, obstacle=True, obstacle_y=-20)
        self.client = TestClient(create_app(data_root=self.data, output_root=self.output,
            model_path=MODEL, detection_map_builder=build_test_map))

    def tearDown(self):
        self.client.close(); self.temp.cleanup()

    def prepare(self):
        response = self.client.post('/api/maps', json=dict(bag_id='source/tiny', topic='/lidar_points'))
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        for _ in range(500):
            job = self.client.get('/api/maps/jobs/'+job['id']).json()
            if job['state'] in ('done', 'failed'):break
            time.sleep(.01)
        self.assertEqual(job['state'], 'done', job)
        return job['map_id']

    def test_map_cache_binding_and_preview(self):
        map_id = self.prepare()
        self.assertEqual(self.prepare(), map_id)
        params=dict(bag_id='source/tiny', topic='/lidar_points')
        response=self.client.get(f'/api/maps/{map_id}/points',params=params)
        self.assertEqual(response.status_code,200)
        points=np.frombuffer(response.content,dtype='<f4').reshape(-1,3)
        self.assertLessEqual(len(points),500000)
        self.assertTrue(np.isfinite(points).all())
        self.assertEqual(self.client.get(f'/api/maps/{map_id}/geometry',params={**params,'topic':'/wrong'}).status_code,400)
        item=dict(model_id='wire_hanging',position=[3,-12,4.2],yaw_deg=37,scale=1.5)
        preview=self.client.post('/api/world-preview',json=item)
        self.assertEqual(preview.status_code,200,preview.text)
        path=MODEL.parents[2]/'wire_hanging/meshes/wire.dae'
        expected,faces=world_mesh(WorldObstacleSpec(model_path=path,**item))
        np.testing.assert_array_equal(preview.json()['vertices'],expected)
        np.testing.assert_array_equal(preview.json()['triangles'],faces)
        self.assertAlmostEqual(float(expected[:,2].max()),4.2,places=5)
        self.assertAlmostEqual(float(expected[:,2].min()),1.2,places=5)
        from obstacle_app.web import MODEL_FILES
        for model_id, (_, relative) in MODEL_FILES.items():
            preview_params={**item,'model_id':model_id}
            result=self.client.post('/api/world-preview',json=preview_params)
            self.assertEqual(result.status_code,200,result.text)
            mesh,_=world_mesh(WorldObstacleSpec(model_path=MODEL.parents[2]/relative,**preview_params))
            np.testing.assert_array_equal(result.json()['vertices'],mesh)
            colors=np.asarray(result.json()['colors'])
            self.assertEqual(colors.shape,(len(result.json()['triangles']),3))
            self.assertTrue(np.isfinite(colors).all())
            self.assertEqual(result.json()['cardboard'],model_id.startswith('box_'))
            self.assertEqual(len(result.json()['local_vertices']),len(mesh))
            if model_id!='wire_hanging':
                self.assertAlmostEqual(float(mesh[:,2].min()),item['position'][2],places=5)
        # Modifying cached poses must invalidate the world coordinate reference.
        with (self.output/'.world_maps'/map_id/'poses.npz').open('ab') as stream:stream.write(b'changed')
        self.assertEqual(self.client.get(f'/api/maps/{map_id}/geometry',params=params).status_code,400)

    def test_reject_mixed_stale_and_wrong_bag_requests(self):
        map_id=self.prepare()
        request=dict(bag_id='source/tiny',topic='/lidar_points',map_id=map_id,
            world_obstacle=dict(model_id='person_standing',position=[4,-10,-1]))
        mixed={**request,'obstacles':[dict(id='timed',model_id='person_standing',encounter_seconds=2)]}
        self.assertEqual(self.client.post('/api/jobs',json=mixed).status_code,400)
        self.assertEqual(self.client.post('/api/jobs',json={**request,'map_id':'f'*32}).status_code,400)
        make_bag(self.data/'other')
        self.assertEqual(self.client.post('/api/jobs',json={**request,'bag_id':'source/other'}).status_code,400)
        metadata=self.bag/'metadata.yaml';metadata.write_text(metadata.read_text()+'\n')
        self.assertEqual(self.client.post('/api/jobs',json=request).status_code,400)

    def test_export_reuses_poses_and_preserves_source(self):
        map_id=self.prepare()
        directory=self.output/'.world_maps'/map_id
        item=WorldObstacleSpec('person_standing',MODEL,(0,-10,-1.1))
        with AnyReader([self.bag],default_typestore=TYPESTORE) as reader:duration=reader.duration/1e9
        scenario=Scenario(self.bag,'/lidar_points',0,duration,world_obstacle=item,world_map_directory=directory)
        output=self.output/'export'
        with patch('obstacle_app.sampling_backend.KissICP',side_effect=AssertionError('Must reuse poses')):
            RayCastingBackend().run(scenario,output,lambda *args:None)
        with np.load(directory/'poses.npz') as before,np.load(output/'merged_rails.npz') as after:
            np.testing.assert_array_equal(before['poses'],after['poses'])
            np.testing.assert_array_equal(before['timestamps_ns'],after['timestamps_ns'])
        with AnyReader([self.bag],default_typestore=TYPESTORE) as reader:
            self.assertEqual([bytes(raw) for _,_,raw in reader.messages()],self.raws)
        with AnyReader([output],default_typestore=TYPESTORE) as reader:
            frames=list(reader.messages());self.assertEqual(len(frames),9)
            self.assertEqual([t for _,t,_ in frames],[i*100000000 for i in range(1,10)])
            for conn,_,raw in frames:reader.deserialize(raw,conn.msgtype)
        metadata=json.loads((output/'obstacle_scenario.json').read_text())
        self.assertEqual(metadata['rail_map_method'],'cached_editor_map')
        self.assertGreater(metadata['total_added_points'],0)
        self.assertEqual(metadata['frame_counts']['900000000']['added'],0)

    def test_multiple_world_objects_export_together(self):
        map_id=self.prepare()
        objects=[dict(model_id='person_standing',position=[0,-10,-1.1]),
                 dict(model_id='box_square',position=[.7,-17,-1.1],yaw_deg=35,scale=1.5),
                 dict(model_id='wire_hanging',position=[-.7,-21,2.5])]
        request=dict(bag_id='source/tiny',topic='/lidar_points',map_id=map_id,world_obstacles=objects)
        self.assertEqual(self.client.post('/api/jobs',json={**request,'world_obstacle':objects[0]}).status_code,422)
        self.assertEqual(self.client.post('/api/jobs',json={**request,'obstacles':[dict(id='timed',model_id='person_standing',encounter_seconds=2)]}).status_code,400)
        response=self.client.post('/api/jobs',json=request)
        self.assertEqual(response.status_code,202,response.text)
        for _ in range(500):
            job=self.client.get('/api/jobs/'+response.json()['id']).json()
            if job['state'] in ('done','failed'):break
            time.sleep(.02)
        self.assertEqual(job['state'],'done',job)
        output=self.output/job['output_bag_id'].split('/',1)[1]
        metadata=json.loads((output/'obstacle_scenario.json').read_text())
        self.assertEqual(len(metadata['obstacles']),3)
        self.assertEqual(len({item['id'] for item in metadata['obstacles']}),3)
        self.assertEqual([item['model'] for item in metadata['obstacles']],[item['model_id'] for item in objects])
        self.assertGreater(metadata['total_added_points'],0)
        self.assertEqual(metadata['rail_map_method'],'cached_editor_map')

    def test_origin_contact_preserves_source(self):
        map_id=self.prepare()
        item=WorldObstacleSpec('box_square',MODEL.parents[2]/'box_square/meshes/box.dae',(0,0,-.25))
        with AnyReader([self.bag],default_typestore=TYPESTORE) as reader:duration=reader.duration/1e9
        output=self.output/'contact'
        RayCastingBackend().run(Scenario(self.bag,'/lidar_points',0,duration,world_obstacle=item,
            world_map_directory=self.output/'.world_maps'/map_id),output,lambda *args:None)
        counts=json.loads((output/'obstacle_scenario.json').read_text())['frame_counts']['100000000']
        self.assertEqual(counts['near_contact_obstacles'],1)
        self.assertEqual(counts['removed'],0)

    def test_world_object_with_rotated_translated_scan(self):
        from obstacle_app.ray_rendering import MeshRayRenderer
        from tests.test_ray_rendering import cloud, sensor
        item=WorldObstacleSpec('box_square',MODEL.parents[2]/'box_square/meshes/box.dae',(0,-10,-.25))
        vertices,faces=world_mesh(item)
        renderer=MeshRayRenderer(vertices,faces,sensor())
        pose=np.eye(4)
        angle=np.deg2rad(20)
        pose[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
        pose[:3,3]=[-2,-3,0]
        result=renderer.render(cloud([[0,0,0]]),pose)
        self.assertGreater(len(result.points),0)
        world=result.points @ pose[:3,:3].T+pose[:3,3]
        self.assertTrue(np.all(world>=vertices.min(axis=0)-1e-4))
        self.assertTrue(np.all(world<=vertices.max(axis=0)+1e-4))
        pose[:3,3]=[0,-13,0]
        passed=renderer.render(cloud([[0,-20,0]]),pose)
        self.assertEqual(len(passed.points),0)
        self.assertTrue(passed.keep.all())

    def test_reduction_is_capped_and_box_transform(self):
        points=np.random.default_rng(1).uniform(-50,50,(2000,3)).astype(np.float32)
        reduced=reduce_cloud(points,100)
        self.assertEqual(len(reduced),100)
        np.testing.assert_array_equal(reduced,reduce_cloud(points,100))
        item=WorldObstacleSpec('box_long_thin',MODEL.parents[2]/'box_long_thin/meshes/box.dae',(5,6,7),90,2)
        vertices,_=world_mesh(item)
        np.testing.assert_allclose(vertices.min(axis=0),[4.8,4,7],atol=1e-6)
        np.testing.assert_allclose(vertices.max(axis=0),[5.2,8,7.4],atol=1e-6)


if __name__=='__main__':unittest.main()
