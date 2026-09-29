import json
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from rosbags.highlevel import AnyReader
from obstacle_app.core import Scenario
from obstacle_app.sampling_backend import RayCastingBackend
from obstacle_app.world_map import file_digest, source_fingerprint
from tools.generate_test_dataset import (ROOT, BAGS, MODELS, encounter_times, sample_obstacles,
    validate_config, inspect_bag, prepare_reference, place_obstacles, verify_export, check_space)
from tests.test_obstacle_app import make_bag, TYPESTORE
from tests.test_world_map import build_test_map


class DatasetGenerationTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT/'config/test_dataset.json').read_text())

    def test_schedule_and_model_balance_are_reproducible(self):
        config=self.config();validate_config(config)
        self.assertEqual(encounter_times(52,config['schedule']),[5.,20.,35.,50.])
        self.assertEqual(encounter_times(51.9,config['schedule']),[5.,20.,35.])
        self.assertEqual(encounter_times(6,config['schedule']),[])
        first=sample_obstacles('new_data',1200,config)
        self.assertEqual(first,sample_obstacles('new_data',1200,config))
        self.assertNotEqual(first,sample_obstacles('roundT_doubleT',1200,config))
        self.assertEqual(len(BAGS),6)
        self.assertNotIn('doubleT_obstacle',BAGS);self.assertNotIn('cloud_with_fake_obj',BAGS)
        counts=[sum(x['model_id']==model for x in first) for model in MODELS]
        self.assertLessEqual(max(counts)-min(counts),1)
        for x in first:
            self.assertIn(x['model_id'],MODELS)
            self.assertLessEqual(abs(x['lateral_offset_m']),1)
            self.assertEqual(x['height_above_track_m'],0)
            lo,hi=config['models'][x['model_id']]['scale']
            self.assertTrue(lo<=x['scale']<=hi)
        self.assertTrue(any(x['lateral_offset_m']<0 for x in first))
        self.assertTrue(any(x['lateral_offset_m']>0 for x in first))

    def test_invalid_config_and_space_rejected(self):
        for modify in (lambda c:c['schedule'].update(interval_s=0),
                       lambda c:c['models'].update(wire_hanging={'scale':[1,1]}),
                       lambda c:c['models']['box_square'].update(scale=[0,1]),
                       lambda c:c['placement'].update(lateral_offset_m=[-2,2])):
            config=self.config();modify(config)
            with self.assertRaises(ValueError):validate_config(config)
        with patch('tools.generate_test_dataset.shutil.disk_usage',return_value=shutil._ntuple_diskusage(10,9,1)):
            with self.assertRaises(OSError):check_space(ROOT,[{'estimated_export_bytes':100}])

    def fixture(self,root):
        bag=root/'source';make_bag(bag,rails=True,obstacle=True,obstacle_y=-20)
        record,stamps=inspect_bag(bag)
        cache=root/'refs'/bag.name;cache.mkdir(parents=True)
        path=cache/'reference.npz';build_test_map(bag,record['topic'],0,path,lambda *a:None)
        with np.load(path) as d:data={k:d[k] for k in d.files}
        data['initial_speed_m_s']=np.asarray(0.)
        np.savez(path,**data)
        identity={k:record[k] for k in ('name','topic','fingerprint','count','initial_speed_m_s','timestamps_sha256')}
        (cache/'reference.json').write_text(json.dumps(dict(identity=identity,sha256=file_digest(path))))
        output=root/'result';output.mkdir()
        directory,data=prepare_reference(record,stamps,output,root/'refs',lambda *a:None)
        return bag,record,stamps,output,directory,data

    def test_placement_uses_reference_and_reports_missing_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            bag,record,stamps,output,directory,data=self.fixture(Path(tmp))
            items=[dict(id='box',model_id='box_square',encounter_seconds=.2,
                        lateral_offset_m=.7,height_above_track_m=.2,relative_yaw_deg=30.,scale=2.)]
            objects,details=place_obstacles(items,record,data,ROOT/'models')
            self.assertEqual(details[0]['status'],'placed')
            np.testing.assert_allclose(objects[0].position,[-.7,-6,-.9],atol=1e-6)
            self.assertAlmostEqual(objects[0].yaw_deg,-60)
            np.testing.assert_allclose(details[0]['dimensions_local_m'],[1,1,1],atol=1e-6)
            shifted={**data,'center':data['center']+np.array([100,0,0])}
            objects,details=place_obstacles(items,record,shifted,ROOT/'models')
            self.assertFalse(objects);self.assertEqual(details[0]['status'],'failed')
            self.assertIn('no reliable rails',details[0]['reason'])

    def test_compressed_export_matches_app_and_retains_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            bag,record,stamps,output,directory,data=self.fixture(Path(tmp))
            config=self.config();config['schedule']=dict(first_encounter_s=.2,interval_s=.3,end_margin_s=.1)
            items=sample_obstacles('fixture',record['duration_s'],config)
            objects,details=place_obstacles(items,record,data,ROOT/'models')
            self.assertTrue(objects);self.assertTrue(all(p['status']=='placed' for p in details))
            scenario=Scenario(bag,record['topic'],0,record['duration_s'],world_obstacles=objects,world_map_directory=directory)
            with patch('obstacle_app.sampling_backend.KissICP',side_effect=AssertionError('Must reuse GT poses')):
                RayCastingBackend().run(scenario,output/'plain',lambda *a:None)
                RayCastingBackend().run(replace(scenario,compression='zstd'),output/'compressed',lambda *a:None)
            meta=verify_export(output/'compressed',record,stamps)
            self.assertGreater(meta['total_added_points'],0)
            self.assertEqual(meta['compression'],'zstd')
            self.assertEqual(source_fingerprint(bag),record['fingerprint'])
            with AnyReader([output/'plain'],default_typestore=TYPESTORE) as a,AnyReader([output/'compressed'],default_typestore=TYPESTORE) as b:
                ar=[(c.topic,t,bytes(raw)) for c,t,raw in a.messages()]
                br=[(c.topic,t,bytes(raw)) for c,t,raw in b.messages()]
                self.assertEqual(ar,br)
                for c,_,raw in b.messages():b.deserialize(raw,c.msgtype)
            self.assertIn('compression_mode: message',(output/'compressed'/'metadata.yaml').read_text())

    def test_stale_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bag,record,stamps,output,directory,data=self.fixture(root)
            with (directory/'poses.npz').open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):prepare_reference(record,stamps,output,root/'refs',lambda *a:None)


if __name__=='__main__':unittest.main()
