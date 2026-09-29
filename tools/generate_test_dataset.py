"""Export six reproducible obstacle bags using the web app's ray-casting backend."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from obstacle_app.bag_storage import estimate_export_bytes
from obstacle_app.core import ObstacleSpec, Scenario, WorldObstacleSpec
from obstacle_app.detection import INITIAL_SPEED_MPS, build_merged_rail_map
from obstacle_app.merged_rails import MergedRailMap
from obstacle_app.mesh import load_collada_mesh
from obstacle_app.sampling_backend import RayCastingBackend
from obstacle_app.timed_placement import timed_anchor
from obstacle_app.world_map import file_digest, source_fingerprint, validate_map

BAGS = ('doubleT_platform', 'new_data', 'roundT_doubleT',
        'roundT_pressureGate_roundT', 'roundT_squareT_pressureGate_squareT',
        'squareT_platform_squareT_switch')
MODELS = {'person_standing': 'person_standing/meshes/person.dae',
          'box_square': 'box_square/meshes/box.dae',
          'box_long_thin': 'box_long_thin/meshes/box.dae'}
STORE = get_typestore(Stores.ROS2_HUMBLE)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validate_config(config):
    seed = config['seed']
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    schedule = config['schedule']
    for key in ('first_encounter_s', 'interval_s', 'end_margin_s'):
        value = schedule[key]
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'Invalid schedule: {key}')
    if schedule['interval_s'] <= 0:
        raise ValueError('Encounter interval must be positive')
    if not config['models'] or not set(config['models']) <= MODELS.keys():
        raise ValueError('Only standing people, square boxes and long boxes are supported')
    ranges = list(config['placement'].values()) + [m['scale'] for m in config['models'].values()]
    for bounds in ranges:
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError('Each sampling range must contain finite [minimum, maximum]')
    if any(m['scale'][0] <= 0 for m in config['models'].values()):
        raise ValueError('Model scales must be positive')
    if config['placement']['height_above_track_m'][0] < 0:
        raise ValueError('Height must be nonnegative')
    if not -1 <= config['placement']['lateral_offset_m'][0] <= config['placement']['lateral_offset_m'][1] <= 1:
        raise ValueError('Lateral offsets must stay within -1 to +1 m')


def encounter_times(duration_s, schedule):
    last = duration_s - schedule['end_margin_s']
    first, step = schedule['first_encounter_s'], schedule['interval_s']
    if last < first:
        return []
    return [float(first + i * step) for i in range(int(np.floor((last-first)/step)) + 1)]


def sample_obstacles(bag_name, duration_s, config):
    # Stable per-bag RNG: subsets and execution order do not change scenes.
    key = hashlib.sha256(f"{config['seed']}:{bag_name}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], 'little'))
    times = encounter_times(duration_s, config['schedule'])
    names = sorted(config['models'])
    assignments = [names[i % len(names)] for i in range(len(times))]
    rng.shuffle(assignments)
    result = []
    for i, (seconds, model) in enumerate(zip(times, assignments)):
        placement = config['placement']
        result.append(dict(id=f'obstacle-{i+1:03d}', model_id=model,
            encounter_seconds=seconds,
            lateral_offset_m=float(rng.uniform(*placement['lateral_offset_m'])),
            height_above_track_m=float(rng.uniform(*placement['height_above_track_m'])),
            relative_yaw_deg=float(rng.uniform(*placement['yaw_relative_to_track_deg'])),
            scale=float(rng.uniform(*config['models'][model]['scale']))))
    return result


def timestamps(bag, topic):
    # Avoid scanning all serialized clouds just to read timestamp metadata.
    if list(bag.glob('*.db3')):
        stamps = []
        for db in sorted(bag.glob('*.db3')):
            with sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True) as conn:
                stamps.extend(row[0] for row in conn.execute(
                    'SELECT m.timestamp FROM messages m JOIN topics t ON m.topic_id=t.id '
                    'WHERE t.name=? ORDER BY m.timestamp', (topic,)))
        return np.asarray(sorted(stamps), np.int64)
    with AnyReader([bag], default_typestore=STORE) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        return np.asarray([stamp for _, stamp, _ in reader.messages(connections=connections)], np.int64)


def inspect_bag(bag, compression="zstd"):
    with AnyReader([bag], default_typestore=STORE) as reader:
        clouds = [c for c in reader.connections if c.msgtype == 'sensor_msgs/msg/PointCloud2']
        if len(clouds) != 1:
            raise ValueError(f'{bag}: expected exactly one PointCloud2 topic')
        topic = clouds[0].topic
        record = dict(name=bag.name, source=str(bag.resolve()), topic=topic,
            fingerprint=source_fingerprint(bag), count=clouds[0].msgcount,
            start_ns=int(reader.start_time), duration_s=reader.duration/1e9,
            initial_speed_m_s=float(INITIAL_SPEED_MPS.get(bag.name, 0.)),
            bytes=sum(p.stat().st_size for p in bag.iterdir() if p.suffix in ('.db3', '.mcap')))
        record['estimated_export_bytes'] = estimate_export_bytes(reader,bag,reader.start_time,reader.end_time,compression)
    stamps = timestamps(bag, topic)
    if len(stamps) != record['count'] or not len(stamps) or np.any(np.diff(stamps) <= 0):
        raise ValueError(f'{bag}: invalid timestamp sequence')
    record['timestamps_sha256'] = hashlib.sha256(stamps.tobytes()).hexdigest()
    return record, stamps


def validate_poses(path, record, stamps):
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(data['timestamps_ns'], stamps)
    poses, center = data['poses'], data['center']
    if poses.shape != (len(stamps), 4, 4) or not np.isfinite(poses).all():
        raise ValueError('Invalid reference poses')
    if center.ndim != 2 or center.shape[1] != 3 or len(center) < 2 or not np.isfinite(center).all():
        raise ValueError('Invalid reference centerline')
    if float(data['initial_speed_m_s']) != record['initial_speed_m_s']:
        raise ValueError('Reference odometry speed mismatch')
    np.testing.assert_allclose(poses[:, 3, :], np.tile([0,0,0,1], (len(poses),1)), atol=1e-6)
    rotation = poses[:, :3, :3]
    np.testing.assert_allclose(rotation @ rotation.transpose(0,2,1), np.tile(np.eye(3),(len(poses),1,1)), atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-5)
    return data


def prepare_reference(record, stamps, output, reference_root, progress):
    identity = {k:record[k] for k in ('name','topic','fingerprint','timestamps_sha256','initial_speed_m_s')}
    map_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]
    directory = output / '.maps' / map_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'poses.npz'
    bag = Path(record['source'])
    if (directory/'manifest.json').exists():
        manifest = validate_map(directory, bag, record['topic'])
        if manifest['initial_speed_m_s'] != record['initial_speed_m_s']:
            raise ValueError('Cached map speed mismatch')
    elif reference_root is not None:
        # Explicit import of existing evaluation maps, with identity and byte validation.
        source = reference_root / record['name']
        manifest = json.loads((source/'reference.json').read_text())
        expected = {k:record[k] for k in ('name','topic','fingerprint','count','initial_speed_m_s','timestamps_sha256')}
        if manifest['identity'] != expected or file_digest(source/'reference.npz') != manifest['sha256']:
            raise ValueError(f"{record['name']}: supplied reference does not match this bag")
        shutil.copyfile(source/'reference.npz', path)
    else:
        build_merged_rail_map(bag, record['topic'], record['initial_speed_m_s'], path, progress)
    data = validate_poses(path, record, stamps)
    write_json(directory/'manifest.json', dict(map_id=map_id, bag_id='source/'+record['name'],
        topic=record['topic'], source_fingerprint=record['fingerprint'],
        poses_sha256=file_digest(path), initial_speed_m_s=record['initial_speed_m_s']))
    return directory, data


def place_obstacles(items, record, data, models_root):
    merged = MergedRailMap(data['center'], data['lengths'])
    placed, details = [], []
    for item in items:
        path = models_root / MODELS[item['model_id']]
        try:
            spec = ObstacleSpec(item['id'], path, item['model_id'], item['encounter_seconds'],
                                item['lateral_offset_m'], item['height_above_track_m'])
            base, tangent, stamp = timed_anchor(spec, record['start_ns'], data['timestamps_ns'],
                data['poses'], [None]*len(data['poses']), merged)
            yaw = float(np.degrees(np.arctan2(tangent[1], tangent[0])) + item['relative_yaw_deg'])
            obstacle = WorldObstacleSpec(item['model_id'], path, tuple(base), yaw, item['scale'])
            vertices, _ = load_collada_mesh(path)
            placed.append(obstacle)
            details.append(dict(item, status='placed', anchor_xyz_m=base.tolist(), yaw_world_deg=yaw,
                encounter_timestamp_ns=int(stamp), dimensions_local_m=(np.ptp(vertices,axis=0)*item['scale']).tolist()))
        except ValueError as exc:
            details.append(dict(item, status='failed', reason=str(exc)))
    return tuple(placed), details


def disk_required(records):
    return sum(r['estimated_export_bytes'] for r in records) + 2*1024**3


def check_space(output, records):
    needed, free = disk_required(records), shutil.disk_usage(output).free
    if free < needed:
        raise OSError(f'Exports need approximately {needed/1024**3:.1f} GiB free; '
                      f'only {free/1024**3:.1f} GiB available at {output}. Choose a larger --output directory.')


def progress_logger(name):
    last = [0.]
    def progress(fraction, message):
        if time.monotonic()-last[0] >= 10 or fraction >= 1:
            print(f'{name}: {fraction:.1%} {message}', flush=True)
            last[0] = time.monotonic()
    return progress


def verify_export(path, record, stamps):
    metadata = json.loads((path/'obstacle_scenario.json').read_text())
    if metadata['source_bag'] != record['name'] or metadata['source_topic'] != record['topic']:
        raise ValueError('Export source mismatch')
    np.testing.assert_array_equal(timestamps(path, record['topic']), stamps)
    with AnyReader([path], default_typestore=STORE) as reader:
        if not reader.message_count:
            raise ValueError('Empty exported bag')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=ROOT/'models')
    parser.add_argument('--config', type=Path, default=ROOT/'config/test_dataset.json')
    parser.add_argument('--compression', choices=['zstd', 'none'], default='zstd')
    parser.add_argument('--reference-root', type=Path, help='Optional existing <bag>/reference.npz + reference.json maps')
    parser.add_argument('--bags', nargs='+', choices=BAGS, default=list(BAGS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan-only', action='store_true', help='Save schedule and storage estimate; no map building or exports')
    mode.add_argument('--prepare-only', action='store_true', help='Prepare/validate reference maps and world placements; no exports')
    args = parser.parse_args()
    config = json.loads(args.config.read_text()); validate_config(config)
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    records, stamps_by_bag = [], {}
    for name in args.bags:
        record, stamps = inspect_bag(args.data/name, None if args.compression == "none" else args.compression)
        if args.output == Path(record['source']) or args.output.is_relative_to(Path(record['source'])):
            raise ValueError('Output must not be inside a source bag')
        records.append(record); stamps_by_bag[name] = stamps
    plans = {r['name']:sample_obstacles(r['name'],r['duration_s'],config) for r in records}
    if any(not items for items in plans.values()):
        raise ValueError('A recording is too short for the configured encounter schedule')
    binding = dict(version=1, config=config, compression=args.compression, sources=records, plans=plans,
        models={m:file_digest(args.models/p) for m,p in MODELS.items()},
        code={str(p.relative_to(ROOT)):file_digest(p) for p in [Path(__file__), *sorted((ROOT/'obstacle_app').glob('*.py'))]},
        gt='Merged-map centerline estimate; not surveyed ground truth. No silent relocation of failed placements.')
    manifest_path = args.output/'dataset.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != binding:
        raise ValueError('Dataset configuration/source/code changed; use a new output directory')
    write_json(manifest_path,binding)
    print(f"Scheduled {sum(map(len,plans.values()))} obstacles in {len(records)} bags; "
          f"estimated required space {disk_required(records)/1024**3:.1f} GiB",flush=True)
    if args.plan_only:return
    if not args.prepare_only:
        pending = [r for r in records if not (args.output/(r['name']+'_obstacles')/'dataset_complete.json').exists()]
        check_space(args.output,pending)
    prepared = []
    for record in records:
        name = record['name']; stamps = stamps_by_bag[name]
        directory, data = prepare_reference(record,stamps,args.output,args.reference_root,progress_logger(name))
        obstacles, placements = place_obstacles(plans[name],record,data,args.models)
        write_json(args.output/(name+'_placements.json'),dict(map_id=directory.name,
            poses_sha256=file_digest(directory/'poses.npz'), obstacles=placements))
        prepared.append((record,directory,obstacles,placements))
    failures = [(r['name'],p['id'],p['reason']) for r,_,_,ps in prepared for p in ps if p['status']=='failed']
    if failures:
        write_json(args.output/'placement_failures.json',failures)
        raise ValueError(f'{len(failures)} placements lack reliable reference coverage; see placement_failures.json. No exports started.')
    if args.prepare_only:
        print('All placements prepared; no bags exported.',flush=True);return
    backend = RayCastingBackend()
    for record, directory, obstacles, placements in prepared:
        name = record['name']; target = args.output/(name+'_obstacles')
        completion = target/'dataset_complete.json'
        identity = dict(dataset_sha256=file_digest(manifest_path), source_fingerprint=record['fingerprint'],
                        poses_sha256=file_digest(directory/'poses.npz'))
        if completion.exists():
            if json.loads(completion.read_text()) != identity:raise ValueError('Completed bag identity mismatch')
            verify_export(target,record,stamps_by_bag[name]);print(f'{name}: verified completed export; skipping',flush=True);continue
        if target.exists():raise FileExistsError(f'Unrecognized output: {target}')
        partial_root = args.output/'.partial';partial_root.mkdir(exist_ok=True)
        partial = partial_root/(name+'_obstacles')
        if partial.exists():
            raise FileExistsError(f'Interrupted export at {partial}; remove only this partial directory before retrying')
        check_space(args.output,[record])
        scenario = Scenario(Path(record['source']),record['topic'],0,record['duration_s'],
            world_obstacles=obstacles,world_map_directory=directory,initial_speed_m_s=record['initial_speed_m_s'],
            compression=None if args.compression == 'none' else args.compression)
        backend.run(scenario,partial,progress_logger(name))
        verify_export(partial,record,stamps_by_bag[name])
        if source_fingerprint(Path(record['source'])) != record['fingerprint']:
            raise ValueError('Source changed during export')
        write_json(partial/'dataset_complete.json',identity)
        partial.rename(target)
        print(f'{name}: saved {target}',flush=True)
    write_json(args.output/'completion.json',dict(state='complete',bags=[r['name']+'_obstacles' for r in records],
        dataset_sha256=file_digest(manifest_path)))


if __name__ == '__main__':
    main()
