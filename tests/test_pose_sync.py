from railpath.pose_sync import PoseSynchronizer


def test_exact_timestamps_either_arrival_order_and_ordered_output():
    emitted, resets = [], []
    sync = PoseSynchronizer(lambda *args: emitted.append(args), lambda: resets.append(1))
    sync.add_pose(1, 'lidar', 'pose1')
    sync.add_cloud(1, 'lidar', 'scan1')
    sync.add_cloud(2, 'lidar', 'scan2')
    sync.add_cloud(3, 'lidar', 'scan3')
    sync.add_pose(3, 'lidar', 'pose3')
    assert emitted == [(1, 'scan1', 'pose1')]
    sync.add_pose(2, 'wrong_sensor', 'wrong')
    assert len(emitted) == 1
    sync.add_pose(2, 'lidar', 'pose2')
    assert emitted == [(1, 'scan1', 'pose1'), (2, 'scan2', 'pose2'), (3, 'scan3', 'pose3')]
    assert len(resets) == 1


def test_missing_pose_timeout_queue_bound_and_rewind():
    emitted, resets, now = [], [], [0.]
    sync = PoseSynchronizer(lambda *args: emitted.append(args), lambda: resets.append(1),
                            max_pending=2, timeout_s=.5, clock=lambda: now[0])
    for i in range(3):
        sync.add_cloud(i, 'lidar', i)
    assert emitted == [(0, 0, None)]
    assert len(sync.pending) == 2
    sync.add_pose(2, 'lidar', 'pose2')
    now[0] = .6
    sync.flush()
    assert emitted == [(0, 0, None), (1, 1, None), (2, 2, 'pose2')]
    for i in range(20):
        sync.add_pose(i+10, 'lidar', i)
    assert len(sync.poses) == 2
    sync.add_cloud(1, 'lidar', 'rewind')
    assert len(resets) == 2 and not sync.poses
    sync.add_pose(1, 'lidar', 'new_pose')
    assert emitted[-1] == (1, 'rewind', 'new_pose')
    sync.add_cloud(2, 'other', 'new_sensor')
    assert len(resets) == 3
