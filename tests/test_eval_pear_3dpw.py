import numpy as np

from tools.eval_pear_3dpw import gt_camera_joints, joint_errors, person_crop, valid_frames


def test_metric_translation_is_removed():
    gt = np.random.default_rng(1).normal(size=(24, 3))
    raw, pa = joint_errors(gt + [0.2, -0.3, 1.0], gt)
    assert raw.max() < 1e-9
    assert pa.max() < 1e-9


def test_metric_pa_removes_rotation_and_scale():
    gt = np.random.default_rng(2).normal(size=(24, 3))
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    raw, pa = joint_errors(gt @ rotation.T * 2 + 3, gt)
    assert raw.mean() > 100
    assert pa.max() < 1e-9


def test_metric_known_joint_displacement():
    gt = np.zeros((24, 3))
    pred = gt.copy()
    pred[4, 0] = 0.01
    raw, _ = joint_errors(pred, gt)
    assert abs(raw[4] - 10.0) < 1e-9
    assert abs(raw.mean() - 10.0 / 24) < 1e-9


def test_camera_transform_and_crop():
    joints = np.zeros((24, 3))
    joints[4] = [1, 2, 3]
    camera = np.eye(4)
    camera[:3, 3] = [4, 5, 6]
    poses2d = np.zeros((1, 3, 18))
    poses2d[0, 0, :6] = [20, 25, 30, 35, 40, 45]
    poses2d[0, 1, :6] = [15, 25, 35, 45, 55, 65]
    poses2d[0, 2, :6] = 1
    sequence = {"jointPositions": [joints.reshape(1, 72)],
                "cam_poses": [camera], "campose_valid": [[1]], "poses2d": [poses2d]}
    np.testing.assert_array_equal(gt_camera_joints(sequence, 0, 0)[4], [5, 7, 9])
    np.testing.assert_array_equal(valid_frames(sequence, 0), [0])
    canvas, box = person_crop(np.zeros((100, 100, 3), np.uint8), poses2d[0], 1.25)
    assert canvas.shape == (256, 256, 3)
    assert len(box) == 4
