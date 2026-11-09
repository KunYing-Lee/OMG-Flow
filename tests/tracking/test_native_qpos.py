import numpy as np
import pytest

from omg.tracking.holomotion.reference import native_qpos


def test_native_motion_validates_at_tracking_boundary_and_copies():
    qpos = np.zeros((12, 36), dtype=np.float64)
    qpos[:, 3] = 1
    checked = native_qpos(qpos, 50)
    assert checked.dtype == np.float32
    np.testing.assert_array_equal(checked, qpos)
    checked[0, 0] = 3
    assert qpos[0, 0] == 0


@pytest.mark.parametrize("case", ["empty", "shape", "nonfinite", "quaternion", "fps"])
def test_invalid_native_motion_is_rejected(case):
    qpos = np.zeros((12, 36), dtype=np.float32)
    qpos[:, 3] = 1
    if case == "empty":
        qpos = qpos[:0]
    elif case == "shape":
        qpos = qpos[:, :35]
    elif case == "nonfinite":
        qpos[0, 0] = np.nan
    elif case == "quaternion":
        qpos[0, 3] = 0
    with pytest.raises(ValueError):
        native_qpos(qpos, 25 if case == "fps" else 50)
