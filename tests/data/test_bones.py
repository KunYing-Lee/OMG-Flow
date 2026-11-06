import json

import numpy as np
import pandas as pd
import pytest
import torch

from omg.data.bones import BonesWindows, encode_bones_window
from omg.data.bones_prepare import CSV_COLUMNS, convert_g1_csv
from omg.motion.feature_codec import G1MotionFeatureCodec
from omg.robots.g1.kinematics import G1Kinematics


def test_native50_grid_units_joint_order_and_shortest_orientation():
    # 13 source frames span 100 ms: native50 has exactly 6 frames, not 7.
    n = 13
    src = pd.DataFrame(np.zeros((n, len(CSV_COLUMNS))), columns=CSV_COLUMNS)
    src["Frame"] = np.arange(n)
    src["root_translateX"] = 100 * np.arange(n) / 12
    src["root_translateZ"] = 75
    src["root_rotateZ"] = (170 + np.arange(n) * 2 + 180) % 360 - 180
    src["left_wrist_pitch_joint_dof"] = 90
    src["right_ankle_roll_joint_dof"] = -30
    poses = convert_g1_csv(src[CSV_COLUMNS[::-1]].to_csv(index=False).encode(), n)
    assert poses.shape == (6, 36)
    np.testing.assert_allclose(poses[:, 0], np.arange(6) / 5, atol=1e-7)
    np.testing.assert_allclose(poses[:, 2], 0.75)
    np.testing.assert_allclose(poses[:, 7 + 20], np.pi / 2)
    np.testing.assert_allclose(poses[:, 7 + 11], -np.pi / 6)
    half = np.deg2rad(170 + (np.arange(6) * 2.4) * 2) / 2
    expected = np.stack([np.cos(half), np.zeros(6), np.zeros(6), np.sin(half)], axis=-1)
    np.testing.assert_allclose(np.abs(np.sum(poses[:, 3:7] * expected, axis=1)), 1, atol=1e-6)
    src.loc[3, "Frame"] = 4
    with pytest.raises(ValueError, match="Frame index"):
        convert_g1_csv(src.to_csv(index=False).encode(), n)


def test_extrinsic_xyz_conversion():
    src = pd.DataFrame(np.zeros((3, len(CSV_COLUMNS))), columns=CSV_COLUMNS)
    src.Frame = [0, 1, 2]
    src[["root_rotateX", "root_rotateY", "root_rotateZ"]] = [90, 0, 90]
    q = convert_g1_csv(src.to_csv(index=False).encode(), 3)[0, 3:7]
    # Rz(90) Rx(90), wxyz; the opposite multiplication order flips y.
    np.testing.assert_allclose(q, [0.5, 0.5, 0.5, 0.5], atol=1e-6)


def test_anchor_matches_lite_codec_kinematics_and_roundtrips_world():
    torch.manual_seed(7)
    q = torch.randn(2, 12, 36)
    q[..., 3:7] = torch.nn.functional.normalize(q[..., 3:7], dim=-1)
    kin = G1Kinematics("assets/robots/g1/g1_kinematics.json")
    codec = G1MotionFeatureCodec(kin, num_prev_states=3)
    bodies = kin.forward_kinematics(q)
    mean, std = torch.randn(38), torch.rand(38) + 0.1
    expected = codec.canonicalize(q, bodies["body_pos_w"], bodies["body_quat_w"], q[:, 2:3, :3], q[:, 2:3, 3:7], 50)
    encoded = encode_bones_window(q, 3, mean, std)
    torch.testing.assert_close(encoded, (codec.assemble_features(expected) - mean) / std)
    reconstructed = codec.decode_to_world_qpos36(codec.split_features(encoded * std + mean), q[:, 2:3, :3], q[:, 2:3, 3:7])
    torch.testing.assert_close(reconstructed[..., :3], q[..., :3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(reconstructed[..., 7:], q[..., 7:], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close((q[..., 3:7] * reconstructed[..., 3:7]).sum(-1).abs(), torch.ones(2, 12))
    torch.testing.assert_close(kin.forward_body_positions(reconstructed), bodies["body_pos_w"], atol=1e-5, rtol=1e-5)


def test_windows_do_not_cross_clips_or_actor_splits(tmp_path):
    n = 102
    poses = np.broadcast_to(np.arange(n, dtype=np.float32)[:, None], (n, 36))
    poses.tofile(tmp_path / "qpos.f32")
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "bones-g1-native50-v1", "fps": 50, "frames": n}))
    (tmp_path / "captions.json").write_text(json.dumps(["one", "two", "three", "four"]))
    pd.DataFrame({
        "clip_id":["short","a","b","c","d"], "offset":[0,2,23,62,82],
        "frames":[2,21,39,20,20], "split":["train","train","train","val","test"],
        "caption_ids":[[0,1,2,3]] * 5,
    }).to_parquet(tmp_path / "clips.parquet")
    ds = BonesWindows(tmp_path, "train", history=3, horizon=7, stride=5)
    assert len(ds) == 3 + 6
    batch = ds.get_batch([0, 2, 3, 8])
    assert batch["clip_id"] == ["a", "a", "b", "b"]
    torch.testing.assert_close(batch["qpos"][:, 0, 0], torch.tensor([2., 12., 23., 48.]))
    torch.testing.assert_close(batch["qpos"][:, -1, 0], batch["qpos"][:, 0, 0] + 9)
    assert [ds[i]["text"] for i in [0,2,3,8]] == batch["text"]
    with pytest.raises(IndexError):
        ds[len(ds)]
    assert BonesWindows(tmp_path, "val", 3, 7, 5)[0]["clip_id"] == "c"


def test_publication_requires_complete_compressed_source(tmp_path, monkeypatch):
    import io
    import tarfile

    from omg.data import bones_prepare

    n = 13
    columns = {key: np.zeros(n) for key in CSV_COLUMNS}
    columns["Frame"] = np.arange(n)
    csv = pd.DataFrame(columns).to_csv(index=False).encode()
    path = "g1/csv/230509/clip.csv"
    archive = tmp_path / "g1.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(path)
        member.size = len(csv)
        tar.addfile(member, io.BytesIO(csv))
    md = tmp_path / "catalog.parquet"
    row = {
        "move_g1_path": path, "move_name": "clip", "actor_uid":"A001", "take_actor":"A001",
        "take_date":230509, "take_org_name":"take", "is_mirror":False, "move_duration_frames":n,
        **{column:"walk" for column in bones_prepare.CAPTION_COLUMNS},
    }
    pd.DataFrame([row]).to_parquet(md)
    monkeypatch.setattr(bones_prepare, "METADATA_SHA", bones_prepare.sha256_file(md))
    monkeypatch.setattr(bones_prepare, "ARCHIVE_SHA", bones_prepare.sha256_file(archive))
    root = tmp_path / "packed"
    bones_prepare.prepare(archive, md, root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["complete"] and manifest["frames"] == 6
    assert manifest["qpos_sha256"] == bones_prepare.sha256_file(root / "qpos.f32")
    with archive.open("ab") as f:
        f.write(b"trailing bytes")
    with pytest.raises(ValueError, match="archive SHA256"):
        bones_prepare.prepare(archive, md, tmp_path / "corrupt")
    assert not (tmp_path / "corrupt" / "manifest.json").exists()
