# Third-party notices

The repository's MIT license covers OMG/OMG-Lite code and project contributions;
it does not replace the licenses of third-party assets or external datasets.

- **OMG / OMG-Lite:** Copyright (c) 2026 MARS Lab, Tsinghua University and
  contributors. [MIT](LICENSE), retained from
  [OMG-Lite](https://github.com/KunYing-Lee/OMG-Lite).
- **HoloMotion G1 scenes:** `assets/holomotion/g1_29dof/` derives from
  [HoloMotion v1.4.1](https://github.com/HorizonRobotics/HoloMotion/tree/v1.4.1/assets/robots/unitree/G1/29dof).
  Copyright 2025 Horizon Robotics. [Apache-2.0](licenses/HoloMotion-LICENSE);
  [upstream notices](licenses/HoloMotion-NOTICE) are retained. The robot XML's
  mesh directory was adapted to this repository. Tracker weights are not bundled.
- **Unitree G1 assets:** `assets/robots/g1/` includes the G1 URDF and meshes from
  [Unitree Robotics](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description).
  Copyright (c) 2016–2022 HangZhou YuShu TECHNOLOGY CO.,LTD.
  [BSD-3-Clause](licenses/Unitree-LICENSE). The URDF and derived kinematic metadata
  are adapted for this project's joint representation.
- **Rotation conversions:** `src/omg/utils/rotation_conversions.py` retains its
  Facebook/PyTorch3D and NVIDIA copyright headers and `BSD-3-Clause AND Apache-2.0`
  SPDX identifier. See [PyTorch3D BSD terms](licenses/PyTorch3D-LICENSE) and
  [Apache-2.0 terms](licenses/HoloMotion-LICENSE).
- **BONES-SEED:** Training data includes Motion Data by
  [Bones Studio](https://bones.studio/). Use of the underlying dataset is subject
  to the [BONES Motion Capture Dataset License Agreement](https://bones.studio/info/seed-license).
  Obtain the dataset directly from the provider under its access conditions.
  No source motion dataset or trained checkpoint is bundled. The gallery contains
  generated references and simulated executions, with selected prompt/clip metadata
  retained as experimental provenance. This notice does not grant rights to the
  underlying data or establish eligibility for its use.
