# BONES-SEED G1

OMG-Flow uses the producer's G1 CSV archive and v004 metadata from revision `2f59b2077b9da34dd4e43618e705c7cb962c9a66`. The archive is 23,499,973,647 bytes. Both files require existing access to the gated dataset. The preparer verifies the SHA-256 of both inputs and writes local training artifacts. [Pinned release](https://huggingface.co/datasets/bones-studio/seed/tree/2f59b2077b9da34dd4e43618e705c7cb962c9a66)

```sh
hf download bones-studio/seed --repo-type dataset \
  --revision 2f59b2077b9da34dd4e43618e705c7cb962c9a66 \
  g1.tar.gz metadata/seed_metadata_v004.parquet --local-dir "$RAW"
python -m omg.data.bones_prepare --archive "$RAW/g1.tar.gz" \
  --metadata "$RAW/metadata/seed_metadata_v004.parquet" --root "$DATA"
```

**Conversion.** Each CSV supplies a contiguous 120 Hz frame index. Root translation is in centimetres and root orientation uses extrinsic XYZ Euler angles in degrees. The converter resolves the 29 joint columns by their G1 joint names, converts positions to metres and angles to radians, and produces world qpos36 with scalar-first root quaternions. The 50 Hz grid starts at the first source frame and ends inside the source time interval. XYZ and joint positions use linear interpolation. Root orientation uses shortest-arc spherical interpolation. The resulting window uses the OMG-Lite native38 coordinate system anchored to the last history frame. [Producer's viewer](https://github.com/bones-studio/seed-viewer) · [NVIDIA's G1 conversion reference](https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst)

**Split.** The release supplies metadata without train/validation/test assignments. We order the 522 actors by SHA-256 of `bones-actor-v1/<actor_uid>`, hold out the first 26 for validation and the next 26 for test, and train on the remaining 470. Each source performance is identified by actor, take date and original take name. Its original and mirrored export always share a split. The preparer also checks that no identical resampled pose sequence crosses splits. There are 71,088 original/mirror pairs and 44 unpaired originals.

The pinned metadata gives the following expected counts for 10 history frames, 100 future frames and stride 25. Each window lies entirely within one source clip. Clips shorter than 110 frames stay in the published data but provide zero windows at this horizon.

| Split | Actors | Clips | 50 Hz frames | Eligible windows |
| --- | ---: | ---: | ---: | ---: |
| Train | 470 | 128,876 | 46,249,938 | 1,350,619 |
| Validation | 26 | 7,164 | 3,451,919 | 110,348 |
| Test | 26 | 6,180 | 2,213,501 | 64,590 |

**Text and provenance.** The four producer-supplied natural description fields contain 33,811 distinct captions. Each training window uses one deterministic caption variant, selected by clip and within-clip window index. Validation and test use the first description. Captions describe the entire clip, including windows within long clips. `clips.parquet` records source path, actor, original take, date, mirror flag, catalog duration, buffer offset and pose hash. `manifest.json` identifies the packed buffer, index and captions by hash. The complete packed pose buffer is 7,475,811,552 bytes. The `--limit` option publishes an explicitly incomplete smoke dataset and leaves the archive hash unverified.
