# OMG-Flow validation

Measurements on 7 November 2026 used the flow implementation at `4098b67`. The pilot used the first 2,048 genuine G1 clips from the pinned Bones-Seed archive. Its manifest was marked incomplete. The 10-frame histories and 100-frame futures contained real frames from a single clip. The pilot’s validation split contained one actor.

## Objective, continuation and restart

- Tests check the exact straight-line interpolation and velocity target, suffix-only loss, binary prefix mask and clean endpoint time. A two-plan test advances the world anchor and retains the exact world poses already exposed to the consumer.
- On CPU, eight uninterrupted updates matched a four-update checkpoint plus four resumed updates bit for bit across all 54 model tensors and 54 optimizer states of the small test model. The 43,545,202-parameter production model separately completed a CPU save and resume.
- An eight-window fixed example test used actual motion, T5 tokens, Gaussian noise, flow times and prefix lengths. Velocity MSE fell from 2.680 to 0.011 in 300 updates. This measures whether the optimizer can fit that finite batch.
- The 33,811-caption cache was checked against an independent FP32 T5 forward pass on eight captions. The 173 valid tokens had identical masks, relative L2 error 0.0002083 and minimum cosine similarity 0.9999997.
- The local suite passed 266 tests and skipped one CUDA-only test. Four 5090 ranks subsequently completed a BF16 matrix multiply, an NCCL all-reduce, real optimizer updates, held-out sampling and a checkpoint restart.

## Four RTX 5090s

Each pilot used the same ten-layer, width-512 transformer and frozen T5 tokens. Throughput is the median steady-state global windows per second. GPU utilization is averaged over simultaneous one-second samples on the four selected cards between retained training logs. Peak memory is PyTorch's maximum allocation per card. The dataset fit in host memory.

| Batch per GPU | Global batch | Windows/s | GPU utilization | Peak GiB |
|---:|---:|---:|---:|---:|
| 128 | 512 | 8,237 | 86.6% | 5.55 |
| 256 | 1,024 | 9,584 | 96.1% | 10.22 |
| 384 | 1,536 | 8,960 | 97.8% | 14.98 |
| 512 | 2,048 | 8,704 | 98.2% | 19.59 |

The long-run preset uses 256 per GPU, the fastest measured batch. The extended run targets 100,000 updates with 250 warm-up steps and 50 Hz native38 output.

## Prefix conditioning

Both arms start with the same weights and consume the same windows, captions and number of optimizer updates. One arm samples P uniformly from 0 to 50, the other always trains at P=0. At evaluation, both receive 25 committed frames. The two-plan seam test generates the first plan, advances 25 frames, then passes only generated history and already-committed poses to the second call. Its first new pose is at absolute frame 50. The matching ground-truth seam measures 11.74 rad/s².

| Training prefix | Step | P25 flow MSE | Joint MAE (rad) | Root error (m) | Replanned seam acceleration (rad/s²) |
|---|---:|---:|---:|---:|---:|
| 0–50 | 500 | 0.430 | 0.230 | 0.195 | 220.4 |
| 0 | 500 | 0.439 | 0.250 | 0.222 | 319.0 |
| 0–50 | 1,000 | 0.321 | 0.194 | 0.149 | 114.7 |
| 0 | 1,000 | 0.353 | 0.248 | 0.169 | 281.2 |
| 0–50 | 1,500 | 0.272 | 0.178 | 0.134 | 91.7 |
| 0 | 1,500 | 0.300 | 0.233 | 0.168 | 268.6 |

At step 1,500 the conditioned arm reduced seam acceleration by 65.9% relative to zero-prefix training. Its interior acceleration was 57.7 rad/s². The tradeoff also appears when the sampler starts from P=0: velocity MSE was 0.314 in the random-prefix arm and 0.290 in the zero-prefix arm. One seed and one held-out pilot actor limit the comparison; physical tracking requires separate evaluation.

On one RTX 5090, the step-500 checkpoint returned 75 editable world-space poses for P=25. Three warmed, synchronized, ten-step BF16 Euler trials averaged 128.0 ms for batch 128 and 585.8 ms for batch 512. Repeating the same noise reproduced the output exactly, as did changing only the Gaussian values underneath the committed prefix. These are batched sampling measurements.

## Full training inputs

The G3 conversion read and verified all 142,220 source clips (51,915,358 native50 frames). Its binary qpos buffer is 7,475,811,552 bytes, SHA-256 `804845a793b7716a2c83c844a845eb5b25f432601d0bd9cb2286d41c12a7e822`. The split contains 470 training actors and 26 each for validation and test. The stride-one training set has 32,237,215 eligible windows. Native38 normalization used 1,350,619 stride-25 training windows, giving 148,568,090 frame observations.

| Input | SHA-256 in the G3 run |
|---|---|
| Source archive | `52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a` |
| Packed-data manifest | `43d2e1631d1d10b1a04eecd127952860380c4839f351eece7e11b2bc9ce699d8` |
| Native38 statistics | `fb88294a29308de23f3989f41c92d8bb9230e14665b3d4f3447a9908ebb2564a` |
| Frozen T5 cache manifest | `9a547738333e37555d11f71637d1513d7254889fcc35a2db9d7ef58655af6675` |
