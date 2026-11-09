# Tracking

HoloMotion consumes native 50 Hz reference poses. The synchronous
[HoloTracker](../deployment/sync_robot.py) builds reference features and combines
them with measured robot state before each tracker call. MuJoCo executes ten
2 ms physics steps per 20 ms control tick using metadata PD gains and scales.

The [SyncScheduler](../deployment/sync_flow.py) preserves a clean 25-frame prefix
and regenerates only the suffix every ten ticks. It requires the prefix to
cover both the tracker preview and its additional velocity-stencil frame.
History is previously committed reference, not measured execution.

There are no balance forces, action corrections or reset-on-fall. Reports retain
the full trajectory after a height/tilt failure. The SDK boundary still needs
native sensor, joint-order, enable, watchdog and emergency-stop integration.
See [deployment](../deployment/README.md).
