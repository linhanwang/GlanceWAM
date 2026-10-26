# lerobot_policy_glancewam (inference reference)

The GlanceWAM policy as we run it on real robots, packaged as a [LeRobot](https://github.com/huggingface/lerobot)
policy plugin. This is a **partial, inference-only release**: it shows how the lookahead (goal) is
generated asynchronously during deployment. Training code for this path and the robot scripts are not
included yet; a fuller release will follow.

## Where the asynchronous goal lives

In simulation the environment clock stops while the policy runs. So the released RoboCasa eval client
only *emulates* lookahead staleness (`GOAL_STALE_ROWS` / `force_gen` in
`examples/Robocasa_kitchen/eval_files/model2robocasa_kitchen_interface.py`). On a real
robot the lookahead is generated in a separate process while the control loop keeps acting:

- **`lerobot_policy_glancewam/goal_worker.py`**: the worker.
  - `GoalWorker` spawns a child process that loads its own copy of the checkpoint and runs only the
    video-DiT sampler (`_goal_worker_main`).
  - Requests and replies are small CPU tensors passed through queues: `submit`, a non-blocking `poll`,
    and a blocking `collect`.
  - The two processes take turns on the GPU. `gpu_busy()` blocks the worker for the length of each
    action-chunk decode, and the worker checks it before every Euler step (`yield_gpu`).
- **`lerobot_policy_glancewam/modeling_glancewam.py`**: the control side. The control loop never waits
  on the sampler.
  - `predict_action_chunk` polls for a finished lookahead and installs it (`_install_goal`). It then
    decodes the chunk with whatever lookahead it holds, and sends the current frame to the worker when a
    refresh is due (`_submit_goal`, `_request_due`).
  - `select_action` / `_goal_step_queued` do the same on every queued tick between chunks. A new
    lookahead therefore takes effect on the tick it arrives, about 10 rows (0.5 s) after its frame at
    20 Hz on an RTX 4090, not at the next chunk boundary.
  - Only the first lookahead of an episode is waited for (`_goal_step_chunk`). Under Real-Time Chunking,
    not even that one: the first chunk uses the no-goal fallback.
  - The goal-time diff fed to the action head is measured from the lookahead's own frame
    (`_goal_distance`), as in training.
- **`configuration_glancewam.py`**: `goal_async` turns the worker on. `goal_refresh_rows` sets the
  refresh period (default: the horizon `future_frame_idx`). `rows_per_second` drives the wall-clock row
  clock under RTC.

With `goal_async=false` (the default, and what `lerobot-eval` uses) the lookahead is sampled inline on
the refresh chunk. That matches the synchronous evaluation in the paper.

## Running it

Tested against LeRobot `0.6.2` at
[`bf31dd7`](https://github.com/huggingface/lerobot/commit/bf31dd794ffb4f87380aba3912f64421e8352d3c).
Install it next to LeRobot with `pip install -e .`, and LeRobot discovers it by package name.
`python -m lerobot_policy_glancewam.convert_checkpoint` turns a GlanceWAM trainer checkpoint into a
LeRobot one. A real-robot rollout looks like this:

```bash
lerobot-rollout --policy.path=<checkpoint> --robot.type=<your_robot> ... \
    --policy.goal_async=true --policy.rows_per_second=<fps> \
    --inference.type=rtc --inference.rtc.execution_horizon=8 \
    --inference.rtc.max_guidance_weight=10 --inference.queue_threshold=12
```

`tests/` holds tiny-model GPU tests, including the async row clock (`pytest tests/`).

## Citation

See the [GlanceWAM README](../../README.md#citation).
