"""WAM — world-action model frameworks.

A video diffusion transformer (SkyReels-V2 DF) serves as the perception backbone: one
forward pass yields both the features an action head reads and the velocity target for a
video-prediction loss.

  - ``CotrainBaseline`` — video + action co-training (the imagination-free baseline).
  - ``GlanceWAM``       — co-training plus lookahead-frame conditioning (the method).

Contract shared with every framework in this repo:
  - forward(examples) -> {"action_loss": Tensor}
  - predict_action(examples) -> {"normalized_actions": np.ndarray}
"""
