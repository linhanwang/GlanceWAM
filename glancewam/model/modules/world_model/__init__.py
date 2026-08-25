def get_world_model(config):
    """Factory for world model backends.

    GlanceWAM ships one backbone: the SkyReels-V2 DF video diffusion transformer,
    selected by ``config.framework.world_model.base_wm``.

    Every world-model wrapper exposes:
      - ``forward(**kwargs)`` → model outputs with hidden_states
      - ``build_inputs(images, instructions)`` → dict of tensors
      - ``generate(**kwargs)`` → generation (optional)
    """
    wm_cfg = config.framework.get("world_model", None)
    wm_name = wm_cfg.get("base_wm", "") if wm_cfg is not None else ""

    if "skyreels" in wm_name.lower():
        from .SkyReelsV2DF import _SkyReelsV2DF_Interface

        return _SkyReelsV2DF_Interface(config)

    raise NotImplementedError(
        f"World model {wm_name!r} is not implemented. GlanceWAM supports SkyReels-V2 DF backbones "
        "(e.g. 'Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers')."
    )
