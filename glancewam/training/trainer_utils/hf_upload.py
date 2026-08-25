"""Upload a finished training run directory to the Hugging Face Hub."""

import logging
import os

logger = logging.getLogger(__name__)


def upload_run_to_hf(cfg) -> None:
    """Upload the run output dir to a Hugging Face Hub repo.

    Controlled by `cfg.hf_upload`:
      enabled (bool, default False), repo_id (str), repo_type (str, default "model"),
      path_in_repo (str, default = cfg.run_id), private (bool, default False).
    Caller must ensure this runs on the main process only.
    """
    hf_cfg = getattr(cfg, "hf_upload", None)
    if hf_cfg is None or not bool(getattr(hf_cfg, "enabled", False)):
        return

    repo_id = getattr(hf_cfg, "repo_id", None)
    if not repo_id:
        logger.warning("hf_upload.enabled=True but hf_upload.repo_id is empty; skipping upload.")
        return

    local_dir = cfg.output_dir
    if not os.path.isdir(local_dir):
        logger.warning(f"Output dir {local_dir} not found; skipping HF upload.")
        return

    path_in_repo = getattr(hf_cfg, "path_in_repo", None) or cfg.run_id
    repo_type = getattr(hf_cfg, "repo_type", "model")
    private = bool(getattr(hf_cfg, "private", False))

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=private)
        logger.info(f"📤 Uploading {local_dir} -> {repo_id}:{path_in_repo} ({repo_type})")
        # Eval/serving artifacts may land inside the run dir before upload (rollout .mp4
        # videos, server logs); HF's commit endpoint rejects raw binaries, so exclude them.
        ignore_patterns = ["**/output_eval_*/**", "**/output_server/**", "*.mp4", "**/wandb/**"]
        extra = getattr(hf_cfg, "ignore_patterns", None)
        if extra:
            ignore_patterns = ignore_patterns + list(extra)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
            ignore_patterns=ignore_patterns,
        )
        logger.info(f"✅ HF upload complete: https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")
    except Exception as e:
        logger.error(f"HF upload failed: {e}")
