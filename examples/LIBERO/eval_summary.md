# LIBERO eval sweep

Each cell is the episode-weighted success rate over the suite's strided shards (a single
pass over all episodes — shards just parallelize it, so no run/shard count is shown). A
`(n_ok/n_shards)` suffix appears only if some shards failed to report.

| Timestamp | Exp | Steps | libero_spatial | libero_object | libero_goal | libero_10 | Avg |
|---|---|---|---|---|---|---|---|
| 2026-08-24 23:10 | glancewam_libero | 15000 | 0.994 | 1.000 | 0.996 | 0.964 | **0.989** |
