# RoboCasa kitchen (cosmos-policy) eval sweep

Each cell is the episode-weighted success rate over the task's strided shards (a single
pass over all episodes — shards just parallelize it). A `(n_ok/n_shards)` suffix appears
only if some shards failed to report. Avg is the unweighted mean across tasks.

| Timestamp | Exp | Steps | Avg | CloseDoubleDoor | CloseDrawer | CloseSingleDoor | CoffeePressButton | CoffeeServeMug | CoffeeSetupMug | OpenDoubleDoor | OpenDrawer | OpenSingleDoor | PnPCabToCounter | PnPCounterToCab | PnPCounterToMicrowave | PnPCounterToSink | PnPCounterToStove | PnPMicrowaveToCounter | PnPSinkToCounter | PnPStoveToCounter | TurnOffMicrowave | TurnOffSinkFaucet | TurnOffStove | TurnOnMicrowave | TurnOnSinkFaucet | TurnOnStove | TurnSinkSpout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-24 22:31 | glancewam_robocasa_kitchen | 10000 | **0.721** | 0.880 | 1.000 | 0.960 | 0.960 | 0.760 | 0.320 | 0.980 | 0.920 | 0.900 | 0.300 | 0.480 | 0.400 | 0.720 | 0.680 | 0.420 | 0.660 | 0.740 | 1.000 | 0.900 | 0.200 | 0.960 | 0.800 | 0.620 | 0.740 |
