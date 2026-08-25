# Copyright 2025 glancewam community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import logging
import os
import time
import traceback

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy

# Set GLANCEWAM_EVAL_PROFILE=1 to log per-batch wall-clock timing (queue depth,
# batch size, forward latency). Off by default so normal eval logs stay quiet.
_PROFILE = os.environ.get("GLANCEWAM_EVAL_PROFILE", "0") not in ("0", "", "false", "False")


def _err(msg, message: str) -> dict:
    """Build the error-response dict the client expects (mirrors _route_message)."""
    req_id = msg.get("request_id", "default") if isinstance(msg, dict) else "default"
    return {
        "status": "error",
        "ok": False,
        "type": "inference_result",
        "request_id": req_id,
        "error": {"message": message},
    }


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    # Batching kwargs that a client may send alongside `examples`. Items are
    # only batched together if their values for ALL of these keys match.
    _BATCH_KW = ("do_sample", "use_ddim", "num_ddim_steps", "cfg_scale")

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # Idle timeout in seconds, -1 means never auto-close
        metadata: dict | None = None,
        max_batch: int = 1,
        batch_wait_ms: float = 0.0,
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        self._max_batch = max(1, int(max_batch))
        self._batch_wait_ms = max(0.0, float(batch_wait_ms))
        self._infer_queue: asyncio.Queue | None = None  # created in run()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        self._infer_queue = asyncio.Queue()
        worker = asyncio.create_task(self._batch_worker(), name="batch_worker")
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
        ) as server:
            logging.info("Batching: max_batch=%d, batch_wait_ms=%.1f", self._max_batch, self._batch_wait_ms)
            try:
                if self._idle_timeout > 0:
                    await self._idle_watchdog(server)
                else:
                    await server.serve_forever()
            finally:
                worker.cancel()
                try:
                    await worker
                except (asyncio.CancelledError, Exception):
                    pass

    async def _idle_watchdog(self, server):
        """Monitor idle time and shut down the server on timeout."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()  # Refresh active time on each received message
                mtype = msg.get("type", "infer")
                if mtype in ("infer", "predict_action"):
                    fut = asyncio.get_running_loop().create_future()
                    await self._infer_queue.put((msg, fut))
                    ret = await fut
                else:
                    ret = self._route_message(msg)
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _batch_worker(self):
        """Drain the infer queue, group by uniform kwargs, run one forward per group."""
        assert self._infer_queue is not None
        while True:
            first = await self._infer_queue.get()
            items = [first]
            if self._max_batch > 1 and self._batch_wait_ms > 0:
                deadline = time.monotonic() + self._batch_wait_ms / 1000.0
                while len(items) < self._max_batch:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    try:
                        items.append(await asyncio.wait_for(self._infer_queue.get(), timeout))
                    except asyncio.TimeoutError:
                        break
            elif self._max_batch > 1:
                # No wait window: opportunistically drain whatever is already enqueued.
                while len(items) < self._max_batch and not self._infer_queue.empty():
                    items.append(self._infer_queue.get_nowait())

            # Group items whose batch-relevant kwargs match exactly so a single
            # forward pass can serve them. Disjoint kwargs run as separate batches.
            groups: dict[tuple, list] = {}
            for msg, fut in items:
                payload = msg.get("payload", msg) if isinstance(msg, dict) else {}
                key = tuple((k, payload.get(k)) for k in self._BATCH_KW)
                groups.setdefault(key, []).append((msg, fut))
            for group in groups.values():
                await self._run_uniform_batch(group)

    async def _run_uniform_batch(self, items: list) -> None:
        """Run predict_action on a list of (msg, future) pairs sharing kwargs."""
        msgs, futs = zip(*items)
        # Flatten each client's `examples` list (clients normally send length 1, but
        # we tolerate >1 — slice the response back per-client by per-msg length).
        examples: list = []
        per_msg_n: list[int] = []
        bad = False
        for msg in msgs:
            payload = msg.get("payload", msg) if isinstance(msg, dict) else {}
            ex = payload.get("examples")
            if not isinstance(ex, list) or not ex:
                bad = True
                break
            examples.extend(ex)
            per_msg_n.append(len(ex))
        if bad:
            for fut in futs:
                if not fut.done():
                    fut.set_result(_err(msgs[0], "Payload must contain non-empty 'examples' list"))
            return

        first_payload = msgs[0].get("payload", msgs[0]) if isinstance(msgs[0], dict) else {}
        kw = {k: first_payload[k] for k in self._BATCH_KW if k in first_payload}

        if _PROFILE:
            qsize_before = self._infer_queue.qsize() if self._infer_queue is not None else -1
            _t0 = time.perf_counter()
        try:
            out = await asyncio.to_thread(self._policy.predict_action, examples=examples, **kw)
        except Exception as e:
            logging.exception("Batched predict_action failed (B=%d)", len(examples))
            for msg, fut in items:
                if not fut.done():
                    fut.set_result(_err(msg, str(e)))
            return
        if _PROFILE:
            _dt_ms = (time.perf_counter() - _t0) * 1000.0
            logging.info(
                "PROFILE infer B=%d dt=%.1fms per_item=%.1fms qsize_after_pull=%d",
                len(examples),
                _dt_ms,
                _dt_ms / max(1, len(examples)),
                qsize_before,
            )

        actions = out.get("normalized_actions") if isinstance(out, dict) else None
        if actions is None:
            for msg, fut in items:
                if not fut.done():
                    fut.set_result(_err(msg, "predict_action returned no 'normalized_actions'"))
            return
        # §8.3 two-stage models also return per-sample predicted states (relative-encoded);
        # forwarded when present so the reachability measurement can read them. Absent for
        # every other model — the response shape is unchanged.
        pred_states = out.get("predicted_states") if isinstance(out, dict) else None
        # Goal-image cotrain returns the per-example goal latent used, so the stateful client
        # can cache a freshly-generated goal and send it back during the hold phase of the
        # refresh cycle. Absent for every other model — the response shape is unchanged.
        goal_latent = out.get("goal_latent") if isinstance(out, dict) else None
        # Async-proposer arm (E2 staleness): rows that asked for `force_gen` get BOTH the goal
        # they acted on (above) and the one just sampled, which they adopt after the emulated
        # proposer latency. Present only when some row asked.
        goal_latent_new = out.get("goal_latent_new") if isinstance(out, dict) else None

        offset = 0
        for (msg, fut), n in zip(items, per_msg_n):
            if not fut.done():
                req_id = msg.get("request_id", "default") if isinstance(msg, dict) else "default"
                data = {"normalized_actions": actions[offset : offset + n]}
                if pred_states is not None:
                    data["predicted_states"] = pred_states[offset : offset + n]
                if goal_latent is not None:
                    data["goal_latent"] = goal_latent[offset : offset + n]
                if goal_latent_new is not None:
                    data["goal_latent_new"] = goal_latent_new[offset : offset + n]
                fut.set_result(
                    {
                        "status": "ok",
                        "ok": True,
                        "type": "inference_result",
                        "request_id": req_id,
                        "data": data,
                    }
                )
            offset += n

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")  # default = infer
        payload = msg.get("payload", msg)  # when no explicit payload, treat top-level as payload

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer --> framework.predict_action
        elif mtype == "infer" or mtype == "predict_action":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))},
                }
            try:
                output_dict = self._policy.predict_action(**payload)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)

                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                    },
                }
            data = output_dict
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
