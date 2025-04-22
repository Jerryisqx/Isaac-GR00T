import asyncio
import logging
import traceback
from gr00t.model.policy import unsqueeze_dict_values, squeeze_dict_values
from gr00t.model.gr00t_n1 import GR00T_N1
import tree
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames
import torch


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                obs = unsqueeze_dict_values(obs)
                normalized_input = unsqueeze_dict_values
                normalized_input = self._policy.apply_transforms(obs)
                # model = GR00T_N1.from_pretrained(
                #     pretrained_model_name_or_path="/srv/beegfs02/scratch/qingxuan_project/data/gr00t_tmp/gr00t_libero_spatial_30k",
                #     compute_dtype="bfloat16")
                model = self._policy.model
                model_dtype = next(model.parameters()).dtype

                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model.validate_inputs(normalized_input)
                    backbone_inputs = model.backbone.prepare_input(normalized_input)
                    # print("backbone_inputs: ", backbone_inputs)
                    action_inputs = model.action_head.prepare_input(normalized_input)
                    # print("action_inputs: ", action_inputs)
                    def to_device_with_maybe_dtype(x):
                        # Only cast to self.compute_dtype if the tensor is floating
                        if not isinstance(x, (torch.Tensor, np.ndarray)):
                            return x
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x)
                        if torch.is_floating_point(x):
                            # return x.to("cuda", dtype=model.action_head.dtype)
                            return x.to("cuda", dtype=model_dtype)
                        else:
                            # Keep original dtype
                            return x.to("cuda")
                    backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
                    action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
                    backbone_outputs = model.backbone(backbone_inputs)
                    # print("Bacbone outputs: ", backbone_outputs)
                    # print("action_inputs: ", action_inputs.state.shape)
                    action_head_outputs = model.action_head.get_action(backbone_outputs, action_inputs)
                    model.validate_data(action_head_outputs, backbone_outputs, is_training=False)
                    # print("Valid action outputs")
                normalized_action = action_head_outputs["action_pred"].float()
                # print("normalized_action")
                unnormalized_action = self._policy._get_unnormalized_action(normalized_action)
                unnormalized_action = squeeze_dict_values(unnormalized_action)
                action = unnormalized_action
                # print("action: ", action)
                # action = self._policy.get_action(obs)
                await websocket.send(packer.pack(action))
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
