# src/fl/server.py

import copy


class FLServer:
    """
    Simple FedAvg server.
    """

    def __init__(self, global_model, global_encoder):
        self.global_model = global_model
        self.global_encoder = global_encoder

    def get_global_parameters(self):
        return {
            "model": {k: v.cpu().detach().clone() for k, v in self.global_model.state_dict().items()},
            "encoder": {k: v.cpu().detach().clone() for k, v in self.global_encoder.state_dict().items()},
        }

    def set_global_parameters(self, params):
        self.global_model.load_state_dict(params["model"])
        self.global_encoder.load_state_dict(params["encoder"])

    def aggregate(self, client_params_list, client_sizes=None):
        if client_sizes is None:
            client_sizes = [1] * len(client_params_list)

        total = float(sum(client_sizes))
        agg = {
            "model": copy.deepcopy(client_params_list[0]["model"]),
            "encoder": copy.deepcopy(client_params_list[0]["encoder"]),
        }

        for key in agg["model"].keys():
            agg["model"][key] = sum(
                (client_sizes[i] / total) * client_params_list[i]["model"][key]
                for i in range(len(client_params_list))
            )

        for key in agg["encoder"].keys():
            agg["encoder"][key] = sum(
                (client_sizes[i] / total) * client_params_list[i]["encoder"][key]
                for i in range(len(client_params_list))
            )

        self.set_global_parameters(agg)
        return agg