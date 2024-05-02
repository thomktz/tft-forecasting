import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


default_quantiles = [0.1, 0.5, 0.9]


class TimeDistributed(nn.Module):
    """
    Stacks the time dimension with the batch dimension of the inputs before applying the module.

    Source: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4

    Parameters
    ----------
    module : nn.Module
        The wrapped module.
    batch_first: bool
        Whether the batch dimension is expected to be the first dimension of the input or not.
    return_reshaped: bool
        Whether to return the output in the corresponding original shape or not.
    """

    def __init__(self, module: nn.Module, batch_first: bool = True, return_reshaped: bool = True) -> None:
        super().__init__()
        self.module: nn.Module = module
        self.batch_first: bool = batch_first
        self.return_reshaped: bool = return_reshaped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TimeDistributed module."""
        # in case the incoming tensor is a two-dimensional tensor - infer no temporal information is involved,
        # and simply apply the module
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and time-steps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * time-steps, input_size)
        # apply the module on each time-step separately
        y = self.module(x_reshape)

        # reshaping the module output as sequential tensor (if required)
        if self.return_reshaped:
            if self.batch_first:
                y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, time-steps, output_size)
            else:
                y = y.view(-1, x.size(1), y.size(-1))  # (time-steps, samples, output_size)

        return y


class TSDataset(Dataset):
    """Time-series dataset."""

    def __init__(
        self,
        id_col,
        static_cols,
        time_col,
        input_cols,
        target_col,
        time_steps,
        max_samples,
        input_size,
        encode_length,
        num_static,
        output_size,
        data,
    ):
        self.time_steps = time_steps
        self.input_size = input_size
        self.output_size = output_size
        self.encode_length = encode_length

        data.sort_values(by=[id_col, time_col], inplace=True)
        print("Getting valid sampling locations.")

        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in data.groupby(id_col):
            num_entries = len(df)
            if num_entries >= self.time_steps:
                valid_sampling_locations += [
                    (identifier, self.time_steps + i) for i in range(num_entries - self.time_steps + 1)
                ]
            split_data_map[identifier] = df

        self.inputs = np.zeros((max_samples, self.time_steps, self.input_size))
        self.outputs = np.zeros((max_samples, self.time_steps, self.output_size))
        self.time = np.empty((max_samples, self.time_steps, 1))
        self.identifiers = np.empty((max_samples, self.time_steps, num_static))

        if max_samples > 0 and len(valid_sampling_locations) > max_samples:
            print("Extracting {} samples...".format(max_samples))
            ranges = [
                valid_sampling_locations[i]
                for i in np.random.choice(len(valid_sampling_locations), max_samples, replace=False)
            ]
        else:
            print("Max samples={} exceeds # available segments={}".format(max_samples, len(valid_sampling_locations)))
            ranges = valid_sampling_locations

        for i, tup in enumerate(ranges):
            if ((i + 1) % 10000) == 0:
                print(i + 1, "of", max_samples, "samples done...")
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - self.time_steps : start_idx]
            self.inputs[i, :, :] = sliced[input_cols]
            self.outputs[i, :, :] = sliced[[target_col]]
            self.time[i, :, 0] = sliced[time_col]
            self.identifiers[i, :, :] = sliced[static_cols]

        self.sampled_data = {
            "inputs": self.inputs,
            "outputs": self.outputs[:, self.encode_length :, :],
            "active_entries": np.ones_like(self.outputs[:, self.encode_length :, :]),
            "time": self.time,
            "identifier": self.identifiers,
        }

    def __getitem__(self, index):
        """Get the sample at the given index."""
        return {
            "inputs": self.inputs[index],
            "outputs": self.outputs[index, self.encode_length :, :],
            "active_entries": np.ones_like(self.outputs[index, self.encode_length :, :]),
            "time": self.time[index],
            "identifier": self.identifiers[index],
        }

    def __len__(self):
        """Length of the dataset."""
        return self.inputs.shape[0]


class QuantileLoss(nn.Module):
    """From: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629."""

    def __init__(self, quantiles: list[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """Forward pass of the quantile loss."""
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
