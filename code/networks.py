import torch
import torch.nn as nn
from typing import List, Tuple, Optional


def dln_model(
    layers: List[int], in_ch: int = 1, in_dim: int = 28, num_class: int = 10
) -> nn.Sequential:
    model_layers = []
    model_layers.append(nn.Flatten())
    in_dim = in_ch * in_dim**2

    for layer in layers:
        model_layers.append(nn.Linear(in_dim, layer))
        in_dim = layer

    model_layers.append(nn.Linear(in_dim, num_class))

    return nn.Sequential(*model_layers)


def dln_conv_model(
    convolutions: List[Tuple[int, int, int, int, float]],
    layers: List[int],
    in_ch: int = 1,
    in_dim: int = 28,
    num_class: int = 10,
) -> nn.Sequential:
    model_layers = []
    img_dim = in_dim
    prev_channels = in_ch

    for n_channels, kernel_size, stride, padding, slope in convolutions:
        model_layers += [
            nn.Conv2d(
                prev_channels, n_channels, kernel_size, stride=stride, padding=padding
            ),
        ]

        prev_channels = n_channels
        img_dim = img_dim // stride

    model_layers.append(nn.Flatten())

    prev_fc_size = prev_channels * img_dim * img_dim

    for layer in layers:
        model_layers.append(nn.Linear(prev_fc_size, layer))
        prev_fc_size = layer

    model_layers.append(nn.Linear(prev_fc_size, num_class))

    return nn.Sequential(*model_layers)


def fc_model(
    activations: List[Tuple[float, int]],
    in_ch: int = 1,
    in_dim: int = 28,
    num_class: int = 10,
) -> nn.Sequential:
    layers = []
    layers.append(nn.Flatten())
    in_dim = in_ch * in_dim**2
    for act in activations:
        layers.append(nn.Linear(in_dim, act[1]))
        if act[0] == 0.0:
            layers.append(nn.ReLU())
        else:
            layers.append(nn.LeakyReLU(act[0]))
        in_dim = act[1]

    layers.append(nn.Linear(in_dim, num_class))

    return nn.Sequential(*layers)


def conv_model(
    convolutions: List[Tuple[int, int, int, int, float]],
    activations: List[Tuple[float, int]],
    in_ch: int = 1,
    in_dim: int = 28,
    num_class: int = 10,
) -> nn.Sequential:
    layers = []
    img_dim = in_dim
    prev_channels = in_ch

    for n_channels, kernel_size, stride, padding, slope in convolutions:
        layers += [
            nn.Conv2d(
                prev_channels, n_channels, kernel_size, stride=stride, padding=padding
            ),
        ]

        if slope == 0.0:
            layers.append(nn.ReLU())
        else:
            layers.append(nn.LeakyReLU(slope))

        prev_channels = n_channels
        img_dim = img_dim // stride

    layers.append(nn.Flatten())

    prev_fc_size = prev_channels * img_dim * img_dim

    for act in activations:
        layers.append(nn.Linear(prev_fc_size, act[1]))
        if act[0] == 0.0:
            layers.append(nn.ReLU())
        else:
            layers.append(nn.LeakyReLU(act[0]))
        prev_fc_size = act[1]

    layers.append(nn.Linear(prev_fc_size, num_class))

    return nn.Sequential(*layers)


def get_network(
    name: str, dataset: str = "mnist", weight_path: str = "", device: str = "cpu"
) -> nn.Sequential:
    """Get network with specific architecture in eval mode.

    Args:
        name (str): Base network architecture
        dataset (str, optional): Dataset used (some architectures have a model for MNIST and
        CIFAR10). Defaults to "mnist".
        weight_path (str, optional): Path to load model weights from. Defaults to "".
        device (str, optional): Device to load model on. Defaults to "cpu".

    Returns:
        nn.Sequential: Resulting model
    """

    model: Optional[nn.Sequential] = None

    assert dataset in ["mnist", "cifar10"], f"Invalid dataset: {dataset}"

    in_ch, in_dim = (1, 28) if dataset == "mnist" else (3, 32)

    if name == "fc_base":  # DLN
        model = dln_model(layers=[50, 50, 50], in_ch=in_ch, in_dim=in_dim, num_class=10)
    elif name == "fc_1":  # RELU
        model = fc_model(
            activations=[(0.0, 50)], in_ch=in_ch, in_dim=in_dim, num_class=10
        )
    elif name == "fc_2":
        model = fc_model(
            activations=[(0.0, 100), (0.0, 100)],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=10,
        )
    elif name == "fc_3":  # Leaky ReLU - Base
        model = fc_model(
            activations=[(0.5, 100), (0.5, 100)],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=10,
        )
    elif name == "fc_4":
        model = fc_model(
            activations=[(2.0, 100), (2.0, 100)],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=10,
        )
    elif name == "fc_5":  # Leaky ReLU - Mixed
        model = fc_model(
            activations=[(0.5, 100), (2.0, 100), (0.5, 100)],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=10,
        )
    elif name == "fc_6":
        model = fc_model(
            activations=[(0.1, 100), (0.3, 100), (0.5, 100), (3, 100), (0.5, 100)],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=10,
        )
    elif name == "fc_7":  # Leaky ReLU - Deep
        model = fc_model(
            activations=[
                (0.5, 100),
                (2.0, 100),
                (0.5, 100),
                (3.0, 100),
                (0.2, 100),
                (2.0, 100),
                (0.2, 100),
            ],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=10,
        )
    elif name == "conv_base":  # No activation just convolution + Linear
        model = dln_conv_model(
            convolutions=[(16, 3, 2, 1, 0.0), (8, 3, 2, 1, 0.0)],
            layers=[50],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=10,
        )
    elif name == "conv_1":  # Conv_1
        model = conv_model(
            convolutions=[(16, 3, 2, 1, 0.0)],
            activations=[(0.0, 100), (0.0, 10)],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=10,
        )
    elif name == "conv_2":  # Conv_2
        model = conv_model(
            convolutions=[(16, 4, 2, 1, 0.0), (32, 4, 2, 1, 0.0)],
            activations=[(0.3, 100), (0.2, 10)],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=10,
        )
    elif name == "conv_3":  # Conv_3
        model = conv_model(
            convolutions=[(16, 4, 2, 1, 0.0), (64, 4, 2, 1, 0.0)],
            activations=[(0.0, 100), (0.0, 100), (0.0, 10)],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=10,
        )
    elif name == "conv_4":  # Conv_4
        model = conv_model(
            convolutions=[(16, 4, 2, 1, 0.3), (64, 4, 2, 1, 1.5)],
            activations=[(0.5, 100), (0.5, 100), (0.5, 10)],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=10,
        )
    ### Hidden networks
    elif name == "fc_8":  # Leaky ReLU - Deep
        model = fc_model(
            activations=[
                (0.1, 100),
                (0.2, 100),
                (0.5, 100),
                (0.75, 100),
                (1.0, 100),
                (1.35, 100),
                (5.0, 100),
            ]
        )
    elif name == "conv_5":  # Conv_5
        model = conv_model(
            convolutions=[(16, 4, 2, 1, 0.2), (64, 4, 2, 1, 2.0)],
            activations=[(0.1, 100), (0.8, 100), (2.0, 10)],
            in_ch=in_ch,
            in_dim=in_dim,
            num_class=10,
        )
    ### Hidden networks            
    else:
        assert False, f"Invalid network name: {name}"

    assert model is not None, f"Model is None for {name}"

    if len(weight_path) > 0:
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))

    model.to(device)
    model.eval()

    return model
