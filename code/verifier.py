import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import get_network
from utils.loading import parse_spec

DEVICE = "cpu"

class Analyzer_linear(torch.nn.Module):
    def __init__(self, layer: torch.nn.Linear):
        super().__init__()
        self.weight = layer.weight
        self.bias = layer.bias
    def forward(self, x):
        lower = x[0]
        upper = x[1]
        c = (lower+upper)/2.
        r = (upper-lower)/2.
        c = F.linear(c,self.weight,self.bias)
        r = F.linear(r,torch.abs(self.weight),None)
        return torch.stack((c-r,c+r),dim=0)

class Analyzer_relu(nn.Module):
    #TODO
    def __init__(self, layer: torch.nn.ReLU, eps: float):
        super().__init__()
        self.layer = layer
        self.eps = eps
    def forward(self, x):
        return None

def create_analyzer(net: nn.Module):
    layers = []
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Linear):
            layers.append(Analyzer_linear(layer))
        if isinstance(layer, torch.nn.ReLU):
            print(name,layer)
        if isinstance(layer, torch.nn.Flatten):
            layers.append(layer)
    return nn.Sequential(*layers)

def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    analyzer_net = create_analyzer(net)
    boxes = torch.cat((inputs-eps,inputs+eps),dim=0)
    outputs = analyzer_net(boxes)
    lower_bound_true_class = outputs[0][true_label]
    upper_bound_other_class = torch.cat((outputs[1][:true_label], outputs[1][true_label+1:]))
    result = torch.all(lower_bound_true_class > upper_bound_other_class)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)
    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
