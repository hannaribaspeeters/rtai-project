import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import get_network
from utils.loading import parse_spec

DEVICE = "cpu"


class RelationalConstraint:
    def __init__(self, weight, bias, lower=True): 
        super().__init__()
        self.weight = weight # (out_features, in_features)
        self.bias = bias # (out_features)
        self.lower = lower # Only used for printing

    def __repr__(self) -> str:
        in_features = self.weight.shape[1]
        out_features = self.weight.shape[0]
        out = ""
        for j in range(out_features):
            out += f"a_{j}>=" if self.lower else f"a_{j}<="
            for i in range(in_features):
                if self.weight[j,i] != 0:
                    out += f"{self.weight[j,i]}*x_{i} + "
            out += f"{self.bias[j]}\n"
        return out
    
class Shape:
    def __init__(self, lb: torch.Tensor, ub: torch.Tensor, rel_lb: RelationalConstraint=None, rel_ub: RelationalConstraint=None, parent=None):
        self.lb = lb
        self.ub = ub
        self.rel_lb = rel_lb
        self.rel_ub = rel_ub
        if self.rel_ub is not None:
            self.rel_ub.lower = False
        self.parent = parent

    def __repr__(self) -> str:
        return f"Shape(lb={self.lb.tolist()},ub={self.ub.tolist()})" + (f"\n{self.rel_lb}" if self.rel_lb is not None else "") + (f"\n{self.rel_ub}" if self.rel_ub is not None else "")
        
    def backsubstitute(self):
        prev = self.parent
        if prev.rel_lb is None or prev.rel_ub is None:
            return
        print("Backsubstituting: \n",prev)
        # lower bound
        rel = self.rel_lb
        W, b = rel.weight, rel.bias
        Wmax = torch.max(torch.zeros_like(W), W)
        Wmin = torch.min(torch.zeros_like(W), W)
        rel_lb_W = Wmax @ prev.rel_ub.weight + Wmin @ prev.rel_lb.weight
        rel_lb_b = b + W @ prev.rel_ub.bias + W @ prev.rel_lb.bias
        self.rel_lb = RelationalConstraint(rel_lb_W, rel_lb_b)

        # upper bound
        rel = self.rel_ub
        W, b = rel.weight, rel.bias
        Wmax = torch.max(torch.zeros_like(W), W)
        Wmin = torch.min(torch.zeros_like(W), W)
        rel_ub_W = Wmax @ prev.rel_lb.weight + Wmin @ prev.rel_ub.weight
        rel_ub_b = b + W @ prev.rel_lb.bias + W @ prev.rel_ub.bias
        self.rel_ub = RelationalConstraint(rel_ub_W, rel_ub_b, lower=False)
        self.parent = prev.parent
        self.backsubstitute()

            

class DeepPolyBase(torch.nn.Module):
    def __init__(self, verbose=False, name=""):
        super().__init__()
        self.verbose = verbose
        self.name = name

    def print(self, out):
        if self.verbose:
            print(self.name)
            print(out)
    
    def forward(self, x: Shape) -> Shape:
        return x

class DeepPolyLinear(DeepPolyBase):
    def __init__(self, layer: torch.nn.Linear, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = layer.weight
        self.bias = layer.bias

    def forward(self, x: Shape) -> Shape:
        rel_lb = RelationalConstraint(self.weight.data, self.bias.data)
        rel_ub = RelationalConstraint(self.weight.data, self.bias.data, lower=False)

        c = (x.lb+x.ub)/2.
        r = (x.ub-x.lb)/2.
        c = F.linear(c,self.weight,self.bias)

        r = F.linear(r,torch.abs(self.weight),None)
        out = Shape(c-r,c+r, rel_lb=rel_lb, rel_ub=rel_ub, parent=x)
        self.print(out)
        return out
    
class DeepPolyReLu(DeepPolyBase):
    def __init__(self, layer: torch.nn.ReLU, eps: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.eps = eps

    def forward(self, x: Shape) -> Shape:
        rel_ub = RelationalConstraint(self.eps*torch.eye(x.lb.shape[0]), -self.eps*x.lb, lower=False)
        rel_lb = RelationalConstraint(torch.zeros(x.lb.shape[0], x.lb.shape[0]),torch.zeros(x.lb.shape[0]))
        out = Shape(F.relu(x.lb),F.relu(x.ub), rel_lb=rel_lb, rel_ub=rel_ub, parent=x)
        self.print(out)
        return out
    
def create_analyzer(net: nn.Module, verbose=False):
    layers = []
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Linear):
            layers.append(DeepPolyLinear(layer, name=name, verbose=verbose))
        if isinstance(layer, torch.nn.ReLU):
            layers.append(DeepPolyReLu(layer,0.5, name=name, verbose=verbose))
        if isinstance(layer, torch.nn.Flatten):
            layers.append(layer)
    return nn.Sequential(*layers)

def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    analyzer_net = create_analyzer(net)
    lb = inputs - eps
    ub = inputs + eps
    lb.clamp_(min=0, max=1)
    ub.clamp_(min=0, max=1)
    print(lb.min(), lb.max())   
    print(ub.min(), ub.max())
    boxes = torch.cat((lb, ub),dim=0)
    outputs = analyzer_net(boxes)
    print(outputs)
    print("True label: ", true_label)
    
    lower_bound_true_class = outputs[0][true_label]
    upper_bound_other_class = torch.cat((outputs[1][:true_label], outputs[1][true_label+1:]))
    print(lower_bound_true_class)
    print(upper_bound_other_class)
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
