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

    def resolve(self, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        Wmax = torch.max(torch.zeros_like(self.weight), self.weight)
        Wmin = torch.min(torch.zeros_like(self.weight), self.weight)
        if self.lower:
            return self.bias + Wmax @ lb + Wmin @ ub
        return self.bias + Wmax @ ub + Wmin @ lb
    
class Shape:
    def __init__(self, lb: torch.Tensor, ub: torch.Tensor, rel_lb: RelationalConstraint=None, rel_ub: RelationalConstraint=None, parent=None):
        self.lb = lb
        self.ub = ub
        assert (self.lb <= self.ub).all()
        self.rel_lb = rel_lb
        self.rel_ub = rel_ub
        if self.rel_ub is not None:
            self.rel_ub.lower = False
        self.parent = parent

        # Resolved is only used for testing script in a notebook. Will store the result. And make sure 
        # that we don't backsubstitute twice, in normal analysis this is not done.
        self._resolved = False

        self._input = False
        if parent is None:
            self._input = True
            self._resolved = True

    def __repr__(self) -> str:
        return f"Shape(lb={self.lb.tolist()},ub={self.ub.tolist()})" + (f"\n{self.rel_lb}" if self.rel_lb is not None else "") + (f"\n{self.rel_ub}" if self.rel_ub is not None else "")
        
    def backsubstitute(self):
        if self._resolved:
            return
        prev = self.parent
        if prev._input: # Recursion base case
            # resolve the relational constraints with respect to the input
            self.lb = self.rel_lb.resolve(prev.lb, prev.ub)
            self.ub = self.rel_ub.resolve(prev.lb, prev.ub)
            assert (self.lb <= self.ub).all()
            self._resolved = True
            return
        
        # lower bound
        rel = self.rel_lb
        W, b = rel.weight, rel.bias
        Wmax = torch.max(torch.zeros_like(W), W)
        Wmin = torch.min(torch.zeros_like(W), W)
        rel_lb_W = Wmax @ prev.rel_lb.weight + Wmin @ prev.rel_ub.weight
        rel_lb_b = b + Wmax @ prev.rel_lb.bias + Wmin @ prev.rel_ub.bias
        self.rel_lb = RelationalConstraint(rel_lb_W, rel_lb_b)

        # upper bound
        rel = self.rel_ub
        W, b = rel.weight, rel.bias
        Wmax = torch.max(torch.zeros_like(W), W)
        Wmin = torch.min(torch.zeros_like(W), W)
        rel_ub_W = Wmax @ prev.rel_ub.weight + Wmin @ prev.rel_lb.weight
        rel_ub_b = b + Wmax @ prev.rel_ub.bias + Wmin @ prev.rel_lb.bias
        self.rel_ub = RelationalConstraint(rel_ub_W, rel_ub_b, lower=False)
        self.parent = prev.parent # We update the parent to the parent of the parent, since we have resolved the parent
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
        self.layer = layer

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
        x.backsubstitute()
        # option 1 (lb = 0) if u <= -l
        lambda_lb = torch.ones_like(x.lb)
        lambda_lb[x.ub <= -x.lb] = 0

        # Note to Hanna
        # We can later register as a parameter and learn this lambda
        # lambda_lb = torch.nn.Parameter(lambda_lb)

        rel_ub = RelationalConstraint(torch.diag(torch.div(x.ub, x.ub-x.lb)),-torch.div(x.ub*x.lb, x.ub-x.lb), lower=False)
        rel_lb = RelationalConstraint(torch.diag(lambda_lb),torch.zeros(x.lb.shape))
        out = Shape(F.relu(x.lb),F.relu(x.ub), rel_lb=rel_lb, rel_ub=rel_ub, parent=x)
        self.print(out)
        return out
    
class DeepPolyFlatten(DeepPolyBase):
    def __init__(self, verbose=False, name=""):
        super().__init__(verbose, name)
    
    def forward(self, x: Shape) -> Shape:
        # assumes that this is the first layer
        out = Shape(torch.flatten(x.lb),torch.flatten(x.ub))
        self.print(out)
        return out

class VerificationHead(nn.Linear):
    """
    Linear layer that verifies that the true class is the largest output. Constraint is verified if
    all values of the output shape lower bound are positive.

    Basically calculates
    x_true - x_other for all classes (and just x_true for the true class)
    """
    def __init__(self, num_classes: int, true_class=False) -> None:
        super().__init__(num_classes, num_classes, bias=True)
        self.weight.data = -torch.eye(num_classes) 
        self.weight.data[:, true_class] = 1
        self.bias.data = torch.zeros(num_classes)
        self.true_class = true_class
    
    def forward(self, x: Shape) -> Shape:
        return super().forward(x)
    
def create_analyzer(net: nn.Module, verbose=False):
    layers = []
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Linear):
            layers.append(DeepPolyLinear(layer, name=name, verbose=verbose))
        if isinstance(layer, torch.nn.ReLU):
            layers.append(DeepPolyReLu(layer,0.5, name=name, verbose=verbose))
        if isinstance(layer, torch.nn.Flatten):
            layers.append(DeepPolyFlatten(name=name, verbose=verbose))
    return nn.Sequential(*layers)

def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int, min: float = 0, max: float = 1) -> bool:
    # Create validation head
    num_classes = net[-1].out_features
    validation_head = VerificationHead(num_classes, true_label)
    net.add_module("verification_head", validation_head)

    # turn off gradients
    for param in net.parameters():
        param.requires_grad = False

    analyzer_net = create_analyzer(net)
    assert inputs.shape[0] == 1 # Only one batchsize one supported, TODO: generalize, should be easy, just need to add the batch dim to the shape class (and their constructions)
    lb = inputs - eps
    ub = inputs + eps
    lb.clamp_(min=min, max=max)
    ub.clamp_(min=min, max=max)

    # remove batch dimension
    lb = lb.squeeze(0)
    ub = ub.squeeze(0)

    input_shape = Shape(lb, ub)
    output_shape = analyzer_net(input_shape)
    output_shape.backsubstitute()

    result = (output_shape.lb > 0).all().item()
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
