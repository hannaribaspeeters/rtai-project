import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import scipy.linalg as linalg

from networks import get_network
from utils.loading import parse_spec

DEVICE = "cpu"

def debug(*args, **kwargs):
    pass#print(*args, **kwargs)

class RelationalConstraint:
    def __init__(self, weight, bias, lower=True): 
        super().__init__()
        self.weight = weight # (out_features, in_features)
        self.bias = bias # (out_features)
        self.lower = lower # Only used for printing

        assert self.weight.shape[0] == self.bias.shape[0]
        assert len(self.weight.shape) == 2 # (out_features, in_features)
        assert len(self.bias.shape) == 1 # (out_features)

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

    def evaluate(self, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
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
        return f"Shape(lb={[round(el, 3) for el in self.lb.tolist()]},ub={[round(el, 3) for el in self.ub.tolist()]})" + (f"\n{self.rel_lb}" if self.rel_lb is not None else "") + (f"\n{self.rel_ub}" if self.rel_ub is not None else "")

    def backsubstitute(self):
        if self._resolved:
            return
        rel_lb = self.rel_lb
        rel_ub = self.rel_ub
        node = self.parent
        assert (self.lb <= self.ub).all()

        while(not node._input):
            # lower bound
            W, b = rel_lb.weight, rel_lb.bias
            Wmax = torch.max(torch.zeros_like(W), W)
            Wmin = torch.min(torch.zeros_like(W), W)

            rel_lb_W = Wmax @ node.rel_lb.weight + Wmin @ node.rel_ub.weight
            rel_lb_b = b + Wmax @ node.rel_lb.bias + Wmin @ node.rel_ub.bias
            rel_lb = RelationalConstraint(rel_lb_W, rel_lb_b)

            # upper bound
            rel = rel_ub
            W, b = rel.weight, rel.bias
            Wmax = torch.max(torch.zeros_like(W), W)
            Wmin = torch.min(torch.zeros_like(W), W)
            rel_ub_W = Wmax @ node.rel_ub.weight + Wmin @ node.rel_lb.weight
            rel_ub_b = b + Wmax @ node.rel_ub.bias + Wmin @ node.rel_lb.bias
            rel_ub = RelationalConstraint(rel_ub_W, rel_ub_b, lower=False)

            node = node.parent

        # resolve the relational constraints with respect to the input
        self.lb = torch.max(rel_lb.evaluate(node.lb, node.ub), self.lb)
        self.ub = torch.min(rel_ub.evaluate(node.lb, node.ub), self.ub)
        
        # check for validity
        assert self.lb.isfinite().all()
        assert self.ub.isfinite().all()
        assert (self.lb <= self.ub).all()
        self._resolved = True
        debug("Resolved:", self)


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
    def __init__(self, layer: nn.Linear, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.weight = layer.weight
        self.bias = layer.bias

    def forward(self, x: Shape) -> Shape:
        debug("LinIn:", x)
        # 
        rel_lb = RelationalConstraint(self.weight.data, self.bias.data)
        rel_ub = RelationalConstraint(self.weight.data, self.bias.data, lower=False)

        c = (x.lb+x.ub)/2.
        r = (x.ub-x.lb)/2.
        c = torch.matmul(self.weight,c)+self.bias
         r = torch.matmul(torch.abs(self.weight),r)

        out = Shape(c-r, c+r, rel_lb=rel_lb, rel_ub=rel_ub, parent=x)
        self.print(out)
        return out
    
def outputShapeConv2d(input_len,kernel_shape,padding,stride):
    h=w=input_len
    _,_,kh,kw = kernel_shape
    ph,pw = padding
    sh,sw = stride
    output_height = int((h + ph+pw - kh) / (sh) + 1)
    output_width= int((w + pw+ph - kw) / (sw) + 1)
    return (output_height,output_width)

    
class DeepPolyConv2d(DeepPolyBase):
    def __init__(self, layer: torch.nn.Conv2d, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Reshape the weight and bias to be compatible with the linear layer
        self.weight = layer.weight.view(layer.weight.shape[0],-1)
        self.bias = layer.bias.unsqueeze(1)
        self.layer = layer

    def forward(self, x: Shape) -> Shape:
        rel_lb = RelationalConstraint(self.weight.data, self.bias.data)
        rel_ub = RelationalConstraint(self.weight.data, self.bias.data, lower=False)
        fold_params = dict(kernel_size=self.layer.kernel_size, dilation=self.layer.dilation, padding=self.layer.padding, stride=self.layer.stride)

        c = (x.lb+x.ub)/2.
        r = (x.ub-x.lb)/2.

        unfold = torch.nn.Unfold(**fold_params)
        c_unf = unfold(c)
        r_unf = unfold(r)

        c_unf = self.weight @ c_unf + self.bias
        r_unf = torch.abs(self.weight) @ r_unf
        
        output_shape = outputShapeConv2d(x.lb.shape[-1],self.layer.weight.shape,self.layer.padding,self.layer.stride)
        fold = torch.nn.Fold(output_size=output_shape,kernel_size=1)

        c = fold(c_unf)
        r = fold(r_unf)

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
        
        # Case 1: Below-zero -> set to zero
        case1 = (x.ub < 0.)
        W_l, b_l = torch.zeros_like(x.lb), torch.zeros_like(x.lb)
        W_u, b_u = torch.zeros_like(x.lb), torch.zeros_like(x.lb)

        # Case 2: Above-zero -> propagate
        case2 = (x.lb > 0.)
        W_l, b_l = torch.where(case2, torch.ones_like(x.lb), W_l), b_l
        W_u, b_u = torch.where(case2, torch.ones_like(x.lb), W_u), b_u

        # Case 3: Crossing
        # Compute lambda for the lower bound (such that the area is minimal)
        lambda_lb = torch.ones_like(x.lb)
        lambda_lb[x.ub <= -x.lb] = 0

        # Note to Hanna
        # We can later register as a parameter and learn this lambda
        # lambda_lb = torch.nn.Parameter(lambda_lb)

        crossing = ~case1 & ~case2
        W_l = torch.where(crossing, lambda_lb, W_l)
        # We don't need to update b_l, since it is zero
        W_u = torch.where(crossing, torch.div(x.ub, x.ub-x.lb), W_l)
        b_u = torch.where(crossing, -torch.div(x.ub*x.lb, x.ub-x.lb), b_l)
       

        rel_ub = RelationalConstraint(torch.diag(W_u), b_u, lower=False)
        rel_lb = RelationalConstraint(torch.diag(W_l), b_l)
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
        if isinstance(layer, torch.nn.Conv2d):
            layers.append(DeepPolyConv2d(layer, name=name, verbose=verbose))
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

    analyzer_net = create_analyzer(net, verbose=False)
    assert inputs.shape[0] == 1 # Only one batchsize one supported, TODO: generalize, should be easy, just need to add the batch dim to the shape class (and their constructions)
    lb = inputs - eps
    ub = inputs + eps
    lb.clamp_(min=min, max=max)
    ub.clamp_(min=min, max=max)
    #Add batch dim
    lb = lb.unsqueeze(0)
    ub = ub.unsqueeze(0)

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
