import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model):
        self.model = model
        layer = model.encoder.layer4[-1]
        layer.register_forward_hook(self.forward_hook)
        layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x, target):
        self.model.zero_grad()
        output = self.model(x)
        output[0, target].backward()

        a = self.activations
        g = self.gradients
        w = g.mean(dim=(2, 3), keepdim=True)

        cam = (w * a).sum(1).relu()
        cam = F.interpolate(
            cam[:, None], (224,224),
            mode="bilinear",
            align_corners=False
        )[:,0]

        lo = cam.amin((1,2), keepdim=True)
        hi = cam.amax((1,2), keepdim=True)
        cam = (cam-lo)/(hi-lo+1e-8)

        return output.detach(), cam.detach()