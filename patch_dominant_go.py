import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from kornia.filters import get_gaussian_kernel2d
from kornia.filters import SpatialGradient
from kornia.constants import pi


class CustomizablePatchDominantGradientOrientation(nn.Module):
    """Module, which estimates the dominant gradient orientation of the given patches, in radians.
    Zero angle points towards right.
    Args:
            patch_size: int, default = 32
            num_angular_bins: int, default is 36
            eps: float, for safe division, and arctan, default is 1e-8"""
    def __init__(self,
                 patch_size: int = 32,
                 num_angular_bins: int = 36, eps: float = 1e-18, conf={}):
        super(CustomizablePatchDominantGradientOrientation, self).__init__()

        self.add_defaults(conf)
        self.config = conf
        print(f"conf: {self.config}")

        self.patch_size = patch_size
        self.num_ang_bins = num_angular_bins
        self.gradient = SpatialGradient('sobel', 1)
        self.eps = eps
        self.angular_smooth = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False, padding_mode="circular")
        with torch.no_grad():
            self.angular_smooth.weight[:] = torch.tensor([[[0.33, 0.34, 0.33]]])
        self.set_kernel()

    def set_kernel(self):

        bool_show_kernel = self.config['show_kernel']
        def show_kernel(kernel, title):
            if bool_show_kernel:
                plt.figure()
                plt.title(title)
                plt.imshow(kernel[0].numpy())
                plt.savefig(f"./plots/kernel_{title.replace(' ', '_')}.png")
                plt.show()
                plt.close()
            return

        sigma: float = float(self.patch_size) / math.sqrt(2.0)
        gaussian_kernel = get_gaussian_kernel2d((self.patch_size, self.patch_size), (sigma, sigma), True)

        triv_kernel = torch.ones_like(gaussian_kernel, dtype=float)
        show_kernel(triv_kernel, "trivial")
        show_kernel(gaussian_kernel, "gaussian")

        # self.weighting = self.weighting * torch.where(self.weighting < self.weighting[:, 0, int(patch_size//2)], 0.0, 1.0)
        r = torch.range(0, self.patch_size - 1)
        grid = torch.meshgrid(r, r)
        center = (self.patch_size - 1) / 2
        d = torch.sqrt((grid[0] - center) ** 2 + (grid[1] - center) ** 2)
        circ_mask_simple = torch.where(d < self.patch_size / 2, 1.0, 0.0)[None]
        show_kernel(circ_mask_simple, "circular mask simple")

        fr = torch.range(0, 99)
        grid_fr = torch.meshgrid(r, r, fr, fr)
        d_fr = torch.sqrt((grid_fr[0] + grid_fr[2] / 100 - 0.5 - center) ** 2 + (grid_fr[1] + grid_fr[3] / 100 - 0.5 - center) ** 2)
        circ_mask_finer = torch.where(d_fr < self.patch_size / 2, 1.0, 0.0).sum(2).sum(2) / 10000
        circ_mask_finer = circ_mask_finer[None]
        show_kernel(circ_mask_finer, "circular mask finer")

        if self.config["gauss_weighted"]:
            base_kernel = gaussian_kernel
        else:
            base_kernel = triv_kernel

        if self.config["circular_kernel_precise"]:
            circular_kernel = circ_mask_finer
        elif self.config["circular_kernel"]:
            circular_kernel = circ_mask_simple
        else:
            circular_kernel = triv_kernel

        self.weighting_kernel = base_kernel * circular_kernel
        show_kernel(self.weighting_kernel, "kernel used")

    def add_defaults(self, conf):
        if not conf.__contains__("refined"):
            conf["refined"] = False
        if not conf.__contains__("gauss_weighted"):
            conf["gauss_weighted"] = False
        if not conf.__contains__("circular_kernel"):
            conf["circular_kernel"] = False
        if not conf.__contains__("circular_kernel_precise"):
            conf["circular_kernel_precise"] = False
        if not conf.__contains__("show_kernel"):
            conf["show_kernel"] = False
        if not conf.__contains__("ref"):
            conf["ref"] = False

    def __repr__(self):
        return self.__class__.__name__ + '('\
            'patch_size=' + str(self.patch_size) + ', ' + \
            'num_ang_bins=' + str(self.num_ang_bins) + ', ' + \
            'eps=' + str(self.eps) + ')' + \
            'conf=' + str(self.config)

    def compute_grads(self, patch: torch.Tensor):
        self.weighting_kernel = self.weighting_kernel.to(patch.dtype).to(patch.device)
        grads: torch.Tensor = self.gradient(patch) * self.weighting_kernel
        return grads

    def forward(self, patch: torch.Tensor, grads: torch.Tensor=None) -> torch.Tensor:  # type: ignore
        """Args:
            patch: (torch.Tensor) shape [Bx1xHxW]
        Returns:
            patch: (torch.Tensor) shape [Bx1] """
        if not torch.is_tensor(patch):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(patch)))
        if not len(patch.shape) == 4:
            raise ValueError("Invalid input shape, we expect Bx1xHxW. Got: {}"
                             .format(patch.shape))
        B, CH, W, H = patch.size()
        if (W != self.patch_size) or (H != self.patch_size) or (CH != 1):
            raise TypeError(
                "input shape should be must be [Bx1x{}x{}]. "
                "Got {}".format(self.patch_size, self.patch_size, patch.size()))
        self.angular_smooth = self.angular_smooth.to(patch.dtype).to(patch.device)
        self.weighting_kernel = self.weighting_kernel.to(patch.dtype).to(patch.device)
        if grads is None:
          grads = self.compute_grads(patch)
        # unpack the edges
        gx: torch.Tensor = grads[:, :, 0]
        gy: torch.Tensor = grads[:, :, 1]

        mag: torch.Tensor = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori: torch.Tensor = torch.atan2(gy, gx + self.eps) + 2.0 * pi

        o_big = float(self.num_ang_bins) * (ori + 1.0 * pi) / (2.0 * pi)
        bo0_big = torch.floor(o_big)
        wo1_big = o_big - bo0_big
        bo0_big = bo0_big % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big) * mag  # type: ignore
        wo1_big = wo1_big * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            ang_bins.append(F.adaptive_avg_pool2d((bo0_big == i).to(patch.dtype) * wo0_big +
                                                  (bo1_big == i).to(patch.dtype) * wo1_big, (1, 1)))
        ang_bins = torch.cat(ang_bins, 1).view(-1, 1, self.num_ang_bins)   # type: ignore
        ang_bins = self.angular_smooth(ang_bins)   # type: ignore
        values, indices = ang_bins.view(-1, self.num_ang_bins).max(1)  # type: ignore
        indices_original = indices

        indices_plus_1 = torch.remainder(indices + 1, self.num_ang_bins)
        values_plus_1 = ang_bins[..., indices_plus_1].float()
        indices_minus_1 = torch.remainder(indices - 1, self.num_ang_bins)
        values_minus_1 = ang_bins[..., indices_minus_1].float()

        values_second = torch.where(values_plus_1 > values_minus_1, values_plus_1, values_minus_1)
        values_third = torch.where(values_plus_1 > values_minus_1, values_minus_1, values_plus_1)

        indices_second = torch.where(values_plus_1 > values_minus_1, indices_plus_1, indices_minus_1)
        indices_third = torch.where(values_plus_1 > values_minus_1, indices_minus_1, indices_plus_1)

        refinement = (values_plus_1 - values_minus_1) / 2.0 / (2.0 * values - (values_plus_1 + values_minus_1))
        eps_refinemnt = 1e-3

        # refinement ver. 2
        #refinement = -refinement

        if self.config["ref"]:

            ref_old = refinement
            # CONTINUE - (indices_second - indices) - modulo num_bins, but careful!
            refinement_new = (values_second - values_third) / (values + values_second - 2 * values_third) * (indices_second - indices)
            ref_on = refinement * 4 / (1 + 6 * refinement)
            #assert torch.allclose(refinement_new, ref_on, atol=1e-4)
            refinement = refinement_new

            # if refinement > 0:
            #     refinement = refinement * 4 / (1 + 6 * refinement)
            # else:
            #     pass
                # refinement = -refinement
                # refinement = refinement * 4 / (1 + 6 * refinement)
                # refinement = -refinement

        assert torch.all(refinement.abs() < 0.5 + eps_refinemnt)

        indices = indices.float() + refinement
        angle_refined = -((2. * pi * indices.to(patch.dtype) / float(self.num_ang_bins)) - pi)  # type: ignore

        # simplify
        angle_original = -((2. * pi * indices_original.to(patch.dtype) / float(self.num_ang_bins)) - pi)  # type: ignore
        # for B, CH != 1 it returns shape [BxCH]?

        #print(f"{angle_original.item() * 180 / math.pi} vs. {angle_refined.item() * 180 / math.pi}")

        if self.config["refined"]:
            angle = angle_refined
        else:
            angle = angle_original
        return angle
