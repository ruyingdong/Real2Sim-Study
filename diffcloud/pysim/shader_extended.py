import warnings
from typing import Optional
import torch
import torch.nn as nn

import itertools
from typing import Tuple
import torch.nn.functional as F
from pytorch3d.renderer import BlendParams
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
#from pytorch3d.common.datatypes import Device
from typing import Optional, Union
device = Union[str, torch.device]
from typing import NamedTuple, Sequence, Union
from pytorch3d import _C
class BlendParams(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Union[torch.Tensor, Sequence[float]] = (1.0, 1.0, 1.0)

def _get_background_color(
    blend_params: BlendParams, device: device, dtype=torch.float32
) -> torch.Tensor:
    background_color_ = blend_params.background_color
    if isinstance(background_color_, torch.Tensor):
        background_color = background_color_.to(device)
    else:
        background_color = torch.tensor(background_color_, dtype=dtype, device=device)
    return background_color

from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
class SplatterBlender(torch.nn.Module):

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        device,
    ):
        """
        A splatting blender. See `forward` docs for details of the splatting mechanism.

        Args:
            input_shape: Tuple (N, H, W, K) indicating the batch size, image height,
                image width, and number of rasterized layers. Used to precompute
                constant tensors that do not change as long as this tuple is unchanged.
        """
        super().__init__()
        self.crop_ids_h, self.crop_ids_w, self.offsets = _precompute(
            input_shape, device
        )



    def to(self, device):
        self.offsets = self.offsets.to(device)
        self.crop_ids_h = self.crop_ids_h.to(device)
        self.crop_ids_w = self.crop_ids_w.to(device)
        super().to(device)



    def forward(
        self,
        colors: torch.Tensor,
        pixel_coords_cameras: torch.Tensor,
        cameras: FoVPerspectiveCameras,
        background_mask: torch.Tensor,
        blend_params: BlendParams,
    ) -> torch.Tensor:
        pixel_coords_screen, colors = _prepare_pixels_and_colors(
            pixel_coords_cameras, colors, cameras, background_mask
        )  # (N, H, W, K, 3) and (N, H, W, K, 4)

        occlusion_layers = _compute_occlusion_layers(
            pixel_coords_screen[..., 2:3].squeeze(dim=-1)
        )  # (N, H, W, 9)

        splat_colors_and_weights = _compute_splatting_colors_and_weights(
            pixel_coords_screen,
            colors,
            blend_params.sigma,
            self.offsets,
        )  # (N, H, W, K, 9, 5)

        splat_colors_and_weights = _offset_splats(
            splat_colors_and_weights,
            self.crop_ids_h,
            self.crop_ids_w,
        )  # (N, H, W, K, 9, 5)

        (
            splatted_colors_per_occlusion_layer,
            splatted_weights_per_occlusion_layer,
        ) = _compute_splatted_colors_and_weights(
            occlusion_layers, splat_colors_and_weights
        )  # (N, H, W, 4, 3) and (N, H, W, 1, 3)

        output_colors = _normalize_and_compose_all_layers(
            _get_background_color(blend_params, colors.device),
            splatted_colors_per_occlusion_layer,
            splatted_weights_per_occlusion_layer,
        )  # (N, H, W, 4)

        return output_colors
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.shader import (
    flat_shading,
    gouraud_shading,
    phong_shading,
)

class ShaderBase(nn.Module):
    def __init__(
        self,
        device = "cpu",
        cameras: Optional[TensorProperties] = None,
        lights: Optional[TensorProperties] = None,
        materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
    ) -> None:
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def _get_cameras(self, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of the shader."
            raise ValueError(msg)

        return cameras

    # pyre-fixme[14]: `to` overrides method defined in `Module` inconsistently.

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self
    

class HardDepthShader(ShaderBase):
    """
    Renders the Z distances of the closest face for each pixel. If no face is
    found it returns the zfar value of the camera.

    Output from this shader is [N, H, W, 1] since it's only depth.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardDepthShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask = fragments.pix_to_face[..., 0:1] < 0

        zbuf = fragments.zbuf[..., 0:1].clone()
        zbuf[mask] = zfar
        return zbuf