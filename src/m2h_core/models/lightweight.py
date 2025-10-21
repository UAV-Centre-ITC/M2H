import math

import torch
import torch.nn as nn
# from mmcv.cnn import ConvModule, Linear, build_activation_layer

import warnings

import torch.nn.functional as F
import itertools

def get_activation(act_layer, inplace=False):
    # If the activation is GELU, ignore inplace because it doesn't support it.
    if act_layer == nn.GELU:
        return act_layer()
    else:
        return act_layer(inplace=inplace)
    
def get_norm(norm_layer, channels):
    return norm_layer(channels) if norm_layer is not None else nn.Identity()

def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class CenterPadding(nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output
# XXX: (Untested) replacement for mmcv.imdenormalize()
def _imdenormalize(img, mean, std, to_bgr=True):
    import numpy as np

    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = (img * std) + mean
    if to_bgr:
        img = img[::-1]
    return img

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        # Store attributes for compatibility.
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = in_channels
        self.transposed = False  # Conv2d is not transposed.
        self.output_padding = (0, 0)  # No output padding for standard conv.

        # Define the depthwise and pointwise convolutions.
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=self.kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


    
class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_layer. Bias will be set as True if `norm_layer` is None, otherwise
            False. Default: "auto".
        conv_layer (nn.Module): Convolution layer. Default: None,
            which means using conv2d.
        norm_layer (nn.Module): Normalization layer. Default: None.
        act_layer (nn.Module): Activation layer. Default: nn.ReLU.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_layer=nn.Conv2d,
        norm_layer=None,
        act_layer=nn.ReLU,
        inplace=False,
        with_spectral_norm=False,
        padding_mode="zeros",
        order=("conv", "norm", "act"),
    ):
        super(ConvModule, self).__init__()
        official_padding_mode = ["zeros", "circular"]
        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(["conv", "norm", "act"])

        self.with_norm = norm_layer is not None
        self.with_activation = act_layer is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            if padding_mode == "zeros":
                padding_layer = nn.ZeroPad2d
            else:
                raise AssertionError(f"Unsupported padding mode: {padding_mode}")
            self.pad = padding_layer(padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        # self.conv = self.conv_layer(
        #     in_channels,
        #     out_channels,
        #     kernel_size,
        #     stride=stride,
        #     padding=conv_padding,
        #     dilation=dilation,
        #     groups=groups,
        #     bias=bias,
        # )

        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size,
                                               stride=stride, padding=padding,
                                               dilation=dilation, bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            norm = partial(norm_layer, num_features=norm_channels)
            self.add_module("norm", norm)
            if self.with_bias:
                from torch.nnModules.batchnorm import _BatchNorm
                from torch.nnModules.instancenorm import _InstanceNorm

                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn("Unnecessary conv bias before batch/instance norm")
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            if act_layer in [nn.PReLU, nn.Sigmoid, nn.GELU]:
                # These activations don't accept the inplace argument.
                self.activate = act_layer()
            else:
                self.activate = partial(act_layer, inplace=inplace)()

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, "init_weights"):
            if self.with_activation and isinstance(self.act_layer, nn.LeakyReLU):
                nonlinearity = "leaky_relu"
                a = 0.01  # XXX: default negative_slope
            else:
                nonlinearity = "relu"
                a = 0
            if hasattr(self.conv, "weight") and self.conv.weight is not None:
                nn.init.kaiming_normal_(self.conv.weight, a=a, mode="fan_out", nonlinearity=nonlinearity)
            if hasattr(self.conv, "bias") and self.conv.bias is not None:
                nn.init.constant_(self.conv.bias, 0)
        if self.with_norm:
            if hasattr(self.norm, "weight") and self.norm.weight is not None:
                nn.init.constant_(self.norm.weight, 1)
            if hasattr(self.norm, "bias") and self.norm.bias is not None:
                nn.init.constant_(self.norm.bias, 0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.pad(x)
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x


class ReassembleBlocks(nn.Module):
    """ViTPostProcessBlock, process cls_token in ViT backbone output and
    rearrange the feature vector to feature map.
    Args:
        in_channels (int): ViT feature channels. Default: 768.
        out_channels (List): output channels of each stage.
            Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
    """

    def __init__(self, in_channels=768, out_channels=[96, 192, 384, 768], readout_type="ignore", patch_size=16):
        super(ReassembleBlocks, self).__init__()

        assert readout_type in ["ignore", "add", "project"]
        self.readout_type = readout_type
        self.patch_size = patch_size

        self.projects = nn.ModuleList(
            [
                ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    act_layer=None,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )
        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()))

    def forward(self, inputs):
        assert isinstance(inputs, list)
        out = []
        for i, x in enumerate(inputs):
            # print(f"Input {i} Shape Before Processing: {x[0].shape}")  # Debug
            assert len(x) == 2
            x, cls_token = x[0], x[1]
            feature_shape = x.shape
            if self.readout_type == "project":
                x = x.flatten(2).permute((0, 2, 1))
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                x = x.permute(0, 2, 1).reshape(feature_shape)
            elif self.readout_type == "add":
                x = x.flatten(2) + cls_token.unsqueeze(-1)
                x = x.reshape(feature_shape)
            else:
                pass
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            # print(f"Reassemble Block {i} Output Shape:", x.shape)  # Debug
            out.append(x)
        return out

class ReassembleBlockSingle(nn.Module):
    """ViTPostProcessBlock, process cls_token in ViT backbone output and
    rearrange the feature vector to feature map.
    Args:
        in_channels (int): ViT feature channels. Default: 768.
        out_channels (List): output channels of each stage.
            Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
    """

    def __init__(self, in_channels=768, out_channel=768, readout_type="ignore", patch_size=16):
        super(ReassembleBlockSingle, self).__init__()

        assert readout_type in ["ignore", "add", "project"]
        self.readout_type = readout_type
        self.patch_size = patch_size

        self.projects = ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    act_layer=None,
                )
        self.resize_layers = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1)
        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()))

    def forward(self, input):
        # print(f"Input {i} Shape Before Processing: {x[0].shape}")  # Debug
        assert len(input) == 2
        x, cls_token = input[0], input[1]
        feature_shape = x.shape
        if self.readout_type == "project":
            x = x.flatten(2).permute((0, 2, 1))
            readout = cls_token.unsqueeze(1).expand_as(x)
            x = self.readout_projects(torch.cat((x, readout), -1))
            x = x.permute(0, 2, 1).reshape(feature_shape)
        elif self.readout_type == "add":
            x = x.flatten(2) + cls_token.unsqueeze(-1)
            x = x.reshape(feature_shape)
        else:
            pass
        x = self.projects(x)
        x = self.resize_layers(x)
        return x

# An efficient pre-activation residual unit using a single depthwise separable convolution.
class PreActResidualUnit(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            get_activation(act_layer, inplace=True),
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),
            get_norm(norm_layer, channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

# An efficient feature fusion block.
class FeatureFusionBlock(nn.Module):
    """
    Efficiently fuses multi-scale features.
    If a second input is provided, it is added via a lightweight residual branch.
    The output is then upsampled and projected to reduce channels.
    """
    def __init__(self, in_channels, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, expand=False, align_corners=True):
        super().__init__()
        self.in_channels = in_channels
        self.align_corners = align_corners
        # If expand is True, we reduce channels by a factor (e.g., 2)
        self.out_channels = in_channels if not expand else in_channels // 2

        # A simple 1x1 conv for final projection.
        self.project = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=True)
        # Use a single efficient residual unit.
        self.res_unit = PreActResidualUnit(in_channels, act_layer, norm_layer)

    def forward(self, x, res_input=None):
        # If a second input is provided, add it via the residual unit.
        if res_input is not None:
            # Match spatial dimensions if needed.
            if x.shape[2:] != res_input.shape[2:]:
                res_input = F.interpolate(res_input, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = x + self.res_unit(res_input)
        # Process x through the residual unit.
        x = self.res_unit(x)
        # Upsample spatially.
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        # Project to the output channel dimension.
        x = self.project(x)
        return x




class UpsampleBlock(nn.Module):
    """An efficient UpsampleBlock that removes ConvTranspose2d in favor of interpolation-based upsampling."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Two depthwise separable conv layers for feature processing.
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # Replace refine_conv with a grouped 1x1 conv:
        self.refine_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, groups=4, bias=False)

        # And similarly for the residual branch (if applicable):
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=4, bias=False)

        # Channel attention as before.
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Save the original input for the residual branch.
        x_orig = x
        
        # Process input features through conv layers.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Upsample using bilinear interpolation and refine.
        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_upsampled = self.refine_conv(x_upsampled)
        
        # Residual branch: upsample the original input and map to out_channels.
        residual = self.residual(F.interpolate(x_orig, scale_factor=2, mode='bilinear', align_corners=True))
        
        # Apply channel attention.
        attn = self.channel_attn(x_upsampled)
        
        return x_upsampled * attn + residual


    

class ASPP(nn.Module):
    """
    A lighter ASPP variant that uses depthwise separable convolutions and fewer branches.
    """
    def __init__(self, in_channels, out_channels, dilations=[1, 6], reduction=2):
        super().__init__()
        # Optionally reduce out_channels for each branch to further lower compute.
        branch_channels = out_channels // reduction

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels, branch_channels, kernel_size=3,
                padding=dilations[1], dilation=dilations[1], bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, branch_channels),  # GroupNorm with 1 group works well for 1x1 features.
                nn.ReLU(inplace=True)
            )

        # Final projection: Concatenate outputs from 3 branches.
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.global_avg_pool(x)
        # Upsample global branch to match spatial dims.
        x3 = F.interpolate(x3, size=x.shape[2:], mode='bilinear', align_corners=True)
        x_cat = torch.cat((x1, x2, x3), dim=1)
        return self.project(x_cat)

class RefinementBlock(nn.Module):
    """
    A lighter refinement block that uses a single depthwise separable convolution
    with a residual connection for efficient feature refinement.
    """
    def __init__(self, channels):
        super(RefinementBlock, self).__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return x + self.conv(x)

    
class HeadDepth(nn.Module):
    """Improved Depth Head with ASPP and a Refinement Block."""
    def __init__(self, features):
        super(HeadDepth, self).__init__()
        self.aspp = ASPP(features, features)
        self.upsample1 = UpsampleBlock(features, 128)
        self.upsample2 = UpsampleBlock(128, 64)
        self.refine = RefinementBlock(64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, depth_features, normals=None):
        # Apply ASPP for multi-scale context
        x = self.aspp(depth_features)
        # Upsample and refine the features
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.refine(x)
        # Apply spatial attention before final prediction
        attn = self.spatial_attn(x)
        return self.final_conv(x * attn)

class HeadSeg(nn.Module):
    """Optimized Segmentation Head with ASPP and a Refinement Block."""
    def __init__(self, features, num_classes=41):
        super(HeadSeg, self).__init__()
        self.aspp = ASPP(features, features)
        self.upsample1 = UpsampleBlock(features, 128)
        self.upsample2 = UpsampleBlock(128, 64)
        self.refine = RefinementBlock(64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, seg_features, edges=None):
        # Apply ASPP to capture multi-scale context
        x = self.aspp(seg_features)
        # Upsample and refine the features
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.refine(x)
        # Apply spatial attention to emphasize salient regions
        attn = self.spatial_attn(x)
        return self.final_conv(x * attn)


class HeadNormals(nn.Module):
    """Optimized Normal Prediction Head"""
    def __init__(self, features):
        super(HeadNormals, self).__init__()
        self.head = nn.Sequential(
            UpsampleBlock(features, 128),
            UpsampleBlock(128, 64),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        return self.head(x)


class HeadEdges(nn.Module):
    """Optimized Edge Prediction Head"""
    def __init__(self, features):
        super(HeadEdges, self).__init__()
        self.head = nn.Sequential(
            UpsampleBlock(features, 128),
            UpsampleBlock(128, 64),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowCrossTaskBlock(nn.Module):
    """
    Window-based cross-attention block with **dilated and shifted windows**
    to efficiently capture **local and global** interactions across tasks.
    """
    def __init__(self, in_channels=96, num_heads=4, window_size=7, dilation=2, shift_size=4, ffn_ratio=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.dilation = dilation  
        self.shift_size = shift_size  

        # Layer Normalization before attention
        self.norm_layers = nn.ModuleList([nn.LayerNorm(in_channels) for _ in range(4)])

        # Multi-Head Attention
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        
        # Feed-Forward Network
        hidden_dim = int(in_channels * ffn_ratio)
        self.ffn_fc1 = nn.Linear(in_channels, hidden_dim)
        self.ffn_fc2 = nn.Linear(hidden_dim, in_channels)
        
        # Post-Attention LayerNorms
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(in_channels) for _ in range(4)])

    def forward(self, eF, nF, sF, dF):
        B, C, H, W = eF.shape

        # Window partitioning with dilation and shifting
        e_win = self._window_partition(eF)
        n_win = self._window_partition(nF)
        s_win = self._window_partition(sF)
        d_win = self._window_partition(dF)

        # Normalize before attention
        e_win, n_win, s_win, d_win = [ln(w) for w, ln in zip([e_win, n_win, s_win, d_win], self.norm_layers)]

        # Concatenate for cross-task attention
        combined = torch.cat([e_win, n_win, s_win, d_win], dim=1)
        attn_out, _ = self.attn(combined, combined, combined)
        combined2 = combined + attn_out  

        # Feed-Forward Network
        ff_out = self.ffn_fc2(F.gelu(self.ffn_fc1(combined2)))
        combined3 = combined2 + ff_out  

        # Separate tasks back
        e2, n2, s2, d2 = torch.split(combined3, [e_win.shape[1], n_win.shape[1], s_win.shape[1], d_win.shape[1]], dim=1)
        e2, n2, s2, d2 = [ln(w) for w, ln in zip([e2, n2, s2, d2], self.norm2_layers)]

        # Reverse window partitioning
        return [self._window_reverse(w, B) for w in [e2, n2, s2, d2]]
    
    def _window_partition(self, x):
        """ Splits feature maps into non-overlapping windows with proper padding handling. """
        B, C, H, W = x.shape
        ws = self.window_size

        # Ensure H and W are multiples of ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)  # Pad as needed

        # Store original and padded dimensions
        self.last_H_orig, self.last_W_orig = H, W  
        self.last_H_after, self.last_W_after = x.shape[2], x.shape[3]

        # Compute number of windows
        num_win_h = self.last_H_after // ws
        num_win_w = self.last_W_after // ws
        num_win = num_win_h * num_win_w

        # Reshape into windows
        x = x.view(B, C, num_win_h, ws, num_win_w, ws)  
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  
        x = x.view(B * num_win, ws * ws, C)  

        return x

    def _window_reverse(self, x, B):
        """ Reverses window partitioning, restores original spatial shape """
        ws = self.window_size
        H, W = self.last_H_orig, self.last_W_orig
        num_win_h = self.last_H_after // ws
        num_win_w = self.last_W_after // ws

        x = x.view(B, num_win_h, num_win_w, ws, ws, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.in_channels, self.last_H_after, self.last_W_after)

        # Remove padding to restore original size
        x = x[:, :, :H, :W]  
        return x


class AdvancedFeatureRestoration(nn.Module):
    """
    A lightweight version of the AdvancedFeatureRestoration module.
    It fuses original, attended, and unique features using a simplified gating mechanism
    and a reduced FFN expansion.
    """
    def __init__(self, in_channels, group_size=4):
        super().__init__()
        # 1x1 projections for each stream.
        self.conv_orig = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=group_size, bias=False)
        self.conv_attn = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=group_size, bias=False)
        self.conv_uniq = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=group_size, bias=False)
        
        # Global pooling to extract context from concatenated streams.
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Simplified gating: single 1x1 convolution followed by Sigmoid.
        self.gate_fc = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        ffn_mid_channels = (in_channels*3)//2
        # Simplified feed-forward network with reduced expansion.
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, ffn_mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ffn_mid_channels, in_channels, kernel_size=1, bias=False)
        )
        # Final 1x1 conv to mix channels remains unchanged.
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
    
    def forward(self, attended_features, original_features, unique_features=None):
        # Compute the projected representations.
        orig = self.conv_orig(original_features)
        attn = self.conv_attn(attended_features)
        uniq = self.conv_uniq(unique_features) if unique_features is not None else torch.zeros_like(orig)

        # Ensure spatial sizes match (if necessary).
        if uniq.shape[2:] != orig.shape[2:]:
            uniq = F.interpolate(uniq, size=orig.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate along channel dimension.
        concat = torch.cat([orig, attn, uniq], dim=1)
        # Compute a global context.
        context = self.global_pool(concat)
        # Compute gating weights.
        gates = self.gate_fc(context)
        # Split gating weights into three equal parts.
        B, total, _, _ = gates.size()
        gate_orig, gate_attn, gate_uniq = torch.split(gates, total // 3, dim=1)
        
        # Apply gating.
        gated_orig = orig * gate_orig
        gated_attn = attn * gate_attn
        gated_uniq = uniq * gate_uniq
        
        # Fuse the gated streams.
        fused = gated_orig + gated_attn + gated_uniq
        # Process with FFN (residual connection).
        out = fused + self.ffn(fused)
        out = self.out_conv(out)
        return out



class UniqueDepthFeature(nn.Module):
    """
    Computes a unique feature for the depth task from the last backbone feature.
    """
    def __init__(
        self,
        dinov2_embed_dims=768,
        post_process_channels=768,  # Use the last element of your post_process_channels list
        readout_type="ignore",
        patch_size=16,
        shared_channels=96,
        act_layer=nn.ReLU,
        norm_layer=None
    ):
        super().__init__()
        # Reassemble tokens into a spatial feature map.
        # Here we assume that ReassembleBlocks is configured to output a single feature map.
        self.reassemble = ReassembleBlockSingle(
            in_channels=dinov2_embed_dims,
            out_channel=post_process_channels,  # using the last channel value only
            readout_type=readout_type,
            patch_size=patch_size
        )
        # Project the reassembled feature into the shared channel space.
        self.conv_proj = ConvModule(
            in_channels=post_process_channels,
            out_channels=shared_channels,
            kernel_size=3,
            padding=1,
            act_layer=act_layer,
            norm_layer=norm_layer
        )
        # Fuse the projected feature using a dedicated fusion block.
        self.fusion_block = FeatureFusionBlock(
            in_channels=shared_channels,
            act_layer=act_layer,
            norm_layer=norm_layer,
            expand=False,
            align_corners=True
        )
        self.fusion_block.res_conv_unit1 = None

    def forward(self, backbone_last_feature):
        # Apply reassembly to the last backbone feature.
        # (Assuming backbone_last_feature is the token representation from the last layer.)
        reassembled_feat = self.reassemble(backbone_last_feature)
        # Project the reassembled feature.
        proj_feat = self.conv_proj(reassembled_feat)
        # Fuse the projected feature.
        unique_feat = self.fusion_block(proj_feat)
        return unique_feat

# -----------------------------------------------------------------------------
# Similarly, you can define a unique semantic feature module.
# If the configuration is identical except for the head later, you can reuse
# the same structure or adjust parameters as needed.
# -----------------------------------------------------------------------------
class UniqueSemanticFeature(nn.Module):
    """
    Computes a unique feature for the semantic task from the last backbone feature.
    """
    def __init__(
        self,
        dinov2_embed_dims=768,
        post_process_channels=[768],  # using the last channel value
        readout_type="ignore",
        patch_size=16,
        shared_channels=96,
        act_layer=nn.ReLU,
        norm_layer=None
    ):
        super().__init__()
        self.reassemble = ReassembleBlockSingle(
            in_channels=dinov2_embed_dims,
            out_channel=post_process_channels,
            readout_type=readout_type,
            patch_size=patch_size
        )
        self.conv_proj = ConvModule(
            in_channels=post_process_channels,
            out_channels=shared_channels,
            kernel_size=3,
            padding=1,
            act_layer=act_layer,
            norm_layer=norm_layer
        )
        self.fusion_block = FeatureFusionBlock(
            in_channels=shared_channels,
            act_layer=act_layer,
            norm_layer=norm_layer,
            expand=False,
            align_corners=True
        )
        self.fusion_block.res_conv_unit1 = None

    def forward(self, backbone_last_feature):
        reassembled_feat = self.reassemble(backbone_last_feature)
        proj_feat = self.conv_proj(reassembled_feat)
        unique_feat = self.fusion_block(proj_feat)
        return unique_feat


class MLTHead(nn.Module):
    """DPT-like structure with window-based cross attention for multi-task features,
    now with unique fusion branches for each task.
    """

    def __init__(
        self,
        embed_dims=768,
        post_process_channels=[96, 192, 384, 768],
        readout_type="ignore",
        patch_size=16,
        channels=96,
        act_layer=nn.ReLU,
        align_corners=False,
        min_depth=1e-3,
        max_depth=None,
        num_classes=41,
        norm_layer=None,
        scale_up=False,
        window_size=7,       # for local cross-attn
        cross_num_heads=4,   # for local cross-attn
        num_cross_layers=2,
    ):
        super(MLTHead, self).__init__()

        self.channels = channels
        self.align_corners = align_corners
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_classes = num_classes
        self.scale_up = scale_up

        reduction_ratio = 4
        reduced_channels = self.channels // reduction_ratio
        # Reassemble ViT output (shared among tasks)
        self.reassemble_blocks = ReassembleBlocks(embed_dims, post_process_channels, readout_type, patch_size)

        # Reduce channels for multi-scale features (shared)
        self.convs = nn.ModuleList(
            [
                ConvModule(channel, self.channels, kernel_size=3, padding=1, act_layer=None, bias=False)
                for channel in post_process_channels
            ]
        )

        # -----------------------
        # Create unique fusion branches for each task:
        # -----------------------

        # For edges:
        self.fusion_blocks_edges = nn.ModuleList(
            [FeatureFusionBlock(self.channels, act_layer, norm_layer) for _ in range(len(self.convs))]
        )
        self.fusion_blocks_edges[0].res_conv_unit1 = None
        self.project_edges = ConvModule(self.channels, self.channels, kernel_size=3, padding=1, norm_layer=norm_layer)

        # For normals:
        self.fusion_blocks_normals = nn.ModuleList(
            [FeatureFusionBlock(self.channels, act_layer, norm_layer) for _ in range(len(self.convs))]
        )
        self.fusion_blocks_normals[0].res_conv_unit1 = None
        self.project_normals = ConvModule(self.channels, self.channels, kernel_size=3, padding=1, norm_layer=norm_layer)

        # For semantic segmentation:
        self.fusion_blocks_sem_seg = nn.ModuleList(
            [FeatureFusionBlock(self.channels, act_layer, norm_layer) for _ in range(len(self.convs))]
        )
        self.fusion_blocks_sem_seg[0].res_conv_unit1 = None
        self.project_sem_seg = ConvModule(self.channels, self.channels, kernel_size=3, padding=1, norm_layer=norm_layer)

        # For depth:
        self.fusion_blocks_depth = nn.ModuleList(
            [FeatureFusionBlock(self.channels, act_layer, norm_layer) for _ in range(len(self.convs))]
        )
        self.fusion_blocks_depth[0].res_conv_unit1 = None
        self.project_depth = ConvModule(self.channels, self.channels, kernel_size=3, padding=1, norm_layer=norm_layer)

        # (NEW) Cross-Task block that merges features in local windows.
        self.cross_attn = nn.ModuleList([
            WindowCrossTaskBlock(
                in_channels=reduced_channels,
                num_heads=cross_num_heads,
                window_size=window_size,
                ffn_ratio=4,
            )
            for _ in range(num_cross_layers)
        ])

        # Unique feature modules for depth and semantics.
        self.unique_depth_feature = UniqueDepthFeature(
            dinov2_embed_dims=embed_dims,
            post_process_channels=post_process_channels[-1],
            readout_type="ignore",
            patch_size=patch_size,
            shared_channels=self.channels,
            act_layer=act_layer,
            norm_layer=norm_layer
        )
        self.unique_semantic_feature = UniqueSemanticFeature(
            dinov2_embed_dims=embed_dims,
            post_process_channels=post_process_channels[-1],
            readout_type="ignore",
            patch_size=patch_size,
            shared_channels=self.channels,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

        # Per-task heads.
        self.conv_depth = HeadDepth(self.channels)
        self.seg_out    = HeadSeg(self.channels, self.num_classes)
        self.conv_normal= HeadNormals(self.channels)
        self.conv_edge  = HeadEdges(self.channels)

        self.restore_depth = AdvancedFeatureRestoration(self.channels)
        self.restore_semseg = AdvancedFeatureRestoration(self.channels)
    

        # Projection layers for each task before cross attention
        self.pre_attn_proj = nn.ModuleDict({
            "edges": nn.Conv2d(self.channels, reduced_channels, kernel_size=1),
            "normals": nn.Conv2d(self.channels, reduced_channels, kernel_size=1),
            "sem_seg": nn.Conv2d(self.channels, reduced_channels, kernel_size=1),
            "depth": nn.Conv2d(self.channels, reduced_channels, kernel_size=1),
        })

        # After cross attention, upsample back
        self.post_attn_proj = nn.ModuleDict({
            "edges": nn.Conv2d(reduced_channels, self.channels, kernel_size=1),
            "normals": nn.Conv2d(reduced_channels, self.channels, kernel_size=1),
            "sem_seg": nn.Conv2d(reduced_channels, self.channels, kernel_size=1),
            "depth": nn.Conv2d(reduced_channels, self.channels, kernel_size=1),
        })

        # For spatial downsampling:
        self.spatial_downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        # For spatial upsampling:
        self.spatial_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def depth_pred(self, depth_output):
        """Predict depth with or without sigmoid scaling."""
        if self.scale_up and self.max_depth is not None:
            return torch.sigmoid(depth_output) * self.max_depth
        else:
            return F.relu(depth_output) + self.min_depth

    def fuse_branch(self, fusion_blocks, project, features):
        """Helper to fuse multi-scale features using a branch-specific fusion path."""
        out = fusion_blocks[0](features[-1])
        for i in range(1, len(fusion_blocks)):
            out = fusion_blocks[i](out, features[-(i+1)])
        out = project(out)
        return out

    def forward(self, inputs):
        # 1) Reassemble multi-scale features
        features = self.reassemble_blocks([inp for inp in inputs])
        features = [self.convs[i](f) for i, f in enumerate(features)]
        unique_depth = self.unique_depth_feature(inputs[-1])
        unique_semantic = self.unique_semantic_feature(inputs[-1])
        
        # 2) Fusion for each task using unique branches
        task_feats = {}
        task_feats["edges"]   = self.fuse_branch(self.fusion_blocks_edges, self.project_edges, features)
        task_feats["normals"] = self.fuse_branch(self.fusion_blocks_normals, self.project_normals, features)
        task_feats["sem_seg"] = self.fuse_branch(self.fusion_blocks_sem_seg, self.project_sem_seg, features)
        task_feats["depth"]   = self.fuse_branch(self.fusion_blocks_depth, self.project_depth, features)
    
        # 3) Window-based cross-attention among the 4 tasks
        # Before cross attention: Project to lower channel dimension and optionally downsample spatially.
        eF = self.pre_attn_proj["edges"](task_feats["edges"])
        nF = self.pre_attn_proj["normals"](task_feats["normals"])
        sF = self.pre_attn_proj["sem_seg"](task_feats["sem_seg"])
        dF = self.pre_attn_proj["depth"](task_feats["depth"])

        # Optional: Spatial downsampling
        eF = self.spatial_downsample(eF)
        nF = self.spatial_downsample(nF)
        sF = self.spatial_downsample(sF)
        dF = self.spatial_downsample(dF)

        # 3) Window-based cross-attention among the 4 tasks
        for cross_layer in self.cross_attn:
            eF, nF, sF, dF = cross_layer(eF, nF, sF, dF)

        # Optional: Upsample spatially back to original size
        eF = self.spatial_upsample(eF)
        nF = self.spatial_upsample(nF)
        sF = self.spatial_upsample(sF)
        dF = self.spatial_upsample(dF)

        # After cross attention: Project back to original channel dimension.
        eF = self.post_attn_proj["edges"](eF)
        nF = self.post_attn_proj["normals"](nF)
        sF = self.post_attn_proj["sem_seg"](sF)
        dF = self.post_attn_proj["depth"](dF)
        
        # 4) Inject unique features into the restoration process.
        dF_restored = self.restore_depth(dF, task_feats["depth"], unique_features=unique_depth)
        sF_restored = self.restore_semseg(sF, task_feats["sem_seg"], unique_features=unique_semantic)
        
        # 5) Final heads for each task.
        edge_output   = self.conv_edge(eF)
        normal_output = self.conv_normal(nF)
        normal_output = F.normalize(normal_output, p=2, dim=1)
        depth_output  = self.conv_depth(dF_restored)
        depth_output  = self.depth_pred(depth_output)
        seg_output    = self.seg_out(sF_restored)

        return depth_output, seg_output, normal_output, edge_output

     

from enum import Enum
from functools import partial
from typing import Optional, Tuple, Union
class Weights(Enum):
    NYU = "NYU"
    KITTI = "KITTI"


def _get_depth_range(pretrained: bool, weights: Weights = Weights.NYU) -> Tuple[float, float]:
    if not pretrained:  # Default
        return (0.001, 10.0)
    # Pretrained, set according to the training dataset for the provided weights
    if weights == Weights.KITTI:
        return (0.001, 80.0)
    if weights == Weights.NYU:
        return (0.001, 10.0)
    return (0.001, 10.0)

class DepthEncoderDecoder(nn.Module):
    """Encoder Decoder depther.
    EncoderDecoder typically consists of backbone and decode_head.
    """

    def __init__(self, backbone, mlt_head ):
        super(DepthEncoderDecoder, self).__init__()

        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.mlt_head = mlt_head
        self.align_corners = self.mlt_head.align_corners

    def forward(self, img):
        """Encode images with backbone and decode into a depth estimation
        map of the same size as input."""
        with torch.no_grad():
            x = self.backbone(img)
        depth, seg, normals, edges = self.mlt_head(x)
        #out = torch.clamp(out, min=self.decode_head.min_depth, max=self.decode_head.max_depth)
        size = img.shape[2:]
        depth = resize(input=depth, size=size, mode="bilinear", align_corners=self.align_corners)
        seg = resize(input=seg, size=size, mode="bilinear", align_corners=self.align_corners)
        normals = resize(input=normals, size=size, mode="bilinear", align_corners=self.align_corners)
        edges = resize(input=edges, size=size, mode="bilinear", align_corners=self.align_corners)
        return depth, seg, normals, edges
    
