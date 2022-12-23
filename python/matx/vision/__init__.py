# -*- coding:utf-8 -*-
# Copyright 2022 ByteDance Ltd. and/or its affiliates.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .base import VISION_CUDA_LIB, VISION_CPU_LIB

from .auto_contrast_op import AutoContrastOp
from .average_blur_op import AverageBlurOp
# from .bilateral_filter import BilateralFilterOp
from .cast_op import CastOp
from .channel_reorder_op import ChannelReorderOp
from .color_linear_adjust_op import ColorLinearAdjustOp
from .conv2d_op import Conv2dOp, SharpenOp, EmbossOp, EdgeDetectOp
from .crop_op import CenterCropOp, CropOp
from .cvt_color_op import CvtColorOp
from .flip_op import FlipOp
from .gamma_contrast_op import GammaContrastOp
from .gauss_noise_op import GaussNoiseOp
from .gaussian_blur_op import GaussianBlurOp
from .hist_equalize_op import HistEqualizeOp
from .imdecode_op import ImdecodeOp, ImdecodeRandomCropOp, ImdecodeNoExceptionOp, ImdecodeNoExceptionRandomCropOp
from .invert_op import InvertOp
from .laplacian_blur_op import LaplacianBlurOp
from .median_blur_op import MedianBlurOp
from .mixup_images_op import MixupImagesOp
from .normalize_op import NormalizeOp, TransposeNormalizeOp
from .pad_op import PadOp, PadWithBorderOp
from .posterize_op import PosterizeOp
from .random_resized_crop_op import RandomResizedCropOp
from .reduce_op import SumOp, MeanOp
from .resize_op import ResizeOp
from .rotate_op import RotateOp
from .salt_n_pepper_op import SaltAndPepperOp, RandomDropoutOp
from .solarize_op import SolarizeOp
from .split_op import SplitOp
from .stack_op import StackOp
from .transpose_op import TransposeOp
from .warp_affine_op import WarpAffineOp
from .warp_perspective_op import WarpPerspectiveOp


from .opencv._cv_defines import *
from .opencv._cv_interpolation_flags import *
from .opencv._cv_color_conversion_codes import *
from .opencv._cv_rotate_flags import *
from .opencv._cv_border_types import *
from .constants._resize_mode import *
from .constants._data_format import *
from .constants._flip_mode import *
from .constants._sync_mode import ASYNC, SYNC, SYNC_CPU

__all__ = [
    "AutoContrastOp",
    "AverageBlurOp",
    "CastOp",
    "ChannelReorderOp",
    "ColorLinearAdjustOp",
    "Conv2dOp",
    "SharpenOp",
    "EmbossOp",
    "EdgeDetectOp",
    "CenterCropOp",
    "CropOp",
    "CvtColorOp",
    "FlipOp",
    "GammaContrastOp",
    "GaussNoiseOp",
    "GaussianBlurOp",
    "HistEqualizeOp",
    "ImdecodeOp",
    "ImdecodeRandomCropOp",
    "ImdecodeNoExceptionOp",
    "ImdecodeNoExceptionRandomCropOp",
    "InvertOp",
    "LaplacianBlurOp",
    "MedianBlurOp",
    "MixupImagesOp",
    "NormalizeOp",
    "TransposeNormalizeOp",
    "PadOp",
    "PadWithBorderOp",
    "PosterizeOp",
    "RandomResizedCropOp",
    "SumOp",
    "MeanOp",
    "ResizeOp",
    "RotateOp",
    "SaltAndPepperOp",
    "RandomDropoutOp",
    "SolarizeOp",
    "SplitOp",
    "StackOp",
    "TransposeOp",
    "WarpAffineOp",
    "WarpPerspectiveOp"
]
