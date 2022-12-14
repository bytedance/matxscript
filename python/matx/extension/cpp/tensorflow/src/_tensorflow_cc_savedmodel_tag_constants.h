// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Taken from
https://github.com/tensorflow/tensorflow/blob/v2.5.3/tensorflow/cc/saved_model/tag_constants.h

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CC_SAVED_MODEL_TAG_CONSTANTS_H_
#define TENSORFLOW_CC_SAVED_MODEL_TAG_CONSTANTS_H_

namespace tensorflow {

/// Tag for the `gpu` graph.
constexpr char kSavedModelTagGpu[] = "gpu";

/// Tag for the `tpu` graph.
constexpr char kSavedModelTagTpu[] = "tpu";

/// Tag for the `serving` graph.
constexpr char kSavedModelTagServe[] = "serve";

/// Tag for the `training` graph.
constexpr char kSavedModelTagTrain[] = "train";

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_TAG_CONSTANTS_H_
