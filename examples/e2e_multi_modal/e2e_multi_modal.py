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
import matx
from .image_modal import MatxImagenetVisionProcessor
from .text_modal import MatxBertTokenizer
from torch.utils.data import DataLoader


@matx.script
class MultiModalPipeline:
    def __init__(self,
                 vocab_path: str,
                 lower_case: bool = False,
                 max_tokens_per_input: int = 256,
                 unk_token: str = '[UNK]',
                 vision_device_id: int = -1,
                 is_train: bool = True):
        self.text_processor: Any = MatxBertTokenizer(
            vocab_path, lower_case, max_tokens_per_input, unk_token
        )
        self.vision_processor: Any = MatxImagenetVisionProcessor(
            vision_device_id, is_train
        )

    # the input is a batch of data
    # assume each data is like {"text": "some text", "image": b"some image"}
    # the output would be collated, organize the result in any format as you want
    # the code below would output the processed data like
    # {"images": batched_image, "input_ids": batched_input_id, "input_mask": batched_input_mask}
    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, matx.NDArray]:
        texts: List[str] = [item["text"] for item in data]
        images: List[bytes] = [item["image"] for item in data]
        processed_texts: Dict[str, matx.NDArray] = self.text_processor(texts)
        processed_images: matx.NDArray = self.vision_processor(images)
        res: Dict[str, matx.NDArray] = {}
        for k in processed_texts:
            res[k] = processed_texts[k]
        res["images"] = processed_images
        return res


class DemoDataset:
    def __init__(self, is_train=True):
        # If want to run the code, please download the demo image and vocabulary file
        # from github, or just replace them with your own ones
        f = open("demo.jpeg", "rb")
        img = f.read()
        f.close()
        text = b"this is a demo"
        self.data = {"text": text, "image": img}
        self.transform = MultiModalPipeline("vocab.txt", is_train=is_train)

    def __len__(self):
        return 100  # some fake number

    def __getitem__(self, indices):
        batch_data = [self.data] * len(indices)
        transformed_data = self.transform(batch_data)
        res = {}
        # convert each matx.NDArray to torch tensor
        for k in transformed_data.keys():
            res[k] = transformed_data[k].torch()
        return res


if __name__ == "__main__":
    dataset = DemoDataset()
    loader = DataLoader(dataset)
    for data in loader:
        print(data["images"].shape)
        print(data["input_ids"].shape)
