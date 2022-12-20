.. example

End-to-end Deep Learning Example
##################################

This section will introduce an end-to-end example of using Matx in a multimodal training task. It will only focus on the data preprocessing part for images and texts, and the multimodal model itself is not covered. To get the source, please visit `here <https://github.com/bytedance/matxscript/blob/main/examples/e2e_multi_modal>`_.

1. Text Modal Preprocessing
**********************************
| We'll first use Matx to implement a Bert style tokenizer.

1.1 Helper Functions
===========
In order to make the code clean, we'll define some helper functions (ops) below

* Text Cleaner
.. code-block:: python3 

    class TextCleaner:
        """TextCleaner impl by matx."""

        def __init__(self) -> None:
            self.white_regex: matx.Regex = matx.Regex(r"[ \t\n\r\p{Zs}]")
            self.control_regex: matx.Regex = matx.Regex(
                r"[\u0000\ufffd\p{Cc}\p{Cf}\p{Mn}]")

            self.space: bytes = " ".encode()
            self.empty: bytes = "".encode()

        def __call__(self, text: bytes) -> bytes:
            t = self.white_regex.replace(text, self.space)
            return self.control_regex.replace(t, self.empty)

* Case Normalizer for lowercase the word is necessary
.. code-block:: python3 

    class CaseNormalizer:
        def __init__(self, do_lowercase: bool = False, unicode_norm: str = '') -> None:
            self.do_lowercase: bool = do_lowercase

        def __call__(self, text: bytes) -> bytes:
            if self.do_lowercase:
                return text.lower()
            else:
                return text

* Processing punctuations
.. code-block:: python3 

    class PunctuationPadding:
        """Pad a space around the punctuation."""

        def __init__(self):
            self.regex_pattern: matx.Regex = matx.Regex(
                r"([\u0021-\u002f]|[\u003a-\u0040}]|[\u005b-\u0060}]|[\u007b-\u007e]|\p{P})")
            self.replace_pattern: bytes = r" ${1} ".encode()

        def __call__(self, text: bytes) -> bytes:
            return self.regex_pattern.replace(text, self.replace_pattern)

1.2 Matx based BertTokenizer
===========
With the helper functions ready, we can then define the Bert Tokenizer
.. code-block:: python3

    import matx
    from matx.text import WordPieceTokenizer

    class MatxBertTokenizer:
        def __init__(self,
                     vocab_path: str,
                     lower_case: bool = False,
                     max_tokens_per_input: int = 256,
                     unk_token: str = '[UNK]'
                     ) -> None:
            """
            matx style BertTokenzier。
            vocab_path: vocabulary path for tokenizer
            lower_case: convert to lowercase or not
            max_tokens_per_input: token length limit
            unk_token: the symbol for unknown tokens
            """
            self.cleaner: TextCleaner = TextCleaner()
            self.normalizer: CaseNormalizer = CaseNormalizer(True)
            self.punc_padding: PunctuationPadding = PunctuationPadding()
            self.max_tokens_per_input: int = max_tokens_per_input
            self.world_piece: Any = WordPieceTokenizer(vocab_path=vocab_path,
                                                       unk_token=unk_token,
                                                       max_bytes_per_token=max_tokens_per_input)
            self.cls_id: int = self.world_piece.tokenize(['[CLS]'])[0]
            self.sep_id: int = self.world_piece.tokenize(['[SEP]'])[0]
            self.pad_id: int = self.world_piece.tokenize(['[PAD]'])[0]
            

        def __call__(self, texts: List[bytes]) -> Dict[str, matx.NDArray]:
            batch_input_ids: List = []
            batch_input_mask: List = []
            batch_segment_ids: List = []
            for text in texts:
                text = self.cleaner(text)
                text = self.normalizer(text)
                text = self.punc_padding(text)
                terms: List = text.split()
                tokens: List[int] = self.world_piece.tokenize(terms)
                # start to create bert style input
                len_tre: int = self.max_tokens_per_input - 2
                input_ids: List = [self.cls_id] + tokens[:len_tre] + [self.sep_id]
                input_mask: List = [1] * len(input_ids) + [0] * (self.max_tokens_per_input - len(input_ids))
                input_ids = input_ids + [self.pad_id] * (self.max_tokens_per_input - len(input_ids))
                segment_ids = [0] * self.max_tokens_per_input
                batch_input_ids.append(input_ids)
                batch_input_mask.append(input_mask)
                batch_segment_ids.append(segment_ids)
            res: Dict = {}
            res["input_ids"] = matx.NDArray(batch_input_ids, [], "int64")
            res["input_mask"] = matx.NDArray(batch_input_mask, [], "int64")
            res["segment_ids"] = matx.NDArray(batch_segment_ids, [], "int64")
            return res


2. Vision Modal Preprocessing
**********************************
| The code snippet below implements the Resnet Vision preprocessing with Matx, and the related vision transforms are Decode,  RandomResizedCrop, CenterCrop, RandomHorizontalFlip, Normalize, etc.

.. code-block:: python3

    from typing import List, Dict, Any
    import matx
    from matx.vision.tv_transforms import Decode, RandomHorizontalFlip, \
    RandomResizedCrop, CenterCrop, Normalize, Stack, Transpose, Compose

    class MatxImagenetVisionProcessor:
        def __init__(self, device_id: int = -1, is_train: bool = True) -> None:
            self.is_train: bool = is_train
            vision_ops: List = []
            if is_train:  # image transform for training
                vision_ops = [
                    matx.script(Decode)(to_rgb=True),
                    matx.script(RandomResizedCrop)(size=[224, 224],scale=(0.08,1.0), ratio=(0.75, 1.33)),
                    matx.script(RandomHorizontalFlip)(),
                    matx.script(Normalize)(mean=[123.675, 116.28, 103.53],
                                           std=[58.395, 57.12, 57.375]),
                    matx.script(Stack)(),
                    matx.script(Transpose)()
                ]
            else:  # image transform for evaluate
                vision_ops = [
                    matx.script(Decode)(to_rgb=True),
                    matx.script(CenterCrop)(size=[224, 224]),
                    matx.script(Normalize)(mean=[123.675, 116.28, 103.53],
                                           std=[58.395, 57.12, 57.375]),
                    matx.script(Stack)(),
                    matx.script(Transpose)()
                ]
            self.vision_op: Any = matx.script(Compose)(device_id, vision_ops)
        
        def __call__(self, images: List[bytes]) -> matx.NDArray:
            return self.vision_op(images)


3. Data Transform Pipeline
**********************************
| Finally, we combine the text and vision transform logic, and create a transform pipeline.

.. code-block:: python3

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


4. PyTorch Dataloader Demo
**********************************
| With the data transform pipeline, we can then integrate it into the data loader and further provide data for the model training process. There is nothing special in this part if you are familiar with the PyTorch DataLoader. We provide a demo below for reference, which uses fake data as the data source, and you could just replace it with your own data.

.. code-block:: python3

    from torch.utils.data import DataLoader

    class DemoDataset:
        def __init__(self, is_train=True):
            # If want to run the code, please download the demo image and vocabulary file
            # from github, or just replace them with your own ones
            f = open("demo.jpeg","rb")
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

