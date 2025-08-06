import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoTokenizer

from example_utils import fg_example


# 从本地按文件夹加载图像
class my_dataset(Dataset):
    def __init__(self, path, preprocess):
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []
        label_list = os.listdir(path)
        for label in label_list:
            image_folder = os.path.join(path, label)
            for file_names in os.listdir(image_folder):
                if file_names.endswith(("png", "jpg", "jpeg")):
                    self.image_paths.append(os.path.join(image_folder, file_names))
                    self.labels.append(label_list.index(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        image = self.preprocess(image)
        label = self.labels[item]
        return image, label


# image:tensor,size=(3,32,32)
# label:int
# tuple:[(image1,label1),(image2,label2),(image3,label3)]
# return (collate_fn([image1,image2,image3]),collate_fn([label1,label2,label3]))
# 实现文本到句子生成任务。label_list是一个大的类标签，如Imagenet，每个子任务选其中的k_subset子类，模型的提示带有三个例子
class llmCustomDataset(Dataset):
    def __init__(self, label_list, example, k_subset=10, shuffle_nums=1):
        self.data = []
        for _ in range(shuffle_nums):
            labels = random.sample(label_list, len(label_list))
            sub_tasks = [
                labels[i: i + k_subset] for i in range(0, len(labels), k_subset)
            ]
            for sub_task in sub_tasks:
                prompts = []
                for label in sub_task:
                    chosen_idx = random.sample(range(len(example)), 3)
                    current_prompt = """Given an object category, Generate one sentence about an image description: {} => {};{} => {};{} => {};{} =>""".format(
                        example[chosen_idx[0]][0],
                        example[chosen_idx[0]][1],
                        example[chosen_idx[1]][0],
                        example[chosen_idx[1]][1],
                        example[chosen_idx[2]][0],
                        example[chosen_idx[2]][1],
                        label,
                    )
                    prompts.append(current_prompt)
                self.data.append({"prompts": prompts, "labels": sub_task})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def print_example(self):
        print(self.data[0])


# pytorch 默认数据收集器
r"""
    Function that takes in a batch of data and puts the elements within the batch
    into a tensor with an additional outer dimension - batch size. The exact output type can be
    a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
    Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

    Here is the general input type (based on the type of the element within the batch) to output type mapping:

        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`

    Args:
        batch: a single batch to be collated

    Examples:
        >>> # xdoctest: +SKIP
        >>> # Example with a batch of `int`s:
        >>> default_collate([0, 1, 2, 3])
        tensor([0, 1, 2, 3])
        >>> # Example with a batch of `str`s:
        >>> default_collate(['a', 'b', 'c'])
        ['a', 'b', 'c']
        >>> # Example with `Map` inside the batch:
        >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        >>> # Example with `NamedTuple` inside the batch:
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> default_collate([Point(0, 0), Point(1, 1)])
        Point(x=tensor([0, 1]), y=tensor([0, 1]))
        >>> # Example with `Tuple` inside the batch:
        >>> default_collate([(0, 1), (2, 3)])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Example with `List` inside the batch:
        >>> default_collate([[0, 1], [2, 3]])
        [tensor([0, 2]), tensor([1, 3])]
"""

# data1={"input":torch.Tensor([1,2]),"output":3}
# data2={"input":torch.Tensor([1,3]),"output":2}
# data2={"input":torch.Tensor([1,4]),"output":1}
# batch={"input":collate_fn([torch.Tensor([1,2]),torch.Tensor([1,3]),torch.Tensor([1,4])],"output":collate_fn([3,2,1])}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
image_data = my_dataset(r"D:\dataset\cifar100_images\train", transform)
image_loader = torch.utils.data.DataLoader(image_data, batch_size=128, shuffle=True, num_workers=0)
for batch in image_loader:
    x, y = batch
    print(x.shape, y.shape)
    break

# pytorch没有很好的处理list的方法，如果list的元素个数相同通常没问题，但是个数不同就会报错
text_data = load_dataset("allenai/common_gen", split="train")
for i in range(4):
    print(text_data[i])
text_loader = torch.utils.data.DataLoader(text_data, batch_size=128, shuffle=True, num_workers=0)
try:
    for batch in text_loader:
        print(batch)
        break
except Exception as e:
    print("error:", e)

model_name = "openai/clip-vit-base-patch32"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def add_eos_to_examples(example):
    string = ",".join(example['concepts'])  # "ski,mountain,skier"
    example['input_text'] = '%s .' % string
    example['target_text'] = '%s ' % example['target']
    return example


def convert_to_features(example_batch):
    input_encodings = tokenizer(example_batch['input_text'], padding="max_length", max_length=16, truncation=True,
                                return_tensors="pt")
    target_encodings = tokenizer(example_batch['target_text'], padding="max_length", max_length=16, truncation=True,
                                 return_tensors="pt").input_ids
    labels_with_ignore_index = []
    for labels_example in target_encodings:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': labels_with_ignore_index
    }
    # print(encodings['input_ids'])

    return encodings


text_data = text_data.map(add_eos_to_examples, batched=False, remove_columns=text_data.column_names)
print(text_data[0])
text_data = text_data.map(convert_to_features, batched=True, remove_columns=text_data.column_names)
print(text_data[0])

# 自定义custom_collate_fn_list函数
def custom_collate_fn_list(batch):
    return {
        'input_ids': torch.tensor([item['input_ids'] for item in batch], dtype=torch.long),
        'attention_mask': torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long),
        'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    }

text_loader = torch.utils.data.DataLoader(
    text_data, 
    batch_size=4, 
    shuffle=False, 
    num_workers=0,
    collate_fn=custom_collate_fn_list
)

try:
    for batch in text_loader:
        print(batch)
        break
except Exception as e:
    print(e)


# collate_fn：输入：List[Sample]，其中 Sample 是 Dataset 返回的单个样本。输出：一个batch，字典或元组，tensor类型
#
# 潜在好处：input_encodings = tokenizer(example_batch['input_text'], padding="max_length", max_length=16, truncation=True,
#                                 return_tensors="pt")
# 这里我们采用的是padding将所有的句子统一长度，但是实际中可能有比16更长的句子，为此我们可以直接：
# input_encodings = tokenizer(example_batch['input_text'])
# 然后在collate_fn里自己动态padding
# def collate_fn(batch):
#     return tokenizer.pad(
#         batch,
#         padding=True,
#         return_tensors="pt",
#     )
def custom_collate_fn(batch):
    # prompts_batch = [item["prompts"] for item in batch]
    prompts_batch = []
    for item in batch:
        prompts_batch += item["prompts"]
    # labels_batch = [item["labels"] for item in batch]
    labels_batch = []
    for item in batch:
        labels_batch += item["labels"]
    return {"prompts": prompts_batch, "labels": labels_batch}


label_list = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

my_data = llmCustomDataset(label_list, fg_example, 5, 10)
my_data.print_example()
format = {"prompts": ["sentence1", "sentence2", "sentence3", "sentence4", "sentence5"],
          "labels": ["label1", "label2", "label3", "label4", "label5"]}
my_loader = torch.utils.data.DataLoader(my_data, batch_size=4, shuffle=False, num_workers=0)
try:
    for batch in my_loader:
        print(batch)
        break
except Exception as e:
    print(e)

my_loader = torch.utils.data.DataLoader(my_data, batch_size=4, shuffle=False, num_workers=0,
                                        collate_fn=custom_collate_fn)
try:
    for batch in my_loader:
        print(batch)
        break
except Exception as e:
    print(e)
