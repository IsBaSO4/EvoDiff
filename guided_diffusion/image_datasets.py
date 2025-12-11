import math
import os
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from PIL import Image
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
from torchsampler import ImbalancedDatasetSampler


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    imablancedsample=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    # if class_cond:
    #     try:
    #         # Assume classes are the first part of the filename,
    #         # before an underscore.
    #         class_names = [path.split("/")[-2] for path in all_files]
    #         # 获取唯一类别并排序
    #         unique_classes = sorted(set(class_names))
    #         # class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #         sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #         classes = [sorted_classes[x] for x in class_names]
            
    #                     # 打印类别信息（调试用）
    #         print(f"找到的类别: {unique_classes}")
    #         print(f"类别数量: {len(unique_classes)}")
    #         print(f"样本数量: {len(classes)}")
            
    #         # 验证类别标签
    #         assert min(classes) >= 0, "存在负数类别标签"
    #         assert max(classes) < len(unique_classes), "类别标签超出范围"
            
    #     except Exception as e:
    #         raise ValueError(f"处理类别标签时出错: {str(e)}")
    
    if class_cond:
        try:
            # 获取所有文件的类别名称
            class_names = [path.split("/")[-2] for path in all_files]
            
            # 过滤掉不需要的类别
            valid_files = []
            valid_class_names = []
            for f, c in zip(all_files, class_names):
                if c != "unlabel_b_t_yilaoshi":
                    valid_files.append(f)
                    valid_class_names.append(c)
            
            # 更新文件列表和类别名称
            all_files = valid_files
            class_names = valid_class_names

            # 创建类别映射（按数字排序）
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]

            # 打印调试信息
            print("\n数据集信息:")
            print(f"有效类别: {sorted(set(class_names))}")
            print(f"类别数量: {len(sorted_classes)}")
            print(f"总样本数: {len(classes)}")

            # 打印每个类别的样本数量
            class_counts = {}
            for c, label in zip(class_names, classes):
                class_counts[c] = class_counts.get(c, 0) + 1
            print("\n每个类别的样本数量:")
            for c in sorted(class_counts.keys()):
                print(f"类别 {c}: {class_counts[c]} 个样本")

            # 验证类别标签
            assert min(classes) >= 0, "存在负数类别标签"
            assert max(classes) < len(sorted_classes), "类别标签超出范围"

        except Exception as e:
            raise ValueError(f"处理类别标签时出错: {str(e)}")
        
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )

    if imablancedsample:
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=1,  # shuffle=False,drop_last=True,
            sampler=ImbalancedDatasetSampler(dataset, classes))
    else:
        if deterministic:
            loader = DataLoader(
                dataset, batch_size=batch_size, num_workers=1, shuffle=False,drop_last=True,
            )
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size, num_workers=1, shuffle=True,drop_last=True,
            )

    while True:
        yield from loader
        
# def load_data(
#     *,
#     data_dir,
#     batch_size,
#     image_size,
#     class_cond=False,
#     deterministic=False,
#     random_crop=False,
#     random_flip=True,
#     imablancedsample=True,
# ):
#     """
#     For a dataset, create a generator over (images, kwargs) pairs.

#     Each images is an NCHW float tensor, and the kwargs dict contains zero or
#     more keys, each of which map to a batched Tensor of their own.
#     The kwargs dict can be used for class labels, in which case the key is "y"
#     and the values are integer tensors of class labels.

#     :param data_dir: a dataset directory.
#     :param batch_size: the batch size of each returned pair.
#     :param image_size: the size to which images are resized.
#     :param class_cond: if True, include a "y" key in returned dicts for class
#                        label. If classes are not available and this is true, an
#                        exception will be raised.
#     :param deterministic: if True, yield results in a deterministic order.
#     :param random_crop: if True, randomly crop the images for augmentation.
#     :param random_flip: if True, randomly flip the images for augmentation.
#     """
#     if not data_dir:
#         raise ValueError("unspecified data directory")
#     all_files = _list_image_files_recursively(data_dir)
#     classes = None
#     if class_cond:
#         # Assume classes are the first part of the filename,
#         # before an underscore.
#         class_names = [path.split("/")[-2] for path in all_files]
#        # print("classssssss", class_names)
#         # class_names = [bf.basename(path).split("_")[0] for path in all_files]
#         sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
#         classes = [sorted_classes[x] for x in class_names]
#         print("After classssssss", classes)   #0,1,2,3,4
    
#     dataset = ImageDataset(
#         image_size,
#         all_files,
#         classes=classes,
#         shard=MPI.COMM_WORLD.Get_rank(),
#         num_shards=MPI.COMM_WORLD.Get_size(),
#         random_crop=random_crop,
#         random_flip=random_flip,
#     )

#     if imablancedsample:
#         loader = DataLoader(
#             dataset, batch_size=batch_size, num_workers=1,  # shuffle=False,drop_last=True,
#             sampler=ImbalancedDatasetSampler(dataset, classes))
#     else:
#         if deterministic:
#             loader = DataLoader(
#                 dataset, batch_size=batch_size, num_workers=1, shuffle=False,drop_last=True,
#             )
#         else:
#             loader = DataLoader(
#                 dataset, batch_size=batch_size, num_workers=1, shuffle=True,drop_last=True,
#             )

#     while True:
#         yield from loader
        

#-------为了加载/home/data/duanyaofei/classification-0328/unlabel_b_t进行训练---------
def load_data_0402(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    imablancedsample=True,
):
    """
    为数据集创建一个生成器，返回 (images, kwargs) 对。
    
    Args:
        data_dir: 数据集目录
        batch_size: 每批次返回的样本数
        image_size: 图像调整后的大小
        class_cond: 是否包含类别标签
        deterministic: 是否按确定性顺序产生结果
        random_crop: 是否随机裁剪图像进行数据增强
        random_flip: 是否随机翻转图像进行数据增强
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files = _list_image_files_recursively_0402(data_dir)
    
    classes = None
    if class_cond:
        # 从文件路径中提取类别（父文件夹名称）
        class_names = [path.split("/")[-2] for path in all_files]
        # 确保类别是0-5的数字
        sorted_classes = {x: int(x) for x in sorted(set(class_names))}
        classes = [sorted_classes[x] for x in class_names]

    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )

    if imablancedsample:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=1,
            sampler=ImbalancedDatasetSampler(dataset, classes)
        )
    else:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=1, 
            shuffle=not deterministic,
            drop_last=True,
        )

    while True:
        yield from loader



def load_source_data_for_domain_translation(
        *,
        batch_size,
        image_size,
        data_dir="./experiments/imagenet",
        class_cond=True
):
    """
    This function is new in DDIBs: loads the source dataset for translation.
    For the dataset, create a generator over (images, kwargs) pairs.
    No image cropping, flipping or shuffling.

    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    image_names = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [path.split("/")[-2] for path in all_files]
        # class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
        image_names = [path.split("/")[-1] for path in all_files]
    dataset = ImageDataset(
        image_size,
        all_files,
        random_flip=False,
        classes=classes,
        filepaths=image_names,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    yield from loader


def list_image_files(data_dir):
    """List images files in the directory (not recursively)."""
    files = sorted(bf.listdir(data_dir))
    results = []
    for entry in files:
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
    return results


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


# def _list_image_files_recursively_0402(data_dir):
#     """
#     递归获取指定目录下的所有图像文件
#     """
#     results = []
#     for entry in sorted(bf.listdir(data_dir)):
#         full_path = bf.join(data_dir, entry)
#         if bf.isdir(full_path):
#             # 只处理数字命名的文件夹(0-5)
#             if entry.isdigit() and 0 <= int(entry) <= 5:
#                 # 遍历子文件夹中的图片
#                 for sub_entry in sorted(bf.listdir(full_path)):
#                     sub_full_path = bf.join(full_path, sub_entry)
#                     if bf.isfile(sub_full_path):
#                         ext = sub_entry.split(".")[-1]
#                         if ext.lower() == "png":
#                             results.append(sub_full_path)
#     return results

def _list_image_files_recursively_0402(data_dir):
    """递归获取所有图片文件路径"""
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        # 检查是否为目录
        try:
            # 如果是数字文件夹（0-5）
            if entry.isdigit() and 0 <= int(entry) <= 5:
                # 遍历子文件夹中的文件
                for sub_entry in sorted(bf.listdir(full_path)):
                    sub_full_path = bf.join(full_path, sub_entry)
                    # 检查文件扩展名
                    if "." in sub_entry:
                        ext = sub_entry.split(".")[-1]
                        if ext.lower() == "png":
                            results.append(sub_full_path)
        except:
            # 如果不是目录，跳过
            continue
    return results




#-------为了加载/home/data/duanyaofei/classification-0328/labeled_b_t进行训练---------

def extract_combined_class(path):
    """
    从文件路径中提取组合类别名称
    例如：从 "dataset0/0/image001.jpg" 提取得到 "dataset0_0"  Appendix_0,Appendix_1,Breast_0.....
    """
    try:
        parts = path.split("/")
        # 组合倒数第三个和倒数第二个元素
        dataset_name = parts[-3]  # 例如 "dataset0"
     #   print('dataset_name',dataset_name)
        class_name = parts[-2]    # 例如 "0"
      #  print('class_name',class_name)
        return f"{dataset_name}_{class_name}"
    except IndexError:
        print(f"警告: 无法从路径提取类别: {path}")
        return None

def load_data_0407(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    imablancedsample=True,
):
    """
    为数据集创建一个生成器，返回 (images, kwargs) 对。
    
    Args:
        data_dir: 数据集目录
        batch_size: 每批次返回的样本数
        image_size: 图像调整后的大小
        class_cond: 是否包含类别标签
        deterministic: 是否按确定性顺序产生结果
        random_crop: 是否随机裁剪图像进行数据增强
        random_flip: 是否随机翻转图像进行数据增强
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files = _list_image_files_recursively(data_dir)
    
    classes = None
    if class_cond:
        # # 从文件路径中提取类别（父文件夹名称） #/home/data/duanyaofei/classification-0328/label_b_t_0407/Breast/0/000004.png
        # class_names = [path.split("/")[-2] for path in all_files]
        # # 确保类别是0-5的数字
        # sorted_classes = {x: int(x) for x in sorted(set(class_names))}
        # classes = [sorted_classes[x] for x in class_names]

        # 提取组合类别名称
        class_names = [extract_combined_class(path) for path in all_files]
      #  print(class_names)
        # 获取唯一类别并排序
        unique_classes = sorted(set(class_names))
        
        # 创建类别到数字的映射
        class_to_num = {class_name: i for i, class_name in enumerate(unique_classes)}
        
        # 转换为数值类别
        classes = [class_to_num[class_name] for class_name in class_names]
        
                # 打印类别映射关系
        # print("\n=== 类别映射关系 ===")
        # for class_name, num in class_to_num.items():
        #     print(f"类别: {class_name} -> 数值: {num}")
        # print("==================\n")
        '''
        === 类别映射关系 ===
        类别: Appendix_0 -> 数值: 0
        类别: Appendix_1 -> 数值: 1
        类别: Breast_0 -> 数值: 2
        类别: Breast_1 -> 数值: 3
        类别: Breast_2 -> 数值: 4
        类别: CUBS_0 -> 数值: 5
        类别: CUBS_1 -> 数值: 6
        类别: Fatty-Liver_0 -> 数值: 7
        类别: Fatty-Liver_1 -> 数值: 8
        类别: MMOTU_0 -> 数值: 9
        类别: MMOTU_1 -> 数值: 10
        类别: TN3K_0 -> 数值: 11
        类别: TN3K_1 -> 数值: 12
        ==================
        '''

    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )

    if imablancedsample:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=1,
            sampler=ImbalancedDatasetSampler(dataset, classes)
        )
    else:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=1, 
            shuffle=not deterministic,
            drop_last=True,
        )

    while True:
        yield from loader
        
        


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        filepaths=None
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.filepaths = filepaths

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip:
            flag_aug = random.randint(0, 1)
            arr = data_augmentation(arr, flag_aug)

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if self.filepaths is not None:
            out_dict["filepath"] = self.filepaths[idx]
        return np.transpose(arr, [2, 0, 1]), out_dict

    # def get_labels(self):
    #     return self.local_classes


def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        out = np.fliplr(image)
    return out

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def get_image_filenames_for_label(label):
    """
    Returns the validation files for images with the given label. This is a utility
    function for ImageNet translation experiments.
    :param label: an integer in 0-1000
    """
    # First, retrieve the synset word corresponding to the given label
    base_dir = os.getcwd()
    synsets_filepath = os.path.join(base_dir, "evaluations", "synset_words.txt")
    synsets = [line.split()[0] for line in open(synsets_filepath).readlines()]
    synset_word_for_label = synsets[label]

    # Next, build the synset to ID mapping
    synset_mapping_filepath = os.path.join(base_dir, "evaluations", "map_clsloc.txt")
    synset_to_id = dict()
    with open(synset_mapping_filepath) as file:
        for line in file:
            synset, class_id, _ = line.split()
            synset_to_id[synset.strip()] = int(class_id.strip())
    true_label = synset_to_id[synset_word_for_label]

    # Finally, return image files corresponding to the true label
    validation_ground_truth_filepath = os.path.join(base_dir, "evaluations", "ILSVRC2012_validation_ground_truth.txt")
    source_data_labels = [int(line.strip()) for line in open(validation_ground_truth_filepath).readlines()]
    image_indexes = [i + 1 for i in range(len(source_data_labels)) if true_label == source_data_labels[i]]
    output = [f"ILSVRC2012_val_{str(i).zfill(8)}.JPEG" for i in image_indexes]
    return output


# def quick_test_loader(data_dir):
#     """
#     快速测试数据加载器
#     """
#     print("=== 开始测试数据加载器 ===")
    
#     # 1. 加载数据
#     loader = load_data_0402(
#         data_dir=data_dir,
#         batch_size=4,
#         image_size=256,
#         class_cond=True
#     )
    
#     # 2. 获取一个批次
#     try:
#         batch = next(iter(loader))
#         images, labels = batch
        
#         # 3. 输出基本信息
#         print("\n基本信息:")
#         print(f"图像形状: {images.shape}")
#       #  print(f"标签形状: {labels.shape}")
#         print(f"标签内容: {labels}")
        
#         # 4. 检查图像数值范围
#         print(f"\n图像数值范围:")
#         print(f"最小值: {images.min():.2f}")
#         print(f"最大值: {images.max():.2f}")
        
#         print("\n=== 测试完成 ===")
#         print("数据加载正常！")
        
#     except Exception as e:
#         print(f"\n测试失败！错误信息：")
#         print(str(e))


# 使用示例
# data_generator = load_data_0407(
#     data_dir="/home/data/duanyaofei/classification-0328/label_b_t_0407",
#     batch_size=32,
#     image_size=256,
#     class_cond=True,
#     random_flip=True,
# )



# # # 检查数据集结构
# def check_dataset_structure(data_dir):
#     """检查数据集目录结构"""
#     class_counts = {}
#     for root, dirs, files in os.walk(data_dir):
#         if files:  # 只统计包含文件的目录
#             class_name = os.path.basename(root)
#             class_counts[class_name] = len(files)
    
#     print("数据集结构:")
#     for class_name, count in class_counts.items():
#         print(f"类别 {class_name}: {count} 个样本")
    
#     return class_counts

# data_dir="/home/data/duanyaofei/classification-0328/label_b_t_0407"
# #data_dir="/home/data/duanyaofei/MGDM-main/classification-0306_re/train_b_t"
# # 在加载数据之前调用
# check_dataset_structure(data_dir)

# # if __name__ == "__main__":
#     # 替换为你的数据目录路径
#     DATA_DIR =  "/home/data/duanyaofei/classification-0328/unlabel_b_t_yilaoshi"
#     quick_test_loader(DATA_DIR)
    
    

# # # # 测试代码
# # data_dir = "/home/data/duanyaofei/classification-0328/unlabel_b_t"
# # batch_size = 4
# # image_size = 256
# # class_cond = True
# # random_flip = True
# # imablancedsample = False
# # all_files, classes = load_data_0402(data_dir=data_dir,
# #                                     batch_size=batch_size,
# #                                     image_size=image_size,
# #                                     class_cond=class_cond,
# #                                     random_flip=random_flip,
# #                                     imablancedsample=imablancedsample,)

# # 打印前几个样本的信息
# for i in range(min(5, len(all_files))):
#     print(f"文件: {all_files[i]}")
#     print(f"类别索引: {classes[i]}")