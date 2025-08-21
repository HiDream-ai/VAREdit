from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, get_h_div_w_template2indices, h_div_w_templates

import webdataset as wds
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
import numpy as np
import PIL.Image as PImage
import io


def pad_image_to_square(img):
    width, height = img.size
    max_side = max(width, height)
    new_img = PImage.new("RGB", (max_side, max_side), (0, 0, 0))
    paste_position = ((max_side - width) // 2, (max_side - height) // 2)
    new_img.paste(img, paste_position)
    return new_img


def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    # print(f'im size {im.shape}')
    return im.add(im).add_(-1)


def preprocess(sample):
    src, tgt, prompt = sample
    h, w = dynamic_resolution_h_w[h_div_w_template][PN]['pixel']
    src_img = PImage.open(io.BytesIO(src)).convert('RGB')
    tgt_img = PImage.open(io.BytesIO(tgt)).convert('RGB').resize((src_img.size))
    src_img = transform(src_img, h, w)
    tgt_img = transform(tgt_img, h, w)
    instruction = prompt.decode('utf-8')
    return src_img, tgt_img, instruction


def WDSEditDataset(
    data_path,
    buffersize,
    pn,
    batch_size,
):
    urls = []
    overall_length = 0

    with open(f"{data_path}/SEEDEdit.txt", "r") as file:
        info_file = file.readlines()
    urls_base = "SEED_EDIT_DATA_SHARD_BASE"
    data_file = []
    for item in info_file:
        file_name, length, shard_num = item.strip('\n').split('\t')
        length, shard_num = int(length), int(shard_num)
        for shard in range(shard_num):
            data_file.append(f"wds_{file_name}_{shard:=04d}.tar")
        overall_length += length
    urls += [urls_base.replace("<FILE>", file) for file in data_file]

    with open(f"{data_path}/ImgEdit.txt", "r") as file:
        info_file = file.readlines()
    urls_base = "IMG_EDIT_DATA_SHARD_BASE"
    data_file = []
    for item in info_file:
        file_name, length, shard_num = item.strip('\n').split('\t')
        length, shard_num = int(length), int(shard_num)
        for shard in range(shard_num):
            data_file.append(f"wds_{file_name}_{shard:=04d}.tar")
        overall_length += length
    urls += [urls_base.replace("<FILE>", file) for file in data_file]

    global PN
    PN = pn
    global h_div_w_template
    h_div_w_template = h_div_w_templates[np.argmin(np.abs(1.0 - h_div_w_templates))]
    dataset = wds.WebDataset(
        urls,
        nodesplitter=wds.shardlists.split_by_node,
        shardshuffle=True,
        resampled=True,
        cache_size=buffersize,
        handler=wds.handlers.warn_and_continue,
    ).with_length(overall_length).shuffle(100).to_tuple("src.jpg", "tgt.jpg", "txt").map(preprocess).batched(batch_size, partial=False).with_epoch(100000)
    return dataset