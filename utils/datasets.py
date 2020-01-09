import torch.utils.data as data
from torchvision.transforms import *
import os
from os import listdir
from os.path import join
from PIL import Image
from settings import Settings
from utils.google_drive import download_file_from_google_drive
from utils.google_drive import download_from_url
from tqdm import tqdm
import zipfile
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def calculate_valid_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)


class TrainDataset(data.Dataset):
    def __init__(self, settings: Settings):
        super(TrainDataset, self).__init__()
        
        self.settings = settings
        self.dataset_path = os.path.join(self.settings.dataset_root, "291")

        # Validate Dataset
        if not os.path.exists(self.settings.dataset_root):
            os.mkdir(self.settings.dataset_root)

        if not os.path.exists(self.dataset_path):
            link = self.settings.dataset_info['291']['link']
            print("Downloading '{dataset_id}' dataset from cloud... id:[{link}]".format(dataset_id='291', link=link))
            comp_file_name = self.download_dataset(dataset_path=self.settings.dataset_root, link=link)

            print("Unzipping...".format(dataset_id='291'))
            with zipfile.ZipFile(comp_file_name, 'r') as zip_ref:
                zip_ref.extractall(self.settings.dataset_root)

            if os.path.exists(self.dataset_path):
                print("Successfully downloaded '{dataset_id}' dataset @ [{dataset_path}]".format(dataset_id='291', dataset_path=self.dataset_path))
                os.remove(comp_file_name)
            else:
                raise Exception('dataset_path does not match downloaded dataset.. please check settings')
        print("Successfully loaded.")
        image_dir = self.dataset_path
        print(image_dir)
        self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.is_gray = self.settings.dataset_info['291']['is_gray']
        self.random_scale = self.settings.dataset_info['291']['random_scale']
        self.crop_size = self.settings.dataset_info['291']['crop_size']
        self.rotate = self.settings.dataset_info['291']['rotate']
        self.fliplr = self.settings.dataset_info['291']['fliplr']
        self.fliptb = self.settings.dataset_info['291']['fliptb']
        self.scale_factor = self.settings.dataset_info['291']['scale_factor']
        self.random_scale_factor = self.settings.dataset_info['291']['random_scale_factor']
        

    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index])

        if self.random_scale_factor:
            random_scale_factor = random.randint(2, 4)

        # determine valid HR image size with scale factor
        self.crop_size = calculate_valid_crop_size(self.crop_size, self.scale_factor)
        hr_img_w = self.crop_size
        hr_img_h = self.crop_size

        # determine LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # random scaling between [0.5, 1.0]
        if self.random_scale:
            eps = 1e-3
            ratio = random.randint(5, 10) * 0.1
            if hr_img_w * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_w + eps
            if hr_img_h * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_h + eps

            scale_w = int(hr_img_w * ratio)
            scale_h = int(hr_img_h * ratio)
            transform = Resize((scale_w, scale_h), interpolation=Image.BICUBIC)
            img = transform(img)

        # random crop
        transform = RandomCrop(self.crop_size)
        img = transform(img)

        # random rotation between [90, 180, 270] degrees
        if self.rotate:
            rv = random.randint(1, 3)
            img = img.rotate(90 * rv, expand=True)

        # random horizontal flip
        if self.fliplr:
            transform = RandomHorizontalFlip()
            img = transform(img)

        # random vertical flip
        if self.fliptb:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # only Y-channel is super-resolved
        if self.is_gray:
            img = img.convert('YCbCr')
            # img, _, _ = img.split()

        # hr_img HR image
        hr_transform = Compose([Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        hr_img = hr_transform(img)

        # lr_img LR image
        lr_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        lr_img = lr_transform(img)

        # Bicubic interpolated image
        bc_transform = Compose([ToPILImage(), Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        bc_img = bc_transform(lr_img)

        return bc_img, hr_img

    def __len__(self):
        return len(self.image_filenames)

    def download_dataset(self, dataset_path: str, link: str):
        file_name = os.path.join(dataset_path, 'compressed.zip')
        download_file_from_google_drive(id=link, destination=file_name)
        return file_name


class TestDataset(data.Dataset):
    def __init__(self, settings:Settings):
        super(TestDataset, self).__init__()

        self.settings = settings
        self.dataset_path = os.path.join(self.settings.dataset_root, "SR_testing_datasets")

        # Validate Dataset
        if not os.path.exists(self.settings.dataset_root):
            os.mkdir(self.settings.dataset_root)

        if not os.path.exists(self.dataset_path):
            link = self.settings.dataset_info['SR_testing_datasets']['link']
            print("Downloading '{dataset_id}' dataset from cloud... id:[{link}]".format(dataset_id='SR_testing_datasets', link=link))
            comp_file_name = self.download_dataset(dataset_path=self.dataset_path, link=link)

            print("Unzipping...".format(dataset_id='SR_testing_datasets'))
            with zipfile.ZipFile(comp_file_name, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.settings.dataset_root, 'SR_testing_datasets'))

            if os.path.exists(self.dataset_path):
                print("Successfully downloaded '{dataset_id}' dataset @ [{dataset_path}]".format(dataset_id='SR_testing_datasets', dataset_path=self.dataset_path))
                os.remove(comp_file_name)
            else:
                raise Exception('dataset_path does not match downloaded dataset.. please check settings')
        print("Successfully loaded.")
        image_dir = os.path.join(self.dataset_path, settings.dataset_info['SR_testing_datasets']['id'])

        self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.is_gray = self.settings.dataset_info['SR_testing_datasets']['is_gray']
        self.scale_factor = self.settings.dataset_info['SR_testing_datasets']['scale_factor']

    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index])

        # original HR image size
        w = img.size[0]
        h = img.size[1]

        # determine valid HR image size with scale factor
        hr_img_w = calculate_valid_crop_size(w, self.scale_factor)
        hr_img_h = calculate_valid_crop_size(h, self.scale_factor)

        # determine lr_img LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # only Y-channel is super-resolved
        if self.is_gray:
            img = img.convert('YCbCr')
            # img, _, _ = lr_img.split()

        # hr_img HR image
        hr_transform = Compose([Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        hr_img = hr_transform(img)

        # lr_img LR image
        lr_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        lr_img = lr_transform(img)

        # Bicubic interpolated image
        bc_transform = Compose([ToPILImage(), Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        bc_img = bc_transform(lr_img)

        return lr_img, hr_img, bc_img

    def __len__(self):
        return len(self.image_filenames)

    def download_dataset(self, dataset_path: str, link: str):
        file_name = os.path.join(dataset_path, 'compressed.zip')
        download_from_url(url=link, destination=file_name)
        return file_name