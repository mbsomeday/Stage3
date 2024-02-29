import os
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm


def save_image(train_save_path, org_name, type_info, image):
    save_name = org_name.split('.')[0] + '_' + str(type_info) + '.jpg'
    save_path = os.path.join(train_save_path, save_name)
    image.save(save_path)
    return save_path

class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=1):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: # 按概率进行
            # 把img转化成ndarry的形式
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # 原始图像的概率（这里为0.9）
            signal_pct = self.snr
            # 噪声概率共0.1
            noise_pct = (1 - self.snr)
            # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            # 将mask按列复制c遍
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255 # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB') # 转化为PIL的形式
        else:
            return img
#添加高斯噪声
class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img


base_dir = r'D:\my_phd\dataset\Stage3\D1_versionTrain'

train_txt = os.path.join(base_dir, 'dataset_txt', 'train.txt')
train_save_path = os.path.join(base_dir, 'augmented_train')

if not os.path.exists(train_save_path):
    os.mkdir(train_save_path)

new_train_list = []

with open(train_txt, 'r') as f:
    data = f.readlines()

for item in tqdm(data):
    item = item.strip()
    image_path = item.split(' ')[0]
    image_path = os.path.join(base_dir, image_path)

    image = Image.open(image_path)

    # 水平翻转
    hor_img = transforms.RandomHorizontalFlip(p=1)(image)

    # # 随机在（-15， 15）度旋转
    # rot_img = transforms.RandomRotation(15)(image)
    #
    # # 随机从0~2之间的亮度变化
    # bright_img = transforms.ColorJitter(brightness=0.8)(image)
    #
    # # 随机从0~2之间的对比度
    # contrast_img = transforms.ColorJitter(contrast=0.8)(image)

    # # 饱和度
    # saturation_img = transforms.ColorJitter(saturation=1)(image)

    # 高斯模糊
    gaussian_img = AddGaussianNoise(mean=random.uniform(0.5, 1.5), variance=0.5, amplitude=random.uniform(0, 35))(image)

    # 椒盐噪声
    pepNoise_img = AddPepperNoise(0.99, 1.0)(image)


    org_name = item.split(' ')[0].split('\\')[-1]
    # image_list = [image, hor_img, rot_img, bright_img, contrast_img, saturation_img, gaussian_img, pepNoise_img]
    # type_list = ['org', 'hor', 'rota', 'bright', 'contrast', 'sat', 'gaussian', 'pepnoise']

    image_list = [image, hor_img, gaussian_img, pepNoise_img]
    type_list = ['org', 'hor', 'gaussian', 'pepnoise']

    for i in range(len(image_list)):
        save_image(train_save_path=train_save_path, org_name=org_name,
                   type_info=type_list[i], image=image_list[i]
                   )

    for type_info in type_list:
        content = item.split(' ')
        old_name = content[0].split('.')[0].split('\\')[-1]
        new_name = old_name + '_' + str(type_info) + '.jpg'
        new_path = os.path.join('augmented_train', new_name)
        msg = new_path + ' ' + content[1] + ' ' + content[2] + '\n'
        new_train_list.append(msg)
    # break


with open(os.path.join(base_dir, 'dataset_txt', 'augmentation_train.txt'), 'w') as f:
    for item in new_train_list:
        f.write(item)































