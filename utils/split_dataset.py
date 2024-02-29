import os
import random


def write_to_txt(index_list, cls_name, cls_code, image_list, dataset_txt_dir, txt_name):
    item_list = []
    for idx in index_list:
        image_path = os.path.join(cls_name, image_list[idx])
        item_list.append(image_path)

    with open(os.path.join(dataset_txt_dir, txt_name), 'a') as f:
        for item in item_list:
            msg = item + ' ' + cls_name + ' ' + str(cls_code) + '\n'
            f.write(msg)


def split_dataset(cls_dir, dataset_txt_dir, cls_name, cls_code):
    '''
    :param cls_dir: 存放'行人'或'非行人'图片的文件夹
    :return:
    '''
    image_list = os.listdir(cls_dir)
    num_images = len(image_list)
    train_num = int(num_images * 0.6)
    val_num = int(num_images * 0.2)
    test_num = int(num_images * 0.2)

    # 打乱顺序
    indices = list(range(num_images))
    random.shuffle(indices)

    train_indices = indices[: train_num]
    val_indices = indices[train_num: (train_num + val_num)]
    test_indices = indices[(train_num + val_num):]

    write_to_txt(index_list=train_indices, cls_name=cls_name, cls_code=cls_code,
                 image_list=image_list, dataset_txt_dir=dataset_txt_dir, txt_name='train.txt')

    write_to_txt(index_list=val_indices, cls_name=cls_name, cls_code=cls_code,
                 image_list=image_list, dataset_txt_dir=dataset_txt_dir, txt_name='val.txt')

    write_to_txt(index_list=test_indices, cls_name=cls_name, cls_code=cls_code,
                 image_list=image_list, dataset_txt_dir=dataset_txt_dir, txt_name='test.txt')


if __name__ == '__main__':
    base_dir = r'D:\my_phd\dataset\Stage3\D1_versionTrain'

    dataset_txt_dir = os.path.join(base_dir, 'dataset_txt')
    nonPed_dir = os.path.join(base_dir, 'nonPedestrian')
    ped_dir = os.path.join(base_dir, 'pedestrian')

    if not os.path.exists(dataset_txt_dir):
        os.mkdir(dataset_txt_dir)

    # 非行人
    split_dataset(cls_dir=nonPed_dir, dataset_txt_dir=dataset_txt_dir, cls_name='nonPedestrian', cls_code='0')
    # 行人
    split_dataset(cls_dir=ped_dir, dataset_txt_dir=dataset_txt_dir, cls_name='pedestrian', cls_code='1')


















