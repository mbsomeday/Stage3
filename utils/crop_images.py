import json
import os
import random
import cv2
from tqdm import tqdm



# # TODO 查看共有多少个图像
# city_dir = r'D:\my_phd\dataset\D1_ECPDaytime\ECP\day\labels\train'
# image_num = 0
# for city in tqdm(os.listdir(city_dir)):
#     city_path = os.path.join(city_dir, city)
#     anno_list = os.listdir(city_path)
#     image_num += len(anno_list)
#
#
# print('total:', image_num)
# print('noPed_num:', noPed_num)


def is_overlap(cropped_x0, cropped_y0, cropped_x1, cropped_y1, obj_x0, obj_y0, obj_x1, obj_y1):
    if cropped_x0 >= obj_x1 or cropped_x1 <= obj_x0:
        # print('不重叠:通过X判断')
        return False, 0
    elif cropped_y0 >= obj_y1 or cropped_y1 <= obj_y0:
        # print('不重叠:通过Y判断')
        return False, 0
    else:
        obj_width = obj_x1 - obj_x0
        obj_height = obj_y1 - obj_y0
        # obj_x0 在其中
        if cropped_x0 <= obj_x0 < cropped_x1:
            overlap_width = min((cropped_x1-obj_x0), obj_width)

        #  obj_x0 不在里面，这里包含两种情况，x0,x1都在内，和x0不在内，x1在内
        else:
            overlap_width = min((obj_x1-cropped_x0), obj_width)

        # 计算重叠的高度
        # obj_y0在内
        if cropped_y0 <= obj_y0 < cropped_y1:
            overlap_height = min((cropped_y1-obj_y0), obj_height)
        else:
            overlap_height= min((obj_y1-cropped_y0), obj_height)

        overlap_ratio = (overlap_width * overlap_height) / (obj_width * obj_height)
        # print('交叠处面积：', overlap_width * overlap_height)
        # print('obj面积：', obj_width * obj_height)
        # print('overlap_ratio:', overlap_ratio)
        # print(f'返回的：True, {overlap_ratio}')
        return True, overlap_ratio


# TODO: 行人只从pedestrian中分割，非行人从不在ped_list中的类别分割

def crop_images_D1_D3(anno_dir, image_dir, ped_save_dir, nonPed_save_dir, crop_size, crop_num):
    '''
        从原图中裁剪crop_size*crop_size大小的图片
    '''
    ped_list = ['pedestrian', 'person-group-far-away', 'rider', 'rider+vehicle-group-far-away', 'buggy-group', 'wheelchair-group']
    not_considered_tag = ['occluded>80', 'behind-glass', 'truncated>80', 'occluded>40', 'truncated>40', 'occluded>10', 'truncated>10']
    city_list = os.listdir(anno_dir)
    for idx, city in enumerate(city_list):
        city_path = os.path.join(anno_dir, city)
        anno_list = os.listdir(city_path)
        for anno in tqdm(anno_list, desc=f'City:{city}', postfix=f'{idx+1}/{len(city_list)}', colour='green'):
            anno_path = os.path.join(city_path, anno)
            with open(anno_path, 'r') as f:
                anno_data = json.load(f)
            imageheight = anno_data['imageheight']
            imagewidth = anno_data['imagewidth']
            image_path = os.path.join(image_dir, city, anno.split('.')[0] + ('.png'))
            image = cv2.imread(image_path)

            children = anno_data['children']
            cropped_ped_num = 0
            cropped_nonPed_num = 0
            loop_number = 0
            while cropped_ped_num < crop_num or cropped_nonPed_num < crop_num:
                if loop_number == 500:
                    break
                loop_number += 1
                cropped_x0 = random.randint(0, (imagewidth - crop_size))
                cropped_y0 = random.randint(0, (imageheight - crop_size))
                cropped_x1 = min(cropped_x0 + crop_size, imagewidth)
                cropped_y1 = min(cropped_y0 + crop_size, imageheight)
                # ped_exists的flag，每个json文件中只允许修改一次
                ped_exists_changed = False
                ped_exists = False

                for obj_idx, obj in enumerate(children):
                    identity = obj['identity']
                    if not ped_exists_changed and identity in ped_list:
                        ped_exists_changed = True
                        ped_exists = True
                    if identity == 'pedestrian':

                        # 每次先默认tag是ok的
                        tag_ok_flag = True

                        obj_x0 = obj['x0']
                        obj_x1 = obj['x1']
                        obj_y0 = obj['y0']
                        obj_y1 = obj['y1']
                        tags = obj['tags']

                        obj_width = obj_x1 - obj_x0
                        obj_height = obj_y1 - obj_y0

                        for tag in tags:
                            if tag in not_considered_tag:
                                tag_ok_flag = False

                        if obj_width > 80 and obj_height > 100 and tag_ok_flag:
                            has_ped, overlap_ratio = is_overlap(cropped_x0, cropped_y0, cropped_x1, cropped_y1, obj_x0,
                                                                obj_y0, obj_x1, obj_y1)

                            if cropped_ped_num < crop_num and has_ped and overlap_ratio >= 0.8:
                                cropped_ped_num += 1
                                cropped = image[cropped_y0:cropped_y1, cropped_x0:cropped_x1]  # 裁剪坐标为[y0:y1, x0:x1]
                                cropped_name = anno.split('.')[0] + '_' + str(cropped_ped_num) + '.jpg'
                                cropped_path = os.path.join(ped_save_dir, cropped_name)
                                cv2.imwrite(cropped_path, cropped)

                                # # 分别是左上顶点，右下顶点，绿色
                                # cv2.rectangle(image, (obj_x0, obj_y0), (obj_x1, obj_y1), (0, 255, 0), 2)
                                # cv2.rectangle(image, (cropped_x0, cropped_y0), (cropped_x1, cropped_y1), (255, 0, 0), 2)
                                # cv2.imshow('pedestrian', image)
                                # cv2.waitKey()

                                break

                # 没有和行人有交集，可作为nonPed crop
                if cropped_nonPed_num < crop_num and not ped_exists:
                    cropped_nonPed_num += 1
                    cropped = image[cropped_y0:cropped_y1, cropped_x0:cropped_x1]  # 裁剪坐标为[y0:y1, x0:x1]
                    cropped_name = anno.split('.')[0] + '_' + str(cropped_nonPed_num) + '.jpg'

                    # image2 = cv2.imread(img_path)
                    # cv2.rectangle(image2, (cropped_x0, cropped_y0), (cropped_x1, cropped_y1), (255, 0, 0), 2)
                    # cv2.imshow('non pedestrian', image2)
                    # cv2.waitKey()
                    cropped_path = os.path.join(nonPed_save_dir, cropped_name)
                    cv2.imwrite(cropped_path, cropped)
            # break
        # break


def crop_images_D2(anno_dir, image_dir, ped_save_dir, nonPed_save_dir, crop_size, crop_num):
    '''
        从原图中裁剪crop_size*crop_size大小的图片
    '''
    ped_list = ['sitting person', 'rider', 'pedestrian', 'person group']
    city_list = os.listdir(anno_dir)
    for city in tqdm(city_list):
        city_path = os.path.join(anno_dir, city)
        anno_list = os.listdir(city_path)
        for anno in anno_list:
            anno_path = os.path.join(city_path, anno)
            with open(anno_path, 'r') as f:
                anno_data = json.load(f)
            imageheight = anno_data['imgHeight']
            imagewidth = anno_data['imgWidth']
            objects = anno_data['objects']

            anno_contents = anno.split('_')
            anno_part = anno_contents[0] + '_' + anno_contents[1] + '_' + anno_contents[2]
            image_path = os.path.join(image_dir, city, anno_part + ('_leftImg8bit_blurred.jpg'))
            image = cv2.imread(image_path)

            cropped_ped_num = 0
            cropped_nonPed_num = 0
            loop_number = 0

            while cropped_ped_num < crop_num or cropped_nonPed_num < crop_num:
                if loop_number == 500:
                    break
                # 用于判断当前裁剪的坐标是否和行人类有交集
                not_has_ped = True
                loop_number += 1
                cropped_x0 = random.randint(0, (imagewidth - crop_size))
                cropped_y0 = random.randint(0, (imageheight - crop_size))
                cropped_x1 = min(cropped_x0 + crop_size, imagewidth)
                cropped_y1 = min(cropped_y0 + crop_size, imageheight)

                for obj_idx, obj in enumerate(objects):
                    label = obj['label']
                    bbox = obj['bbox']
                    obj_x0 = bbox[0]
                    obj_y0 = bbox[1]
                    obj_width = bbox[2]
                    obj_height = bbox[3]

                    obj_x1 = obj_x0 + obj_width
                    obj_y1 = obj_y0 + obj_height

                    has_ped, overlap_ratio = is_overlap(cropped_x0, cropped_y0, cropped_x1, cropped_y1, obj_x0, obj_y0, obj_x1, obj_y1)
                    if label in ped_list and overlap_ratio > 0:
                        not_has_ped = False

                    if label == 'pedestrian' and cropped_ped_num < crop_num and has_ped and overlap_ratio >= 0.9 and obj_width > 80 and obj_height > 100:
                        cropped_ped_num += 1
                        cropped = image[cropped_y0:cropped_y1, cropped_x0:cropped_x1]  # 裁剪坐标为[y0:y1, x0:x1]
                        cropped_name = anno.split('.')[0] + '_' + str(cropped_ped_num) + '.jpg'
                        cropped_path = os.path.join(ped_save_dir, cropped_name)
                        cv2.imwrite(cropped_path, cropped)

                        # # 分别是左上顶点，右下顶点，绿色
                        # cv2.rectangle(image, (obj_x0, obj_y0), (obj_x1, obj_y1), (0, 255, 0), 2)
                        # cv2.rectangle(image, (cropped_x0, cropped_y0), (cropped_x1, cropped_y1), (255, 0, 0), 2)
                        # cv2.imshow('pedestrian', image)
                        # cv2.waitKey()

                        break

                # 没有和行人有交集，可作为nonPed crop
                if cropped_nonPed_num < crop_num:
                    if len(objects) == 0 or not_has_ped:
                        cropped_nonPed_num += 1
                        cropped = image[cropped_y0:cropped_y1, cropped_x0:cropped_x1]  # 裁剪坐标为[y0:y1, x0:x1]
                        cropped_name = anno.split('.')[0] + '_' + str(cropped_nonPed_num) + '.jpg'

                        # image2 = cv2.imread(img_path)
                        # cv2.rectangle(image2, (cropped_x0, cropped_y0), (cropped_x1, cropped_y1), (255, 0, 0), 2)
                        # cv2.imshow('non pedestrian', image2)
                        # cv2.waitKey()
                        cropped_path = os.path.join(nonPed_save_dir, cropped_name)
                        cv2.imwrite(cropped_path, cropped)

            # break
        # break


if __name__ == '__main__':

    anno_dir = r'D:\my_phd\dataset\D1_ECPDaytime\ECP\day\labels\train'
    image_dir = r'D:\my_phd\dataset\D1_ECPDaytime\ECP\day\img\train'

    save_base_dir = r'D:\my_phd\dataset\Stage3\D1_versionTrain'
    ped_save_dir = os.path.join(save_base_dir, 'pedestrian')
    nonPed_save_dir = os.path.join(save_base_dir, 'nonPedestrian')
    # 创建存储的文件夹
    if not os.path.exists(ped_save_dir):
        os.mkdir(ped_save_dir)
    if not os.path.exists(nonPed_save_dir):
        os.mkdir(nonPed_save_dir)
    crop_size = 224
    crop_images_D1_D3(anno_dir=anno_dir,
                image_dir=image_dir,
                ped_save_dir=ped_save_dir,
                nonPed_save_dir=nonPed_save_dir,
                crop_size=crop_size,
                crop_num=1
                )



























