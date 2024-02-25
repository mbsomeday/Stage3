import json
import os
import random
import cv2
from tqdm import tqdm



# TODO 查看共有多少个图像
# city_dir = r'D:\my_phd\dataset\D3_ECPNight\ECP\night\labels'
# image_num = 0
# ped_list = ['pedestrian', 'person-group-far-away', 'scooter-group', 'rider','rider+vehicle-group-far-away', 'buggy-group', 'motorbike-group', 'wheelchair-group']
# no_ped = True
# noPed_num = 0
# for city in tqdm(os.listdir(city_dir)):
#     city_path = os.path.join(city_dir, city)
#     anno_list = os.listdir(city_path)
#     for anno in anno_list:
#         image_num += 1
#         with open(os.path.join(city_path, anno)) as f:
#             anno_data = json.load(f)
#             children = anno_data['children']
#         for obj in children:
#             identity = obj['identity']
#             if identity in ped_list:
#                 no_ped = False
#         if no_ped:
#             noPed_num += 1
#
# print('total:', image_num)
# print('noPed_num:', noPed_num)


def is_x_overlap(cropped_x0, cropped_x1, obj_x0, obj_x1):
    '''
        判断crop 的bbox 是否与obj bbox的x有交集
    '''
    if cropped_x0 < obj_x0 < cropped_x1 or cropped_x0 < obj_x1 < cropped_x1:
        print('X重叠')
        return True

def is_y_overlap(cropped_y0, cropped_y1, obj_y0, obj_y1):
    if cropped_y0 < obj_y0 < cropped_y1 or cropped_y0 < obj_y1 < cropped_y1:
        print('Y重叠')
        return True

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

def crop_images(anno_dir, image_dir, ped_save_dir, nonPed_save_dir):
    '''
        从原图中裁剪512*512大小的图片
    '''
    ped_list = ['pedestrian', 'scooter-group', 'rider', 'rider+vehicle-group-far-away', 'buggy-group',
                'motorbike-group', 'wheelchair-group']
    city_list = os.listdir(anno_dir)
    for city in tqdm(city_list):
        city_path = os.path.join(anno_dir, city)
        anno_list = os.listdir(city_path)
        for anno in anno_list:
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
            while cropped_ped_num < 2 or cropped_nonPed_num < 2:
                if loop_number == 500:
                    break
                loop_number += 1
                cropped_x0 = random.randint(0, (imagewidth - 512))
                cropped_y0 = random.randint(0, (imageheight - 512))
                cropped_x1 = min(cropped_x0 + 512, imagewidth)
                cropped_y1 = min(cropped_y0 + 512, imageheight)
                # ped_exists的flag，每个json文件中只允许修改一次
                ped_exists_changed = False
                ped_exists = False

                for obj_idx, obj in enumerate(children):
                    identity = obj['identity']
                    if identity in ped_list:
                        obj_x0 = obj['x0']
                        obj_x1 = obj['x1']
                        obj_y0 = obj['y0']
                        obj_y1 = obj['y1']

                        has_ped, overlap_ratio = is_overlap(cropped_x0, cropped_y0, cropped_x1, cropped_y1, obj_x0,
                                                            obj_y0, obj_x1, obj_y1)
                        if not ped_exists_changed and has_ped:
                            ped_exists = has_ped
                        if cropped_ped_num < 2 and has_ped and overlap_ratio >= 0.8:
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
                if cropped_nonPed_num < 2 and not ped_exists:
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






























