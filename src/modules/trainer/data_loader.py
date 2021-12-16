import math
import os
import random
import re
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms

import time
from PIL import Image
import math
import cv2
import pickle
import math
import traceback, itertools
from src.io import load_char_label_dicts
from src.modules.trainer import train_util as util
from torchvision.io.image import ImageReadMode

def get_geomap(scoremap, rectangles, instance_mask, scale=0.25):
    scoremap = cv2.resize(scoremap, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    instance_mask = cv2.resize(instance_mask, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = scoremap.shape
    geo_map = np.zeros((h, w, 5), dtype = np.float32)
    poly_mask = scoremap * instance_mask
    #poly_mask = instance_mask
    rectangles = rectangles * scale
    for i in range(len(rectangles)):
        poly = rectangles[i].reshape((4,2))
        p0_rect, p1_rect, p2_rect, p3_rect = poly
        # bottom left side p2,p3, bottom right side p1,p2
        angle = np.arctan((p2_rect[1]-p3_rect[1])/max(p2_rect[0]-p3_rect[0],1))
        xy_in_poly = np.argwhere(poly_mask == (i + 1))
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype = np.float32)
            # t, r, d, l -> t, b, l, r
            geo_map[y, x, 0] = util.point_dist_to_line(p0_rect, p1_rect, point)
            geo_map[y, x, 3] = util.point_dist_to_line(p1_rect, p2_rect, point)
            geo_map[y, x, 1] = util.point_dist_to_line(p2_rect, p3_rect, point)
            geo_map[y, x, 2] = util.point_dist_to_line(p3_rect, p0_rect, point)
            geo_map[y, x, 4] = angle
    return geo_map.transpose(2,0,1), scoremap


class DefaultTextDataLoader(object):
    def __init__(self, data_dir, char_dict, label_dict, text_pool_num=10):
        self.set_basic_info(data_dir, char_dict,label_dict,text_pool_num)
    def set_basic_info(self, data_dir, char_dict,label_dict, text_pool_num=10, font_num=100,ignore_label=-255):
        self.pool_num = text_pool_num
        self.label_dict = label_dict
        self.label_dict_keys = label_dict.keys()
        self.data_dir = data_dir
        self.save_data_info = pickle.load(open(os.path.join(self.data_dir,'save_data_info.pkl'),'rb'))
        self.bg_dir = self.save_data_info.bg_dir
        self.image_dir = self.save_data_info.img_dir
        self.alpha_dir = self.save_data_info.alpha_dir
        self.metadata_dir = self.save_data_info.metadata_dir
        self.prefixes = self.save_data_info.prefixes
        self.ignore_label = -255
    def get_valid_id(self, index):
        if index > self.__len__()-1:
            index = random.randint(0,self.__len__()-1)
        while(self.check_valid_id(index)==1):
            index = random.randint(0,self.__len__()-1)
        return index
    def check_valid_id(self, index):
        file_names=[]
        file_names.append(os.path.join(self.image_dir, f'{self.prefixes[index]}.jpg'))
        file_names.append(os.path.join(self.metadata_dir,f'{self.prefixes[index]}.pkl'))
        flag = 0
        for file_name in file_names:
            if os.path.isfile(file_name) == False:
                flag=1
                break
        return flag
    def check_valid_samples(self, index):
        file_names=[]
        file_names.append(os.path.join(self.image_dir, f'{self.prefixes[index]}.jpg'))
        file_names.append(os.path.join(self.metadata_dir,f'{self.prefixes[index]}.jpg'))
        flag = 0
        for i in range(len(file_names)):
            if os.path.isfile(file_names[i]) == False:
                flag=1
                break
        return flag
    def load_bg(self, index):
        prefix = self.prefixes[index]
        imn = os.path.join(self.bg_dir, f'{self.prefixes[index]}.jpg')
        bgimg = np.array(Image.open(imn).convert("RGB"))
        return bgimg
    def sort_clockwise(self, _rectangles):
        rectangles = np.zeros_like(_rectangles)
        for i in range(rectangles.shape[2]):
            pts = _rectangles[:,:,i].transpose((1,0))
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            rectangles[:,:,i]=rect.transpose((1,0))
        return rectangles
    def get_text_rectangles_from_metadata(self, metadata):
        text_rectangles = metadata.wordBB[::-1,:,:]
        text_rectangles = self.sort_clockwise(text_rectangles)        
        _, _, numOfWords = text_rectangles.shape
        text_rectangles = text_rectangles.reshape([8, numOfWords], order = 'F').T.astype(np.int32)
        return text_rectangles, numOfWords
    def get_char_rectangles_from_metadata(self, metadata):
        char_rectangles = metadata.charBB[::-1,:,:]
        _, _, numOfchar = char_rectangles.shape
        char_rectangles = np.expand_dims(char_rectangles, axis = 2) if (char_rectangles.ndim == 2) else char_rectangles
        char_rectangles = char_rectangles.reshape([8, numOfchar], order = 'F').T.astype(np.int32)  # num_words * 8
        return char_rectangles, numOfchar
    def get_transchar(self, transcripts):
        transchar = ''
        cnt = 0 
        for i in range(len(transcripts)):
            for j in range(len(transcripts[i])):
                if transcripts[i][j]==' ':
                    pass
                else:
                    transchar += transcripts[i][j]
                cnt+=1
        return transchar
    def get_shrinked_poly(self, poly):
        poly_shrinked = []
        for p in range(len(poly)):
            r = [None, None, None, None]
            for i in range(4):
                r[i] = min(np.linalg.norm(poly[p][i] - poly[p][(i + 1) % 4]),
                           np.linalg.norm(poly[p][i] - poly[p][(i - 1) % 4]))
            poly_shrinked.append(util.shrink_poly(poly[p].copy(),r,R=0.3))
        return np.array(poly_shrinked)
    def get_ins_mask(self, poly, imgsize):
        ins_mask = np.zeros((imgsize[1],imgsize[0]),dtype=np.int32)
        for i in range(len(poly)):
            cv2.fillPoly(ins_mask, poly[i:i+1].astype(np.int32), i+1)
        return ins_mask
    def get_scoremap(self, poly_shrinked, imgsize):
        scoremap = np.zeros((imgsize[1],imgsize[0]),dtype=np.int32)
        for i in range(len(poly_shrinked)):
            cv2.fillPoly(scoremap, poly_shrinked[i:i+1].astype(np.int32), 1)
        return scoremap
    def get_char_cls_mask(self, cpoly, imgsize, transchar):
        char_cls_mask = np.zeros((imgsize[1],imgsize[0]),dtype=np.int32)
        for i in range(len(cpoly)):
            char = transchar[i]
            if char in self.label_dict_keys:
                cv2.fillPoly(char_cls_mask, cpoly[i:i+1], int(self.label_dict[char])+1)
        return char_cls_mask
    def load_metadata(self, index):
        sn = os.path.join(self.metadata_dir,f'{self.prefixes[index]}.pkl')
        metadata = pickle.load(open(sn, 'rb'))
        return metadata
    def load_image(self, index):
        imn = os.path.join(self.image_dir, f'{self.prefixes[index]}.jpg')
        img = Image.open(imn).convert("RGB")
        return np.array(img)
    def load_ocr_data(self, index, metadata, imgsize):
        transcripts = metadata.texts
        text_rectangles, numOfWords = self.get_text_rectangles_from_metadata(metadata)
        char_rectangles, numOfchar = self.get_char_rectangles_from_metadata(metadata)
        transchar = self.get_transchar(transcripts)
        tpoly = text_rectangles.reshape(numOfWords, 4, 2)  # num_of_words * 4 * 2
        tpoly_shrinked = self.get_shrinked_poly(tpoly)
        text_ins_mask = self.get_ins_mask(tpoly, imgsize)
        text_scoremap = self.get_scoremap(tpoly_shrinked, imgsize)
        cpoly = char_rectangles[:,0:8].reshape(numOfchar, 4, 2)
        cpoly_shrinked = self.get_shrinked_poly(cpoly)
        char_ins_mask = self.get_ins_mask(cpoly, imgsize)
        char_scoremap = self.get_scoremap(cpoly_shrinked, imgsize)
        char_cls_mask = self.get_char_cls_mask(cpoly, imgsize,transchar)
        return (text_rectangles, text_ins_mask, text_scoremap,
                char_rectangles, char_ins_mask, char_scoremap, char_cls_mask)
    def load_alpha(self, img_size, index):
        file_name = os.path.join(self.alpha_dir, f'{self.prefixes[index]}.npz')
        if os.path.isfile(file_name) == True:
            _alpha = np.load(file_name)['arr_0'].astype(np.float32)/255.
            # (shadow, fill, stroke) -> (fill, shadow, stroke)
            alpha = _alpha.copy()
            alpha[:,:,0]=_alpha[:,:,1]
            alpha[:,:,1]=_alpha[:,:,0]
            alpha[:,:,1]=np.maximum(alpha[:,:,1]-alpha[:,:,0]-alpha[:,:,2],np.zeros_like(alpha[:,:,1]))
        else:
            alpha = np.zeros((img_size[1],img_size[0],self.effect_num))
        return alpha
    def get_font_data(self, metadata,text_num):
        font_label_vec = np.zeros((self.pool_num,),dtype=np.float32)+self.ignore_label
        for i in range(text_num):
            if (i >= self.pool_num):
                continue
            font_label_vec[i]=metadata.font_data[i].font_id
        return font_label_vec
    def load_font_size(self, metadata,text_num):
        font_size_vec = np.zeros((self.pool_num,),dtype=np.float32)+self.ignore_label
        for i in range(text_num):
            if i < self.pool_num:
                font_size_vec[i]=metadata.font_data[i].font_size
        return font_size_vec
    def get_stroke_data(self, metadata,text_num):
        stroke_param_num = 1
        stroke_param_vec = np.zeros((self.pool_num,stroke_param_num),dtype=np.float32)+self.ignore_label
        stroke_visibility_vec = np.zeros((self.pool_num,),dtype=np.float32)+self.ignore_label
        div_num = 5
        for i in range(text_num):
            if (i >= self.pool_num):
                continue
            if metadata.effect_visibility[i].stroke_visibility_flag==False:
                stroke_visibility_vec[i]=0
            else:
                stroke_visibility_vec[i]=1
            stroke_size = 25*(metadata.effect_params[i].stroke_param.border_weight/float(metadata.font_data[i].font_size)-0.01)
            stroke_param_vec[i,0,]=min((stroke_size*div_num*div_num)//div_num,div_num-1)
        return stroke_param_vec, stroke_visibility_vec
    def get_shadow_data(self, metadata, text_num):
        shadow_param_num = 5
        shadow_param_vec = np.zeros((self.pool_num,shadow_param_num),dtype=np.float32)+self.ignore_label
        shadow_visibility_vec = np.zeros((self.pool_num,),dtype=np.float32)+self.ignore_label
        for i in range(text_num):
            if (i >= self.pool_num):
                continue
            if metadata.effect_visibility[i].shadow_visibility_flag==False:
                shadow_visibility_vec[i]=0
            else:
                shadow_visibility_vec[i]=1
            shadow_param_vec[i,0]=float(metadata.effect_params[i].shadow_param.opacity)
            shadow_param_vec[i,1]=2*float(metadata.effect_params[i].shadow_param.blur)/metadata.font_data[i].font_size
            shadow_param_vec[i,2]=10*float(metadata.effect_params[i].shadow_param.dilation)/metadata.font_data[i].font_size#0~0.25
            shadow_param_vec[i,3]=(float(metadata.effect_params[i].shadow_param.offset_x)/metadata.font_data[i].font_size*5)#-0.5~0.5
            shadow_param_vec[i,4]=(float(metadata.effect_params[i].shadow_param.offset_y)/metadata.font_data[i].font_size*5)
        return shadow_param_vec, shadow_visibility_vec
    def get_effect_data(self, metadata,text_num):
        shadow_data = self.get_shadow_data(metadata,text_num)
        stroke_data = self.get_stroke_data(metadata,text_num)
        return shadow_data, stroke_data
    
    def load_char(self, index, label_dict, meta_info):
        txt = metadata.texts
        char_label_np = np.zeros((500),dtype=np.int32)-2
        char_index2text_index = np.zeros((500),dtype=np.int32)
        cnt = 0
        for i in range(len(txt)):
            for j in range(len(txt[i])):
                if txt[i][j]==' ':
                    pass
                else:
                    char_index2text_index[cnt]=i
                    char_label_np[cnt]=int(label_dict[txt[i][j]])
                    cnt+=1
        return char_label_np, txt, char_index2text_index
    def __len__(self):
        return len(self.prefixes)

def get_setup_data(data_list, char_dict,label_dict,text_pool_num):
    dataset_list=[]
    _dataset_priors=[]
    dataset_options=[]
    for dcfg in data_list:
        dataset = DefaultTextDataLoader(
            data_dir=dcfg.data_dir,
            char_dict=char_dict,
            label_dict=label_dict, 
            text_pool_num=text_pool_num
        )
        dataset_list.append(dataset)
        _dataset_priors.append(dcfg.prior)
        dataset_options.append(dcfg.noisy_bg_option)
    
    dataset_priors = []
    psum = sum(_dataset_priors)
    pr_tmp = 0
    for pr in _dataset_priors:
        pr /= psum
        pr_tmp += pr
        dataset_priors.append(pr_tmp)
        
    return dataset_list, dataset_priors, dataset_options

class TextParserLoader(torch.utils.data.Dataset):
    def __init__(self, data_list, text_pool_num=10, char_pool_num=40):
        super().__init__()
        self.text_pool_num = text_pool_num
        self.char_pool_num = char_pool_num
        char_dict, label_dict = load_char_label_dicts()
        self.transform = transforms.Compose([
            util.RandomResizeWithBB(640,640),
            util.RandomCropWithBB(640),
        ])
        self.norm_charnetinp = util.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        self.norm_alphainp = util.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        self.dataset_list, self.dataset_priors, self.dataset_options = get_setup_data(data_list, char_dict, label_dict, text_pool_num)
    def choice_dataset(self):
        v = random.random()
        dataset_id=-1
        for index, p in enumerate(self.dataset_priors):
            if v < p:
                dataset_id = index
                break
        return self.dataset_list[dataset_id], dataset_id
    def all_squeeze(self, tensor_list):
        new_list=[]
        for i in range(len(tensor_list)):
            new_list.append(tensor_list[i].squeeze())
        return new_list
    def get_fgmask_from_insmask(self, instance_mask, ):
        fg_mask = instance_mask.copy()
        fg_mask[fg_mask>0]=1
        return fg_mask
    def get_valid_text_index(self, text_ins_mask):
        bbox_num = self.text_pool_num
        valid_text_indexes = np.zeros((bbox_num,),dtype=np.float32)
        for b in range(bbox_num):
            mask_region = text_ins_mask == (b + 1)
            mask_num = np.sum(mask_region)
            if mask_num>0:
                valid_text_indexes[b]=1
        return valid_text_indexes
    def jpeg_compression(self, img):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 45+random.random()*50]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg
    def pre_process(self, img):
        img_norm = self.norm_charnetinp(img).transpose(2,0,1)
        img_alpha = self.norm_alphainp(img).transpose(2,0,1)
        return img_norm, img_alpha
    def get_ocr_data(self, text_char_data, rectangle_data):
        text_ins_mask, char_ins_mask, char_cls_mask, text_scoremap, char_scoremap = text_char_data
        [text_rectangles, char_rectangles] = rectangle_data
        # limit rectangle number for convinience
        text_rectangles_array = np.zeros((self.text_pool_num,8))
        text_rectangles_array[0:min(len(text_rectangles),self.text_pool_num),:] = np.array(text_rectangles)[0:min(len(text_rectangles),self.text_pool_num),:]
        char_rectangles_array = np.zeros((self.char_pool_num,8))
        char_rectangles_array[0:min(len(char_rectangles),self.char_pool_num),:] = np.array(char_rectangles)[0:min(len(char_rectangles),self.char_pool_num),:]
        valid_text_index = self.get_valid_text_index(text_ins_mask)
        # get foreground mask from instance mask
        text_fg_mask = self.get_fgmask_from_insmask(text_ins_mask)
        char_fg_mask = self.get_fgmask_from_insmask(char_ins_mask)
        if self.dataset_options[self.dataset_id]:
            text_fg_mask[:]=self.ignore_label
        # remove background category
        char_cls_mask = char_cls_mask-1
        char_masks = (char_fg_mask, char_ins_mask, char_cls_mask)
        
        # get geometory map
        text_geomap, text_scoremap_qt = get_geomap(text_scoremap, text_rectangles, text_ins_mask)
        char_geomap, char_scoremap_qt = get_geomap(char_scoremap, char_rectangles, char_ins_mask)
        # ocr data
        text_level_data = (text_fg_mask, text_ins_mask, text_scoremap, text_scoremap_qt, text_geomap, text_rectangles_array)
        char_level_data = (char_fg_mask, char_ins_mask, char_scoremap, char_scoremap_qt, char_geomap, char_cls_mask, char_rectangles_array)
        ocr_data = (text_level_data, char_level_data)

        # maximum text number in training data
        text_num = int(np.max(text_ins_mask))
        return ocr_data, valid_text_index, text_num

    def get_style_data(self, dataset, metadata, text_num, alpha, valid_text_index):
        shadow_data, stroke_data = dataset.get_effect_data(metadata, text_num)
        font_label = dataset.get_font_data(metadata, text_num)
        style_data = (alpha, shadow_data, stroke_data, font_label, valid_text_index)
        return style_data

    def load_data(self, index, dataset):
        img_org = dataset.load_image(index)
        metadata = dataset.load_metadata(index)
        text_char_data = dataset.load_ocr_data(index, metadata, (img_org.shape[1],img_org.shape[0]))
        alpha = dataset.load_alpha((img_org.shape[1],img_org.shape[0]), index)
        return img_org, metadata, text_char_data, alpha
    def __getitem__(self, index):
        # set index
        dataset,dataset_id = self.choice_dataset()
        self.dataset_id = dataset_id
        index = dataset.get_valid_id(index)
        # load data
        img_org, metadata, text_char_data, alpha = self.load_data(index, dataset)
        (text_rectangles, text_ins_mask, text_scoremap,
        char_rectangles, char_ins_mask, char_scoremap, char_cls_mask) = text_char_data
        # augmentation by jpeg compression
        if random.random()<0.5:
            img_org = self.jpeg_compression(img_org)
        # transformation
        input_array = [img_org, text_ins_mask, char_ins_mask, char_cls_mask, text_scoremap, char_scoremap, alpha]
        input_rectangles = [text_rectangles, char_rectangles]
        output_array, output_rectangles = self.transform((input_array, input_rectangles))
        [img_org, text_ins_mask, char_ins_mask, char_cls_mask, text_scoremap, char_scoremap, alpha] = output_array
        # pre-processing and reshape
        img_norm, img_alpha = self.pre_process(img_org)
        alpha = alpha.transpose(2,0,1)
        squeeze_outs = self.all_squeeze((text_ins_mask, char_ins_mask, char_cls_mask, text_scoremap, char_scoremap))
        # get ocr data
        ocr_data, valid_text_index, text_num  = self.get_ocr_data(squeeze_outs, output_rectangles)
        # get style data
        style_data = self.get_style_data(dataset, metadata, text_num, alpha, valid_text_index)
        return img_norm, img_alpha, ocr_data, style_data

    def __len__(self):
        return 10000
        #return len(self.dataset_list[0].prefixes)




class InpaintorLoader(torch.utils.data.Dataset):
    def __init__(self, data_list, text_pool_num=10):
        super().__init__()
        char_dict, label_dict = load_char_label_dicts()
        self.transform = transforms.Compose([
            util.RandomResizeWithBB(640,640),
            util.RandomCropWithBB(640),
        ])
        self.norm = util.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        self.dataset_list, self.dataset_priors, self.dataset_options = get_setup_data(data_list, char_dict, label_dict, text_pool_num)
    def choice_dataset(self):
        v = random.random()
        dataset_id=-1
        for index, p in enumerate(self.dataset_priors):
            if v < p:
                dataset_id = index
                break
        return self.dataset_list[dataset_id], dataset_id
    def all_squeeze(self, tensor_list):
        new_list=[]
        for i in range(len(tensor_list)):
            new_list.append(tensor_list[i].squeeze())
        return new_list
    def pre_process(self, img):
        img_norm = self.norm(img).transpose(2,0,1)
        return img_norm
    def load_data(self, index, dataset):
        img_org = dataset.load_image(index)
        bg_org = dataset.load_bg(index)
        alpha = dataset.load_alpha((img_org.shape[1],img_org.shape[0]), index)
        return img_org, bg_org, alpha

    def get_coords_for_paste(self, alpha_aug, alpha):
        dy = abs(alpha.shape[0]-alpha_aug.shape[0])//2
        if alpha.shape[0]>alpha_aug.shape[0]:
            aug_y0, aug_y1 = 0, alpha_aug.shape[0]
            tmp_y0, tmp_y1 = dy, dy+alpha_aug.shape[0]
        else:
            aug_y0, aug_y1 = dy, dy+alpha.shape[0]
            tmp_y0, tmp_y1 = 0, alpha.shape[0]
        dx = abs(alpha.shape[1]-alpha_aug.shape[1])//2
        if alpha.shape[1]>alpha_aug.shape[1]:
            aug_x0, aug_x1 = 0, alpha_aug.shape[1]
            tmp_x0, tmp_x1 = dx, dx+alpha_aug.shape[1]
        else:
            aug_x0, aug_x1 = dx, dx+alpha.shape[1]
            tmp_x0, tmp_x1 = 0, alpha.shape[1]
        return (aug_y0,aug_y1,aug_x0,aug_x1),(tmp_y0,tmp_y1,tmp_x0,tmp_x1)

    def aug_alpha(self, img, alpha, dataset):
        aug_alpha_num = random.randint(0,2)
        alpha_aug = np.zeros((alpha.shape[0],alpha.shape[1],max(aug_alpha_num,1)))
        alpha_aug[:,:,0]=np.max(alpha,2)
        for i in range(1,aug_alpha_num):
            alpha = dataset.load_alpha(img, random.randint(0,len(dataset.prefixes)-1))
            alpha = np.max(alpha,2)
            (aug_y0,aug_y1,aug_x0,aug_x1),(tmp_y0,tmp_y1,tmp_x0,tmp_x1) = self.get_coords_for_paste(alpha_aug, alpha)
            alpha_aug[aug_y0:aug_y1,aug_x0:aug_x1,i]=alpha[tmp_y0:tmp_y1,tmp_x0:tmp_x1]
        alpha = np.max(alpha_aug,2)
        return alpha

    def __getitem__(self, index):
        # set index
        dataset,dataset_id = self.choice_dataset()
        self.dataset_id = dataset_id
        index = dataset.get_valid_id(index)
        # load data
        img_org, bg_org, alpha = self.load_data(index, dataset)
        alpha = self.aug_alpha(img_org, alpha, dataset)
        input_array = [img_org, bg_org, alpha]
        output_array, _ = self.transform((input_array, []))
        [img_org, bg_org, alpha] = output_array
        # structure image
        # we simplify this step from the original paper
        bg_smooth = cv2.bilateralFilter(bg_org,9,75,75)
        for i in range(5):
            bg_smooth = cv2.bilateralFilter(bg_smooth,9,75,75)
        # norm
        img_norm = self.pre_process(img_org)
        bg_norm = self.pre_process(bg_org)
        bg_smooth_norm = self.pre_process(bg_smooth)
        alpha = alpha.transpose(2,0,1)
        return img_norm, bg_norm, bg_smooth_norm, alpha

    def __len__(self):
        return 10000
        #return 100
