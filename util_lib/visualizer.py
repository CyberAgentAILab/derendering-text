import os
import pickle
import cv2
import numpy as np
from .tfd_renderer import render_tfd


def main(args):
    tfd = pickle.load(open(args.filename, 'rb'))
    output_img = render_tfd(tfd)
    # save rendered image from rendering parameters
    save_dir = 'gen_data/vis'
    cv2.imwrite(os.path.join(save_dir, 'rendered_img.jpg'),output_img[:,:,::-1])
    img_save = output_img.copy().astype(np.uint8)
    
    # visualize character bounding boxes
    for j in range(tfd.charBB.shape[2]):
        box = tfd.charBB[:,:,j]
        y1, y2, y3, y4  = box[0]
        x1, x2, x3, x4  = box[1]
        ys = min(y1,y2,y3,y4)
        ye = max(y1,y2,y3,y4)
        xs = min(x1,x2,x3,x4)
        xe = max(x1,x2,x3,x4)
        img_save = cv2.line(img_save,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
        img_save = cv2.line(img_save,(int(x2),int(y2)),(int(x3),int(y3)),(255,0,0),2)
        img_save = cv2.line(img_save,(int(x3),int(y3)),(int(x4),int(y4)),(255,0,0),2)
        img_save = cv2.line(img_save,(int(x4),int(y4)),(int(x1),int(y1)),(255,0,0),2)
    cv2.imwrite(os.path.join(save_dir, 'charBB.jpg'),img_save[:,:,::-1])
    
    # visualize text bounding boxes
    img_save = output_img.copy().astype(np.uint8)
    for j in range(tfd.wordBB.shape[2]):
        box = tfd.wordBB[:,:,j]
        y1, y2, y3, y4  = box[0]
        x1, x2, x3, x4  = box[1]
        ys = min(y1,y2,y3,y4)
        ye = max(y1,y2,y3,y4)
        xs = min(x1,x2,x3,x4)
        xe = max(x1,x2,x3,x4)
        img_save = cv2.line(img_save,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
        img_save = cv2.line(img_save,(int(x2),int(y2)),(int(x3),int(y3)),(255,0,0),2)
        img_save = cv2.line(img_save,(int(x3),int(y3)),(int(x4),int(y4)),(255,0,0),2)
        img_save = cv2.line(img_save,(int(x4),int(y4)),(int(x1),int(y1)),(255,0,0),2)
    cv2.imwrite(os.path.join(save_dir, 'textBB.jpg'),img_save[:,:,::-1])
    
    # visualize alpha maps
    alpha_maps = np.load(tfd.alpha)['arr_0']
    cv2.imwrite(os.path.join(save_dir, 'shadow_alpha.jpg'),alpha_maps[:,:,0])
    cv2.imwrite(os.path.join(save_dir, 'fill_alpha.jpg'),alpha_maps[:,:,1])
    cv2.imwrite(os.path.join(save_dir, 'border_alpha.jpg'),alpha_maps[:,:,2])

    # visualize effect bounding boxes
    img_save = output_img.copy().astype(np.uint8)
    for j in range(tfd.effect_merged_alphaBB.shape[2]):
        box = tfd.effect_merged_alphaBB[:,:,j]
        y1, y2, y3, y4  = box[0]
        x1, x2, x3, x4  = box[1]
        ys = min(y1,y2,y3,y4)
        ye = max(y1,y2,y3,y4)
        xs = min(x1,x2,x3,x4)
        xe = max(x1,x2,x3,x4)
        img_save = cv2.line(img_save,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
        img_save = cv2.line(img_save,(int(x2),int(y2)),(int(x3),int(y3)),(255,0,0),2)
        img_save = cv2.line(img_save,(int(x3),int(y3)),(int(x4),int(y4)),(255,0,0),2)
        img_save = cv2.line(img_save,(int(x4),int(y4)),(int(x1),int(y1)),(255,0,0),2)
    cv2.imwrite(os.path.join(save_dir, 'merged_alpha_BB.jpg'),img_save[:,:,::-1])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize generated training data.')
    parser.add_argument('--filename', type=str,default='gen_data/load_eng_tmp/metadata/0_0.pkl',help='filename for visualization')
    args,_ = parser.parse_known_args()
    main(args)


