from ultralytics import YOLO

import torch
import glob
import cv2
import numpy as np
import wandb
import argparse
from pathlib import Path
import random
from scipy.special import comb
import matplotlib.pyplot as plt
#matplotlib inline

class BezierSampler(torch.nn.Module):
    # Fast Batch Bezier sampler
    def __init__(self, order, num_sample_points, proj_coefficient=0):
        super().__init__()
        self.proj_coefficient = proj_coefficient
        self.num_control_points = order + 1
        self.num_sample_points = num_sample_points
        self.control_points = []
        self.bezier_coeff = self.get_bezier_coefficient()
        self.bernstein_matrix = self.get_bernstein_matrix()

    def get_bezier_coefficient(self):
        Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
        BezierCoeff = lambda ts: [[Mtk(self.num_control_points - 1, t, k) for k in range(self.num_control_points)] for t
                                  in ts]
        return BezierCoeff

    def get_bernstein_matrix(self):
        t = torch.linspace(0, 1, self.num_sample_points)
        if self.proj_coefficient != 0:
            # tokens = tokens + (1 - tokens) * tokens ** self.proj_coefficient
            t[t > 0.5] = t[t > 0.5] + (1 - t[t > 0.5]) * t[t > 0.5] ** self.proj_coefficient
            t[t < 0.5] = 1 - (1 - t[t < 0.5] + t[t < 0.5] * (1 - t[t < 0.5]) ** self.proj_coefficient)
        c_matrix = torch.tensor(self.bezier_coeff(t))
        return c_matrix

    def get_sample_points(self, control_points_matrix):
        if control_points_matrix.numel() == 0:
            return control_points_matrix  # Looks better than a torch.Tensor
        if self.bernstein_matrix.device != control_points_matrix.device:
            self.bernstein_matrix = self.bernstein_matrix.to(control_points_matrix.device)

        return upcast(self.bernstein_matrix).matmul(upcast(control_points_matrix))



def bezier_curve(points, t):
    n = len(points) - 1
    curve = np.zeros_like(points[0])
    for i in range(n + 1):
        #curve += comb(n, i) * ((1 - t) ** (n - i)) * (t ** i) * points[i]
        curve = curve + comb(n, i) * ((1 - t) ** (n - i)) * (t ** i) * points[i]
    return curve


def fit_bezier_curve(data, num_points=100):
    t = np.linspace(0, 1, num_points)
    curve_points = np.array([bezier_curve(data, ti) for ti in t])
    return curve_points



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_name", type=str, default='ouc_visiongroup')
    parser.add_argument("--project_name", type=str, default='SC_DUO_Ablation')
    parser.add_argument("--experiment_name", type=str, default='no_endpointloss')
    parser.add_argument("--scenario_name", type=str, default='no_endpointloss')
    parser.add_argument("--seed", type=int, default=0)
    opt = parser.parse_args()

    wandb.init(config=opt,
               project=opt.project_name,
               entity=opt.team_name,
               name=opt.weights,
               group=opt.scenario_name,
               job_type="training",
               reinit=True)

    wandb_images = []
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    #model path
    model = YOLO('/data/m_DUO_best.pt')
    model.to(device)
    #datapath
    img_path = 'fig/2258.jpg'
    img_path = Path(img_path)

    pict_folder = []
    for pic in glob.glob(str(img_path / '*.jpg')):
        pic_path = str(img_path/pic)
        # results = model.predict(pic_path, conf=0.3, iou=0.3, save_txt =True, visualize = True)
        results = model.predict(pic_path, conf=0.3, iou=0.3)
        num_bbox = len(results[0].boxes.cls)
        bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
        bboxes_keypoints = results[0].keypoints.xy.cpu().numpy().astype('uint32')
        img_bgr = cv2.imread(pic_path)

        bbox_color = (0, 0, 255) 
        bbox_thickness = 3 

 
        bbox_labelstr = {
            'font_size': 1,
            'font_thickness': 1,  
            'offset_x': 0, 
            'offset_y': 0,  
        }
        kpt_color_map = {

            0: {'name': '1', 'color': [60, 45, 255], 'radius': 5},  
            1: {'name': '2', 'color': [150, 150, 255], 'radius': 5},  
            2: {'name': '3', 'color': [20, 120, 255], 'radius': 5},
            3: {'name': '4', 'color': [30, 170, 255], 'radius': 5},  
            4: {'name': '5', 'color': [40, 210, 210], 'radius': 5},
        }


        kpt_labelstr = {
            'font_size': 1,  
            'font_thickness': 1, 
            'offset_x': 1,  
            'offset_y': 1,  
        }
        for idx in range(num_bbox): 

           
            bbox_xyxy = bboxes_xyxy[idx]

            
            bbox_label = results[0].names[0]

         
            img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                                    bbox_thickness)
            
            bbox_label_conf = 'SC'

            bbox_keypoints = bboxes_keypoints[idx]


            color = (0, 0, 255)

            curve = fit_bezier_curve(bbox_keypoints, 50)
            curve_points = curve.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img_bgr, [curve_points], False, color, 3)

            for kpt_id in kpt_color_map:

                kpt_color = kpt_color_map[kpt_id]['color']
                kpt_radius = kpt_color_map[kpt_id]['radius']
                kpt_x = bbox_keypoints[kpt_id][0]
                kpt_y = bbox_keypoints[kpt_id][1]

                img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)


                kpt_label = str(kpt_color_map[kpt_id]['name']) 

        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        wandb_images.append(wandb.Image(image_rgb, caption=pic))
    wandb.log({'SC-DUO': wandb_images})
    wandb.finish()

