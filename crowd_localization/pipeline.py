import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import tqdm
from model.locator import Crowd_locator
from misc.utils import read_pred_and_gt
from PIL import Image
import cv2
from collections import OrderedDict
import numpy as np

# Constants for configuration
MODEL_PATH = '../PretrainedCrowdLocModel/NWPU-HR-ep_241_F1_0.802_Pre_0.841_Rec_0.766_mae_55.6_mse_330.9.pth'
BASE_DATASET = 'NWPU'
BASE_DATA_ROOT = f'../ProcessedData/{BASE_DATASET}'
TEST_DATA_ID = '제주탑동광장056'
TEST_DATA_NAME = "TS_2.시나리오_56.Outdoor_제주탑동광장056(544)"
TEST_DATA_ROOT = f'../ProcessedData/Aihub/images/Training/{TEST_DATA_NAME}'
TEST_LIST = 'list.txt'
OUT_FILE_NAME = f'./saved_exp_results/{TEST_DATA_ID}_result.txt'
EXP_NAME = f'./saved_exp_results/{TEST_DATA_ID}_vis_results'
WIDTH = 1280
HEIGHT = 720
VIDEO_OUTPUT_PATH = f'./saved_exp_results/{TEST_DATA_ID}_video.avi'
FPS = 3
VIDEO_CODEC = 'XVID'

def main():
    torch.backends.cudnn.benchmark = True

    mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    txtpath = os.path.join(TEST_DATA_ROOT, TEST_LIST)
    with open(txtpath) as f:
        lines = f.readlines()
    
    test(lines, MODEL_PATH, OUT_FILE_NAME, TEST_DATA_ROOT, img_transform)
    visualize_predictions(OUT_FILE_NAME, TEST_DATA_ROOT, EXP_NAME)
    create_video_from_images(EXP_NAME, VIDEO_OUTPUT_PATH, FPS, WIDTH, HEIGHT, VIDEO_CODEC)

def get_boxInfo_from_Binar_map(Binar_numpy, min_area=3):
    Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
    assert Binar_numpy.ndim == 2
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)

    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]
    points = points[index]
    pre_data = {'num': len(points), 'points': points}
    return pre_data, boxes

def test(file_list, model_path, out_file_name, data_root, img_transform):
    device = torch.device('cpu')
    net = Crowd_locator('HR_Net', pretrained=True).to(device)
    state_dict = torch.load(model_path, map_location=device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if 'module.' in k else k
        new_state_dict[new_key] = v

    net.load_state_dict(new_state_dict)
    net.eval()

    file_list = tqdm.tqdm(file_list)
    for infos in file_list:
        filename = infos.split()[0]
        imgname = os.path.join(data_root, 'images', f'{filename}.jpg')
        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)[None, :, :, :]
        slice_h, slice_w = 512, 1024

        with torch.no_grad():
            img = img.to(device)
            b, c, h, w = img.shape
            crop_imgs, crop_masks = [], []

            if h * w < slice_h * 2 * slice_w * 2 and h % 16 == 0 and w % 16 == 0:
                [pred_threshold, pred_map, __] = [i.cpu() for i in net(img, mask_gt=None, mode='val')]
            else:
                if h % 16 != 0:
                    pad_dims = (0, 0, 0, 16 - h % 16)
                    h = (h // 16 + 1) * 16
                    img = F.pad(img, pad_dims, "constant")

                if w % 16 != 0:
                    pad_dims = (0, 16 - w % 16, 0, 0)
                    w = (w // 16 + 1) * 16
                    img = F.pad(img, pad_dims, "constant")

                for i in range(0, h, slice_h):
                    h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                    for j in range(0, w, slice_w):
                        w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                        crop_imgs.append(img[:, :, h_start:h_end, w_start:w_end])
                        mask = torch.zeros(1, 1, img.size(2), img.size(3)).cpu()
                        mask[:, :, h_start:h_end, w_start:w_end].fill_(1.0)
                        crop_masks.append(mask)
                crop_imgs, crop_masks = torch.cat(crop_imgs, dim=0), torch.cat(crop_masks, dim=0)

                crop_preds, crop_thresholds = [], []
                nz, period = crop_imgs.size(0), 4
                for i in range(0, nz, period):
                    [crop_threshold, crop_pred, __] = [i.cpu() for i in net(crop_imgs[i:min(nz, i + period)], mask_gt=None, mode='val')]
                    crop_preds.append(crop_pred)
                    crop_thresholds.append(crop_threshold)

                crop_preds = torch.cat(crop_preds, dim=0)
                crop_thresholds = torch.cat(crop_thresholds, dim=0)

                idx = 0
                pred_map = torch.zeros(b, 1, h, w).cpu()
                pred_threshold = torch.zeros(b, 1, h, w).cpu().float()
                for i in range(0, h, slice_h):
                    h_start, h_end = max(min(h - slice_h, i), 0), min(h, i + slice_h)
                    for j in range(0, w, slice_w):
                        w_start, w_end = max(min(w - slice_w, j), 0), min(w, j + slice_w)
                        pred_map[:, :, h_start:h_end, w_start:w_end] += crop_preds[idx]
                        pred_threshold[:, :, h_start:h_end, w_start:w_end] += crop_thresholds[idx]
                        idx += 1
                mask = crop_masks.sum(dim=0)
                pred_map = (pred_map / mask)
                pred_threshold = (pred_threshold / mask)

            a = torch.ones_like(pred_map)
            b = torch.zeros_like(pred_map)
            binar_map = torch.where(pred_map >= pred_threshold, a, b)

            pred_data, boxes = get_boxInfo_from_Binar_map(binar_map.cpu().numpy())

            with open(out_file_name, 'a') as f:
                f.write(f'{filename} {pred_data["num"]} ')
                for ind, point in enumerate(pred_data['points'], 1):
                    if ind < pred_data['num']:
                        f.write(f'{int(point[0])} {int(point[1])} ')
                    else:
                        f.write(f'{int(point[0])} {int(point[1])}')
                f.write('\n')

def visualize_predictions(pred_file, data_root, exp_name):
    img_path = os.path.join(data_root, 'images')

    if not os.path.exists(exp_name):
        os.mkdir(exp_name)

    pred_data, _ = read_pred_and_gt(pred_file)

    with open(os.path.join(data_root, TEST_LIST), 'r') as f:
        img_filenames = [line.strip() for line in f]

    for filename in img_filenames:
        sample_id = int(filename.split('.')[0])
        pred_p = []

        if pred_data[sample_id]['num'] != 0:
            pred_p = pred_data[sample_id]['points']

        img = Image.open(os.path.join(img_path, f'{filename}.jpg'))
        img = img.resize((WIDTH, HEIGHT))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        point_r_value = 10
        if pred_data[sample_id]['num'] != 0:
            for point in pred_p:
                cv2.circle(img, (int(point[0]), int(point[1])), point_r_value, (0, 255, 0), 2)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img.save(os.path.join(exp_name, f'{sample_id}_pred.jpg'))

def create_video_from_images(image_folder, video_output_path, fps, width, height, codec='XVID'):
    def sort_key(file_name):
        return int(file_name.split('_')[0])
    
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))], key=sort_key)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        with Image.open(image_path) as img:
            img_resized = img.resize((width, height))
            img_array = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        
        if img_array is not None:
            video.write(img_array)
    
    video.release()
    print(f"Video saved to {video_output_path}")

if __name__ == '__main__':
    main()
