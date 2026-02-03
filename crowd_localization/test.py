import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import tqdm
from model.locator import Crowd_locator
from misc.utils import *
from PIL import Image, ImageOps
import cv2 
from collections import OrderedDict

# DATASET = 'NWPU'
# DATA_ROOT = '../ProcessedData/' + DATASET
# TEST_LIST = 'val.txt'
# MODEL_PATH = '../PretrainedCrowdLocModel/NWPU-HR-ep_241_F1_0.802_Pre_0.841_Rec_0.766_mae_55.6_mse_330.9.pth'
# OUT_FILE_NAME = './saved_exp_results/' + DATASET + '_HR_Net_' + TEST_LIST

# Constants for configuration
DATASET = 'NWPU'
DATA_ROOT = '../ProcessedData/' + DATASET
TEST_LIST = 'new.txt'  # List of images to process
MODEL_PATH = '../PretrainedCrowdLocModel/NWPU-HR-ep_241_F1_0.802_Pre_0.841_Rec_0.766_mae_55.6_mse_330.9.pth'
OUT_FILE_NAME = './saved_exp_results/' + DATASET + '_HR_Net_' + TEST_LIST

def main():
    torch.backends.cudnn.benchmark = True

    # Define mean and standard deviation for normalization based on dataset
    if DATASET == 'NWPU':
        mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
    elif DATASET == 'SHHA':
        mean_std = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
    elif DATASET == 'SHHB':
        mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])
    elif DATASET == 'QNRF':
        mean_std = ([0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449])
    elif DATASET == 'FDST':
        mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])
    elif DATASET == 'JHU':
        mean_std = ([0.429683953524, 0.437104910612, 0.421978861094], [0.235549390316, 0.232568427920, 0.2355950474739])
    else:
        raise ValueError(f"Unknown dataset: {DATASET}")

    # Define image transformation for input images
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    
    # Define image restoration transformation
    restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    # Read the list of image filenames to process
    txtpath = os.path.join(DATA_ROOT, TEST_LIST)
    with open(txtpath) as f:
        lines = f.readlines()
    
    # Call the test function to process the images
    test(lines, MODEL_PATH, OUT_FILE_NAME, DATA_ROOT, img_transform)

def get_boxInfo_from_Binar_map(Binar_numpy, min_area=3):
    """Get bounding box information from binary map."""
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

def test(file_list, model_path, out_file_name, dataRoot, img_transform):
    """Process each image in the file list, perform crowd localization, and save results."""
    device = torch.device('cpu')
    net = Crowd_locator('HR_Net', pretrained=True).to(device)
    state_dict = torch.load(model_path, map_location=device)

    # Remove 'module.' prefix if present in state_dict keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if 'module.' in k else k
        new_state_dict[new_key] = v

    net.load_state_dict(new_state_dict)
    net.eval()

    file_list = tqdm.tqdm(file_list)
    for infos in file_list:
        filename = infos.split()[0]
        imgname = os.path.join(dataRoot, 'images', filename + '.jpg')
        img = Image.open(imgname)

        # Convert grayscale images to RGB
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)[None, :, :, :]
        slice_h, slice_w = 512, 1024
        with torch.no_grad():
            img = img.to(device)
            b, c, h, w = img.shape
            crop_imgs, crop_dots, crop_masks = [], [], []
            if h * w < slice_h * 2 * slice_w * 2 and h % 16 == 0 and w % 16 == 0:
                [pred_threshold, pred_map, __] = [i.cpu() for i in net(img, mask_gt=None, mode='val')]
            else:
                # Pad the image if its dimensions are not multiples of 16
                if h % 16 != 0:
                    pad_dims = (0, 0, 0, 16 - h % 16)
                    h = (h // 16 + 1) * 16
                    img = F.pad(img, pad_dims, "constant")

                if w % 16 != 0:
                    pad_dims = (0, 16 - w % 16, 0, 0)
                    w = (w // 16 + 1) * 16
                    img = F.pad(img, pad_dims, "constant")

                # Crop the image into smaller patches and process each patch
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
                f.write(filename + ' ')
                f.write(str(pred_data['num']) + ' ')
                for ind, point in enumerate(pred_data['points'], 1):
                    if ind < pred_data['num']:
                        f.write(str(int(point[0])) + ' ' + str(int(point[1])) + ' ')
                    else:
                        f.write(str(int(point[0])) + ' ' + str(int(point[1])))
                f.write('\n')

if __name__ == '__main__':
    main()
