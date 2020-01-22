import os

import argparse
import cv2
import numpy as np
import torch

from Network.SDCNet import SDCNet_VGG16_classify
from Network.SSDCNet import SSDCNet_classify


def main(model_path, video_path, SDCNet=False):

    cap = cv2.VideoCapture(video_path)

    # --1.2 use initial setting to generate
    # set label_indice
    label_indice = np.arange(0.5, 22+0.5/2, 0.5)
    add = np.array([1e-6, 0.05, 0.10, 0.15, 0.20,
                    0.25, 0.30, 0.35, 0.40, 0.45])
    label_indice = np.concatenate((add, label_indice))

    # init networks
    label_indice = torch.Tensor(label_indice)
    class_num = len(label_indice)+1

    div_times = 2

    if SDCNet:
        net = SDCNet_VGG16_classify(
            class_num, label_indice, load_weights=True).cuda()
    else:
        net = SSDCNet_classify(class_num, label_indice, div_times=div_times,
                               frontend_name='VGG16', block_num=5,
                               IF_pre_bn=False, IF_freeze_bn=False, load_weights=True,
                               parse_method='maxp').cuda()

    if os.path.exists(model_path):
        print("Adding Weights ....")
        all_state_dict = torch.load(model_path)
        net.load_state_dict(all_state_dict['net_state_dict'])
        net.eval()
    else:
        print("Can't find trained weights!!")
        exit()

    first_frame = True
    while cap.isOpened():
        flag, image = cap.read()
        output_image = np.copy(image)
        if flag:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            break
        if first_frame:
            rois = cv2.selectROIs("frame", image, False, False)
            first_frame = False
            # print(roi)

        sum = 0
        for roi in rois:
            roi_image = image[int(roi[1]):int(roi[1]+roi[3]),
                          int(roi[0]):int(roi[0]+roi[2])]

            roi_image = cv2.resize(roi_image, (256, 256))

            roi_image = np.transpose(roi_image, (2, 0, 1))
            roi_image = torch.Tensor(roi_image[None, :, :, :])
            # w = image.shape[-1]
            # h = image.shape[-2]
            # pad_w = 64 - w%64
            # padding_left = int(pad_w/2)
            # padding_right = pad_w - padding_left
            # pad_h = 64 - h%64
            # padding_top = int(pad_h/2)
            # padding_bottom = pad_h - padding_top
            # image = torch.nn.functional.pad(image, (padding_left, padding_right, padding_top, padding_bottom))

            if torch.cuda.is_available():
                roi_image = roi_image.cuda()
            with torch.no_grad():
                features = net(roi_image)
                div_res = net.resample(features)
                merge_res = net.parse_merge(div_res)
                if SDCNet:
                    outputs = merge_res['div'+str(net.args['div_times'])]
                else:
                    outputs = merge_res['div'+str(net.div_times)]

                del merge_res

            cv2.rectangle(output_image, (int(roi[0]), int(roi[1])), (int(
                roi[0]+roi[2]), int(roi[1]+roi[3])), (255, 0, 0), thickness=3)
            sum += int(outputs.sum().item())

        cv2.putText(output_image, "{}".format(sum),
                    (30, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 3)

        cv2.imshow("frame", output_image)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            exit()

    cap.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "S-DCNet: Spatial Divide-and-Conquer Crowd Counting")
    parser.add_argument("model", type=str, help="Pretrained weights")
    parser.add_argument("--video", "-v", type=str,
                        required=True, help="The video path to crowd count")
    parser.add_argument("--SDCNet", action='store_true',
                        default=False, help="Check it if you want to use SDCNet")
    args = parser.parse_args()

    main(args.model, args.video, SDCNet=args.SDCNet)
