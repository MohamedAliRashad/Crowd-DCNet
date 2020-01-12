import os

import argparse
import cv2
import numpy as np
import torch

from Network.SDCNet import SDCNet_VGG16_classify


def main(model_path, video_path):

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
    net = SDCNet_VGG16_classify(
        class_num, label_indice, load_weights=True).cuda()

    if os.path.exists(model_path):
        print("Adding Weights ....")
        all_state_dict = torch.load(model_path)
        net.load_state_dict(all_state_dict['net_state_dict'])
        net.eval()
    else:
        print("Can't find trained weights!!")
        exit()

    # first_frame = True
    while cap.isOpened():
        flag, image = cap.read()
        output_image = np.copy(image)
        if flag:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            break
        # if first_frame:
        #     image = cv2.selectROI("frame", image, False, False)
        image = cv2.resize(image, (512, 512))

        image = np.transpose(image, (2, 0, 1))
        image = torch.Tensor(image[None, :, :, :])

        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            features = net(image)
            div_res = net.resample(features)
            merge_res = net.parse_merge(div_res)
            outputs = merge_res['div'+str(net.args['div_times'])]
            del merge_res

        cv2.putText(output_image, "{}".format(int(outputs.sum().item())),
                    (30, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 3)

        cv2.imshow("frame", output_image)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            exit()

    cap.release()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("S-DCNet: Spatial Divide-and-Conquer Crowd Counting")
    parser.add_argument("model", type=str, help="Pretrained weights")
    parser.add_argument("--video", "-v", type=str, required=True, help="The video path to crowd count")
    args = parser.parse_args()

    main(args.model, args.video)
