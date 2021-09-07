import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.plots import plot_one_box, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

import numpy as np
from collections import deque
from scipy.spatial import distance
import json


moving_threshold = 20
# res_area = [[(100,50),(200,400)],[(300,300),(600,400)]]
res_pts = []
mouse_pts = []
mouse = []
poly_t = []
counter = 0
poly_counter = 0


def mousePoint(event,x,y,flags,params):
    global counter, poly_counter
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pts.append((x,y))
        print(x,y)
        counter += 1
        poly_counter += 1
    if poly_counter % 4 == 0 and poly_counter > 0:
        poly_m = np.array([mouse_pts[counter - 4], mouse_pts[counter - 3], mouse_pts[counter - 2], mouse_pts[counter - 1]])
        poly_m = poly_m.reshape((-1, 1, 2))
        print(poly_m)
        mouse.append(poly_m)
        poly_counter = 0


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def detect(opt):
    global res_pts, poly_t, counter
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    pts = [deque(maxlen=30) for _ in range(9999)]       # tracker centre point memory
    timer = [0 for _ in range(9999)]                    # move timer

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset): # for each frame
        start_time = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # read config file
        f = open("cfg.json", "r")
        data = json.load(f)
        f.close()
        res_pts = data['points']        # list of list, [[x1,y1], [x2,y2]...]
        cfg_class = data['class']       # list, None for all classes
        tracking = data["tracking"]     # boolean
        vtimer = data["timer"]
        if len(cfg_class) == 0:
            cfg_class = None
        # print(opt.classes)

        # put res_pts into poly_t
        poly_t = []
        fours = len(res_pts) - len(res_pts) % 4
        for i in range(0, fours, 4):
            poly = np.array([res_pts[i], res_pts[i+1], res_pts[i+2], res_pts[i+3]])
            poly = poly.reshape((-1, 1, 2))
            poly_t.append(poly)


        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        # pred = non_max_suppression(
        #     pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=cfg_class, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()


        # Process detections
        for i, det in enumerate(pred):  # detections per frame
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            # parking box
            mask = np.zeros(im0.shape[:2], dtype="uint8")
            for poly in poly_t:
                cv2.polylines(im0, [poly], True, (255, 0, 0), 3)
                cv2.fillPoly(mask, [poly], 255)

            for poly in mouse:
                cv2.polylines(im0, [poly], True, (255, 0, 255), 3)
                cv2.fillPoly(mask, [poly], 255)

            for t in range(0, len(res_pts)):
                cv2.circle(im0, (res_pts[t][0], res_pts[t][1]), 3, (0, 255, 0), cv2.FILLED)
            # print(res_pts)

            for t in range(0, len(mouse_pts)):
                cv2.circle(im0, (mouse_pts[t][0], mouse_pts[t][1]), 3, (0, 0, 255), cv2.FILLED)


            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                if not tracking:
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        # if save_img or opt.save_crop or show_vid:  # Add bbox to image
                        if show_vid:
                            c = int(cls)  # integer class
                            # label = None if opt.hide_labels else (
                            #     names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            label = f'{names[c]} {conf:.2f}'

                            plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                         line_thickness=2)
                            # if opt.save_crop:
                            #     save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                else:
                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):

                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            c = int(cls)  # integer class
                            color = compute_color_for_id(id)
                            label = f'{id} {names[c]} {conf:.2f}'

                            center = (int(((bboxes[0]) + (bboxes[2])) / 2), int(((bboxes[1]) + (bboxes[3])) / 2))
                            thickness = 3
                            cv2.circle(im0, (center), 1, color, thickness)
                            pts[id].append(center)
                            # print(pts[id])

                            # draw motion path
                            for k in range(1, len(pts[id])):
                                if pts[id][k - 1] is None or pts[id][k] is None:
                                    continue
                                thickness = int(np.sqrt(64 / float(k + 1)) * 2)
                                cv2.line(im0, (pts[id][k - 1]), (pts[id][k]), (color), thickness)

                            # print(label)
                            if distance.euclidean(center, pts[id][int(len(pts[id])/2)]) > moving_threshold:
                                timer[id] = 0
                                label += " moving"
                            else:
                                timer[id] += 1
                                if timer[id] >= vtimer and mask[center[1]][center[0]]:
                                    label += " violation"
                                    color = (0, 0, 255)
                                # for area in res_area:
                                # for a in range(0, len(res_pts), 4):
                                #     # number of frames
                                #     # if timer[id] >= 200 and area[0][0] <= center[0] <= area[1][0] and area[0][1] <= center[1] <= area[1][1]:
                                #     if timer[id] >= 200 and mask[center[1]][center[0]]:
                                #         label += " violation"
                                #         color = (0,0,255)
                                if color != (0,0,255):
                                    label += " not moving"
                                # print(label)

                            plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)


                            if save_txt:
                                # to MOT format
                                bbox_top = output[0]
                                bbox_left = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                # Write MOT compliant results to file
                                with open(txt_path, 'a') as f:
                                   f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_top,
                                                               bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))

            cv2.putText(im0, "FPS: %f" % (1/(time.time() - start_time)), (int(20), int(40)), 0, 5e-3 * 200, (255, 255, 0), 3)


            # Stream results
            if show_vid:
                cv2.imshow(p, im0)
                cv2.setMouseCallback(p, mousePoint)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    # print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
