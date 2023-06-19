# from elements.yolo import YOLO
from elements.deep_sort import DEEPSORT
from elements.perspective_transform import Perspective_Transform
from elements.assets import transform_matrix, detect_color
from arguments import Arguments
# from yolov5.utils.plots import plot_one_box
from utils.plots import plot_one_box
from utils.general import scale_coords
from models.player import detect_player
from models.ball2 import detect_ball
from optimization import interpolate_coordinates, optimize_M
import torch
import os
import cv2
import numpy as np
import sys

from util import (
    Color, 
    Detection, 
    MarkerAnntator, 
    BALL_MARKER_FILL_COLOR, 
    PLAYER_MARKER_FILL_COLOR, 
    filter_detections_by_class, 
    filter_detections_by_team, 
    get_k_means_center, 
    k_add_team_id
)

def save_list_to_txt(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')

bar_width = 400
bar_height = 30

def create_progress_bar_window(window_name, bar_width, bar_height):
    progress_bar = np.zeros((bar_height, bar_width, 3), np.uint8)
    progress_bar[:] = (255, 255, 255)

    frame_text = 'Wait for setting up!'
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(frame_text, font, 1, 2)[0]
    textX = int((bar_width - textsize[0]) / 2)
    textY = int((bar_height + textsize[1]) / 2)

    cv2.putText(progress_bar, frame_text, (textX, textY), font, 0.7, (0, 0, 0), 2)

    cv2.namedWindow(window_name)
    cv2.imshow(window_name, progress_bar)
    cv2.waitKey(1)

def update_progress_bar(window_name, current_frame, total_frames, bar_width):
    # Calculate the progress percentage
    progress = int((current_frame / total_frames) * bar_width)

    # Create a black image with a white progress bar and frame number
    progress_bar = np.zeros((30, bar_width, 3), np.uint8)
    progress_bar[:] = (255, 255, 255)

    # Update the progress bar with the current progress
    progress_bar[:, :progress] = (3, 230, 18)

    # Add the frame number to the image
    frame_text = f'Loading: {current_frame}/{int(total_frames)}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(frame_text, font, 1, 2)[0]
    textX = int((bar_width - textsize[0]) / 2)
    textY = int((bar_height + textsize[1]) / 2)
    cv2.putText(progress_bar, frame_text, (textX, textY), font, 0.7, (0, 0, 0), 2)

    # Display the updated progress bar
    cv2.imshow(window_name, progress_bar)
    cv2.waitKey(1)


def main(opt):
    
    create_progress_bar_window('Progress Bar', bar_width, bar_height)
    cv2.setWindowProperty("Progress Bar", cv2.WND_PROP_TOPMOST, 1)
    # Load models
    # detector = YOLO(opt.yolov5_model, opt.conf_thresh, opt.iou_thresh)
    deep_sort = DEEPSORT(opt.deepsort_config)
    perspective_transform = Perspective_Transform()

    # Video capture
    cap = cv2.VideoCapture(opt.source)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    opt.outputfps = FPS
    torch.cuda.empty_cache()
    # Save output
    if opt.save:
        output_name = opt.source.split('/')[-1]
        output_name = output_name.split('.')[0] + '_output.' + output_name.split('.')[-1]

        output_path = os.path.join(os.getcwd(), 'inference/output')
        os.makedirs(output_path, exist_ok=True)
        output_name = os.path.join(os.getcwd(), 'inference/output', output_name)

        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        out = cv2.VideoWriter(output_name,  
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                opt.outputfps, (int(w), int(h)))


    frame_num = 0

    # Green Image (Soccer Field)
    bg_ratio = int(np.ceil(w/(3*115)))
    gt_img = cv2.imread('./inference/green.jpg')
    gt_img = cv2.resize(gt_img,(115*bg_ratio, 74*bg_ratio))
    gt_h, gt_w, _ = gt_img.shape  # 296, 460
    


    if opt.save:
        output_name = 'virtual.mp4'
        output_path = os.path.join(os.getcwd(), 'inference/output')
        os.makedirs(output_path, exist_ok=True)
        output_name = os.path.join(os.getcwd(), 'inference/output', output_name)
        out_virtual = cv2.VideoWriter(output_name,  
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                opt.outputfps, (int(gt_w), int(gt_h)))




    f_track = False
    count = 0
    ball_xyxy = [w/2, h/4, w/2, h/4]
    n_ball = 0

    countdown_frame = 2*int(FPS)
    cluster_center_ls = []

    player_list = []
    ball_list = []
    virtual_player_list = []
    M_list = []


    # Create the progress bar window
    # create_progress_bar_window('Progress Bar', bar_width, bar_height)

    while(cap.isOpened()): #  and frame_num < 20
        ret, frame = cap.read()
        bg_img = gt_img.copy()
        if ret:
            main_frame = frame.copy()
            pred = detect_player(main_frame)
            f_track, ball_xyxy ,count, haveball = detect_ball(main_frame, f_track, ball_xyxy ,count)
            # Output: Homography Matrix and Warped image 
            if frame_num % 1 == 0: # Calculate the homography matrix every 5 frames
                M1, warped_image = perspective_transform.homography_matrix(frame)

            l_t = transform_matrix(M1, (0, 0), (h, w), (gt_h, gt_w))
            l_d = transform_matrix(M1, (0, h-1), (h, w), (gt_h, gt_w))
            r_t = transform_matrix(M1, (w-1, 0), (h, w), (gt_h, gt_w))
            r_d = transform_matrix(M1, (w-1, h-1), (h, w), (gt_h, gt_w))
            l_m = transform_matrix(M1, (0, h/2), (h, w), (gt_h, gt_w))
            r_m = transform_matrix(M1, (w-1, h/2), (h, w), (gt_h, gt_w))

            flag = True
            if (r_t[0] - l_t[0]) < (r_d[0] - l_d[0]):
                flag = False
            if l_m[1] < 0 or l_m[1] > h or r_m[1] < 0 or r_m[1] > h:
                flag = False
            if frame_num == 0:
                flag = True
            if flag:
                M = M1
            else:
                print("fail M")
            M_list.append(M)

            yoloOutput = pred[0] # yoloOutput : [[x1, y1, x2, y2, conf, cls], ...]

            detections = Detection.from_results(
                # pred=yoloOutput.numpy(), 
                pred=yoloOutput, 
                names=['ball', 'goalkeeper', 'player', 'referee'])
            player_detections = detections
            ##### CLASSIFY TEAM #####
            # use the top N frames to get the median cluster center and classify the team
            if countdown_frame > 0:
                k_mean_center = get_k_means_center(player_detections, frame)
                cluster_center_ls.append(k_mean_center)
                countdown_frame -= 1

            elif countdown_frame == 0:
                medians_cluster_center = np.median(cluster_center_ls, axis=0)
                countdown_frame -= 1
            else: 
                k_add_team_id(player_detections, medians_cluster_center, frame)
            v_p_list = []
            if len(player_detections): # [[x1, y1, x2, y2, conf, cls, team_id, player_id], ...]
                
                # Tracking
                output_from_deepsort = deep_sort.detection_to_deepsort(player_detections, frame)
                # p_list.append(output_from_deepsort)
                if (type(output_from_deepsort) == list):
                    a = output_from_deepsort
                else:
                    a = output_from_deepsort.tolist()
                player_list.append(a)

                for player in (player_detections[::-1]): # det(conf:high->low)
                    color = [(255, 0, 0), (0, 255, 0)] # b g r
                    Green = Color(0, 255, 0)
                    Blue = Color(0, 0, 255)
                    team1_marker_annotator = MarkerAnntator(color=Green)
                    team2_marker_annotator = MarkerAnntator(color=Blue)
                    frame = team1_marker_annotator.annotate(
                        image=frame, 
                        detections = filter_detections_by_team(detections=player_detections, team_name=1))
                    frame = team2_marker_annotator.annotate(
                        image=frame, 
                        detections = filter_detections_by_team(detections=player_detections, team_name=0))

                    x_center, y_center = player.rect.bottom_center.int_xy_tuple
                    # xyxy = [i.rect.min_x, i.rect.min_y, i.rect.max_x, i.rect.max_y]
                    # plot_one_box(xyxy, frame, (0, 0, 255), label="player")
                    # v_p_list.append(v_p_l)
                    v_p_list.append([x_center, y_center, player.team_id])
            else:
                deep_sort.deepsort.increment_ages()
                v_p_list.append([])

            b_list = []
            if haveball:
                ball_x_center = int((ball_xyxy[0] + ball_xyxy[2])/2)
                ball_y_center = int((ball_xyxy[1] + ball_xyxy[3])/2)
                b_list = [ball_x_center, ball_y_center]
                n_ball += 1
                plot_one_box(ball_xyxy, frame, (0, 0, 255), label=None, line_thickness=2)
                # if f_track == True:
                #     plot_one_box(ball_xyxy, frame, (0, 0, 255), label=None, line_thickness=2)
                # else:
                #     plot_one_box(ball_xyxy, frame, (255, 0, 255), label=None, line_thickness=2)
            else:
                print("fail ball")
                
            ball_list.append(b_list)
            virtual_player_list.append(v_p_list)
            if opt.view:
                cv2.imshow('frame',frame)
                cv2.imshow('virtual frame',bg_img)
                if cv2.waitKey(1) & ord('q') == 0xFF:
                    break

            # Saving the output
            if opt.save:
                out.write(frame)
                # out_virtual.write(bg_img)

            frame_num += 1

            # Update the progress bar
            if (frame_num == frame_count):
                cv2. destroyAllWindows()
            else:
                update_progress_bar('Progress Bar', frame_num, frame_count, bar_width)
            
            if(frame_num == 20):
                break

        else:
            break

        sys.stdout.write(
            "\r[Input Video : %s] [%d/%d Frames Processed]"
            % (
                opt.source,
                frame_num,
                frame_count,
            )
        )

        # if cv2.waitKey(10) & 0xFF == 27:
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     break
    
    new_virtual_ball_list = []
    ball_list = interpolate_coordinates(ball_list)
    M_list = optimize_M(M_list, (w/2, h/2), (h, w), (gt_h, gt_w))
    t_list = ball_list.copy()
    for xy , M in zip(t_list, M_list):
        x, y = xy
        if len(xy) == 0:
            new_virtual_ball_list.append([])
        else:
            coords = transform_matrix(M, (x, y), (h, w), (gt_h, gt_w))
            new_virtual_ball_list.append([int(coords[0]), int(coords[1])])
            
    new_virtual_player_list = []
    for n , M in zip(virtual_player_list, M_list):
        temp_p = []
        for xyt in n:
            x, y, t = xyt
            if len(xyt) == 0:
                temp_p.append([])
            coords = transform_matrix(M, (x, y), (h, w), (gt_h, gt_w))
            temp_p.append([int(coords[0]), int(coords[1]), t])
        new_virtual_player_list.append(temp_p)
    if opt.save:
        color = [(255, 0, 0), (0, 255, 0)] # b g r
        for bxy, player ,M in zip(new_virtual_ball_list, new_virtual_player_list, M_list):
            bg_img = gt_img.copy()
            # l_t = transform_matrix(M, (0, 0), (h, w), (gt_h, gt_w))
            # l_d = transform_matrix(M, (0, h-1), (h, w), (gt_h, gt_w))
            # cv2.line(bg_img, l_d, l_t, (0, 0, 0), 2)
            # r_t = transform_matrix(M, (w-1, 0), (h, w), (gt_h, gt_w))
            # r_d = transform_matrix(M, (w-1, h-1), (h, w), (gt_h, gt_w))
            # cv2.line(bg_img, r_d, r_t, (0, 0, 0), 2)
            cv2.circle(bg_img, (bxy[0], bxy[1]), bg_ratio + 1, (0, 0, 255), -1)
            for pxyt in player:
                cv2.circle(bg_img, (pxyt[0], pxyt[1]), bg_ratio + 1, color[pxyt[2]], -1)
            out_virtual.write(bg_img)
    print('n_ball:', n_ball, '/', frame_num)
    save_list_to_txt(player_list, r'./inference/output/player_list.txt')
    save_list_to_txt(ball_list, r'./inference/output/ball_list.txt')
    save_list_to_txt(new_virtual_player_list, r'./inference/output/virtual_player_list.txt')
    save_list_to_txt(new_virtual_ball_list, r'./inference/output/virtual_ball_list.txt')
    
    if opt.save:
        print(f'\n\nOutput video has been saved in {output_path}!')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    opt = Arguments().parse()
    with torch.no_grad():
        main(opt)

# conda activate footballproject
# python main.py --source ncku.mp4 --view --save
