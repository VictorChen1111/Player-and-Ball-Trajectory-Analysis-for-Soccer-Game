import sys
import cv2
import os
from PyQt5.QtCore import Qt, QTimer, QSize, QProcess
from PyQt5.QtGui import QImage, QPixmap, QFont, qRgb
from PyQt5.QtWidgets import (
    QApplication, 
    QLabel, 
    QMainWindow,
    QVBoxLayout, 
    QHBoxLayout,
    QWidget,
    QPushButton,
    QStyle,
    QFileDialog,
    QDesktopWidget,
    QComboBox,
    QStatusBar,
    QCheckBox,
    QProgressBar,
)


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        ## INITIALIZE VARIABLE
        
        self.video_path = None
        self.cap_real = None
        self.cap_virtual = None
        self.playing = False
        self.virtual_ball_list = None
        self.fps = None
        self.first_real_frame = None
        self.real_frame_width = 900
        self.virtual_frame_width = 600
        self.trajectory_time = 0.35
        self.virtual_img = r'inference\green.jpg'
        self.output_video_path = None
        self.video_path_virtual = r'inference\output\virtual.mp4'
        # self.output_video_path = r'ncku_output.mp4'
        # self.video_path_virtual = r'virtual.mp4'
        # self.output_video_path = r'inference\output\ncku_output.mp4'
        self.frame_counter = 0


        ## SETTING INTERFACE

        self.setWindowTitle('Player-and-Ball-Trajectory-Analysis-for-Soccer-Game')
        self.resize(self.real_frame_width, self.real_frame_width)  # Set the initial window size
        self.move(10, 10)

        self.load_video_button = QPushButton('Load Video', self)
        self.load_video_button.setFont(QFont("Noto Sans"))
        self.load_video_button.clicked.connect(self.load_video)

        self.load_video_status_bar = QStatusBar()
        self.load_video_status_bar.showMessage("Ready")
        self.load_video_status_bar.setFont(QFont("Noto Sans", 8))
        self.load_video_status_bar.setFixedHeight(18)

        self.hsv_combobox = QComboBox(self)
        self.hsv_combobox.setFont(QFont("Noto Sans"))
        self.hsv_combobox.addItems(['HSV', 'HS'])
        self.hsv_combobox.setCurrentIndex(0)
        self.hsv_combobox.currentTextChanged.connect(self.hsv_changed)

        self.analyze_button = QPushButton('Analyze', self)
        self.analyze_button.setFont(QFont("Noto Sans"))
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.analyze_video)
        self.analyze_button.clicked.connect(self.start_progress)

        self.process = QProcess()  # command line for running main.py
        
        self.restart_button = QPushButton('Restart Video', self)
        self.restart_button.setFont(QFont("Noto Sans"))
        self.analyze_button.setEnabled(False)
        self.restart_button.clicked.connect(self.restart_video)

        self.clear_connect_button = QPushButton("Clear Connect")
        self.clear_connect_button.setFont(QFont("Noto Sans"))
        self.clear_connect_button.clicked.connect(self.clear_connect)

        self.show_ball_checkbox = QCheckBox('Show Ball Trajectory', self)
        self.show_ball_checkbox.setFont(QFont("Noto Sans"))

        self.frame_label = QLabel(self)
        self.frame_label.setAlignment(Qt.AlignCenter)
        placeholder_image = QImage(16, 9, QImage.Format_RGB888)
        placeholder_image.fill(qRgb(211, 211, 211))  # shoeing a gray picture as placeholder
        self.frame_label.setPixmap(QPixmap.fromImage(placeholder_image.scaledToWidth(self.real_frame_width)))
        
        self.play_button = QPushButton()
        self.play_button.setEnabled(False)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play)


        self.virtual_field_img_box = QLabel(self)
        self.virtual_field_img_box.setAlignment(Qt.AlignCenter)
        qpixmap_virtual_field_img_box = QPixmap.fromImage(QImage(self.virtual_img)).scaledToWidth(self.virtual_frame_width)
        self.virtual_field_img_box.setPixmap(qpixmap_virtual_field_img_box)

        
        load_layout = QVBoxLayout()
        load_layout.addWidget(self.load_video_button)
        load_layout.addWidget(self.load_video_status_bar)
        load_layout.addWidget(self.hsv_combobox)
        load_layout.addWidget(self.analyze_button)

        function_layout = QVBoxLayout()
        function_layout.addWidget(self.clear_connect_button)
        function_layout.addWidget(self.show_ball_checkbox)


        setting_layout = QVBoxLayout()
        setting_layout.addLayout(load_layout)
        setting_layout.addWidget(self.restart_button)
        setting_layout.addLayout(function_layout)

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.frame_label)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.virtual_field_img_box)

        layout = QHBoxLayout()
        layout.addLayout(setting_layout)
        layout.addLayout(control_layout)


        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

   
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_real_frame)
        self.timer.timeout.connect(self.update_virtual_frame)
        self.timer.start(10)  # Update frame every 33 milliseconds (30 fps)


    def load_video(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Video Files (*.mp4 *.flv *.ts *.mts *.avi)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.video_path = file_path
            self.load_video_status_bar.showMessage(f"{self.video_path}")
            self.load_video_status_bar.setFont(QFont("Noto Sans", 8))
            self.analyze_button.setEnabled(True)
            # self.update_output_path(file_path)
    
    def update_output_path(self, input_file_path):
        output_name = input_file_path.split('/')[-1]
        output_name = output_name.split('.')[0] + '_output.' + output_name.split('.')[-1]
        self.output_video_path = f'inference\output\{output_name}'
        print(f'Output Path: {self.output_video_path}')

    def hsv_changed(self, s):
        print(s)


    def analyze_video(self):  # execute 'python main.py --source self.video_path [--view] [--save]'
        executable = 'python'
        print(f'Analyze video: {self.video_path}')
        arguments = ['main.py', '--source', self.video_path, '--save']
        self.process.start(executable, arguments)


    def start_progress(self):  # wait until the process was done

        try:
            self.open_real_video()
            self.open_virtual_video()
            self.play_button.setEnabled(True)
            self.analyze_button.setEnabled(False)
        except ZeroDivisionError:
            pass

        
    def restart_video(self):

        real_rgb_image = cv2.cvtColor(self.first_real_frame, cv2.COLOR_BGR2RGB)
        h, w, _ = real_rgb_image.shape
        q_image = QImage(real_rgb_image.data, w, h, QImage.Format_RGB888)
        self.frame_label.setPixmap(QPixmap.fromImage(q_image).scaledToWidth(self.real_frame_width))

        qpixmap_virtual_field_img_box = QPixmap.fromImage(QImage(self.virtual_img)).scaledToWidth(self.virtual_frame_width)
        self.virtual_field_img_box.setPixmap(qpixmap_virtual_field_img_box)
        
        self.playing = False
        self.frame_counter = 0
        self.virtual_frame_counter = 0

        self.open_real_video()
        self.open_virtual_video()
        self.play_button.setEnabled(True)


    def open_real_video(self):

        self.cap_real = cv2.VideoCapture(self.output_video_path)
        self.fps = int(self.cap_real.get(cv2.CAP_PROP_FPS))
        self.timer.start(int(1000/self.fps-1))

        self.virtual_ball_list = open(r'./inference/output/virtual_ball_list.txt', "r")
        self.real_ball_list = open(r'./inference/output/ball_list.txt', "r")
        self.player_list = open(r'./inference/output/player_list.txt', "r")
        self.points = []  # save points to draw line on the temporary frame
        self.points_id = []  # save points id to draw line on future frame
        self.virtual_ball_trajectory_list = []
        self.real_ball_trajectory_list = []


    def open_virtual_video(self):
        self.cap_virtual = cv2.VideoCapture(self.video_path_virtual)

        if self.cap_virtual.isOpened():
            self.cap_virtual_width = self.cap_virtual.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.cap_virtual_height = self.cap_virtual.get(cv2.CAP_PROP_FRAME_HEIGHT)


    def update_real_frame(self):
        if self.cap_real is not None and self.playing:
            ret, self.real_frame = self.cap_real.read()
            if ret:
                # self.n_frame += 1  # store image
                # if(self.n_frame == 150):
                #     print(150)
                #     cv2.imshow('real_frame', self.real_frame)
                #     cv2.imwrite('inference\output.jpg', self.real_frame)
                self.original_height, self.original_width, _ = self.real_frame.shape
                self.real_frame_copy = self.real_frame.copy()
                rgb_image = cv2.cvtColor(self.real_frame, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb_image.shape
                q_image = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
                self.frame_label.setPixmap(QPixmap.fromImage(q_image).scaledToWidth(self.real_frame_width))
                self.frame_label.mousePressEvent = self.label_clicked

                ## update player list
                self.player_list_new_line = eval(self.player_list.readline())
                
                ## clear points list
                self.points.clear()

                ## connect each id
                self.connect_id()

                ## update ball trajectory
                if self.show_ball_checkbox.isChecked():
                    self.draw_real_ball_trajectory()
                else:
                    self.real_ball_trajectory_list.clear()
                if len(self.real_ball_trajectory_list) > self.trajectory_time * self.fps:
                    self.real_ball_trajectory_list.pop(0)
                
                new_line = eval(self.real_ball_list.readline())
                self.real_ball_trajectory_list.append(new_line)

                ## get the first frame
                if self.frame_counter == 0:
                    print("update first frame")
                    self.first_real_frame = self.real_frame.copy()
                self.frame_counter += 1




    def update_virtual_frame(self):

        if self.cap_virtual is not None and self.playing:
            ret, self.virtual_frame = self.cap_virtual.read()
            if ret:
                self.virtual_rgb_image = cv2.cvtColor(self.virtual_frame, cv2.COLOR_BGR2RGB)
                h, w, _ = self.virtual_rgb_image.shape
                q_image = QImage(self.virtual_rgb_image.data, w, h, QImage.Format_RGB888)
                self.virtual_field_img_box.setPixmap(QPixmap.fromImage(q_image).scaledToWidth(self.virtual_frame_width))
                
                self.virtual_transform_width = self.virtual_field_img_box.width()
                self.virtual_transform_height = self.virtual_field_img_box.height()

                ## update ball trajectory
                if self.show_ball_checkbox.isChecked():
                    self.draw_virtual_ball_trajectory()
                else:
                    self.virtual_ball_trajectory_list.clear()
                new_line = eval(self.virtual_ball_list.readline())
                self.virtual_ball_trajectory_list.append(new_line)
                if len(self.virtual_ball_trajectory_list) > self.trajectory_time * self.fps:
                    self.virtual_ball_trajectory_list.pop(0)
                

    def play(self):
        self.playing = not self.playing


    def draw_real_ball_trajectory(self):
        n_color = 255
        for b_t in reversed(self.real_ball_trajectory_list):
            x, y = b_t
            cv2.circle(self.real_frame,(int(x), int(y)),5,(0, 0, n_color),-1) # red
            if n_color >= 130:
                n_color -= 8
        self.rgb_image = cv2.cvtColor(self.real_frame, cv2.COLOR_BGR2RGB)
        h, w, _ = self.rgb_image.shape
        q_image = QImage(self.rgb_image.data, w, h, QImage.Format_RGB888)
        self.frame_label.setPixmap(QPixmap.fromImage(q_image).scaledToWidth(self.real_frame_width))


    def draw_virtual_ball_trajectory(self):
        n_color = 255
        for b_t in  reversed(self.virtual_ball_trajectory_list):
            x, y = b_t
            cv2.circle(self.virtual_rgb_image,(int(x), int(y)), 2, (n_color, 0, 0),-1) # red
            if n_color >= 130:
                n_color -= 8
        h, w, _ = self.virtual_rgb_image.shape
        q_image = QImage(self.virtual_rgb_image.data, w, h, QImage.Format_RGB888)
        self.virtual_field_img_box.setPixmap(QPixmap.fromImage(q_image).scaledToWidth(self.virtual_frame_width))


    def connect_id(self):
        player_list = self.player_list_new_line
        temp_points_list = []
        
        for i in self.points_id:
            for p in player_list:
                if p[4] == i:
                    center_x = (p[0] + p[2])/2
                    center_y = (p[1] + p[3])/2
                    temp_points_list.append([center_x, center_y])
        if (len(temp_points_list)):
            self.draw_line_between_points(temp_points_list)



    def label_clicked(self, event):
        if event.button() == Qt.LeftButton:
            x, y = event.x(), event.y()
            # Scale the coordinates to the original frame dimensions
            scaled_x = int(x * self.original_width / self.frame_label.width())
            scaled_y = int(y * self.original_height / self.frame_label.height())
            self.draw_point(scaled_x, scaled_y, (255, 255, 0))

            x, y = self.find_bbox(scaled_x, scaled_y)
            if(x > 0 or y > 0):
                self.draw_point(x, y, (255, 0, 255))
                if (len(self.points)):
                    self.draw_line_between_points(self.points)


    def find_bbox(self, x, y):
        player_list = self.player_list_new_line
        
        for i in range(len(player_list)):
            if (x > player_list[i][0]
            and x < player_list[i][2]
            and y > player_list[i][1]
            and y < player_list[i][3]):
                center_x = (player_list[i][0] + player_list[i][2])/2
                center_y = (player_list[i][1] + player_list[i][3])/2
                self.points.append([center_x, center_y])
                self.points_id.append(player_list[i][4])
                return center_x, center_y
        return -1, -1


    def draw_point(self, x, y, rgb:tuple):
        cv2.circle(self.real_frame,(int(x), int(y)),5,rgb,-1)
        rgb_image = cv2.cvtColor(self.real_frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_image.shape
        q_image = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
        self.frame_label.setPixmap(QPixmap.fromImage(q_image).scaledToWidth(self.real_frame_width))


    def draw_line_between_points(self, pts_ls):
        for i in range(len(pts_ls) - 1):
            p_1_x = int(pts_ls[i][0])
            p_1_y = int(pts_ls[i][1])
            p_2_x = int(pts_ls[i+1][0])
            p_2_y = int(pts_ls[i+1][1])
            cv2.line(self.real_frame, (p_1_x, p_1_y), (p_2_x, p_2_y), (255, 0, 255), 2)

            rgb_image = cv2.cvtColor(self.real_frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb_image.shape
            q_image = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
            self.frame_label.setPixmap(QPixmap.fromImage(q_image).scaledToWidth(self.real_frame_width))


    def clear_connect(self):
        self.points_id.clear()
        self.points.clear()
        self.real_frame = self.real_frame_copy
        rgb_image = cv2.cvtColor(self.real_frame_copy, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_image.shape
        q_image = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
        self.frame_label.setPixmap(QPixmap.fromImage(q_image).scaledToWidth(self.real_frame_width))


    def closeEvent(self, event):
        if self.cap_real is not None:
            self.cap_real.release()
        if self.virtual_ball_list is not None:
            self.virtual_ball_list.close()
            self.real_ball_list.close()
            self.player_list.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())