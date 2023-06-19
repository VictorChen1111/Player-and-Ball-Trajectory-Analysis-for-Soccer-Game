
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any

import cv2

import numpy as np
import torch

from sklearn.cluster import KMeans
import math

from sklearn.cluster import KMeans


# geometry utilities

@dataclass(frozen=True)
class Point:
    x: float
    y: float
    
    @property
    def int_xy_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def min_x(self) -> float:
        return self.x
    
    @property
    def min_y(self) -> float:
        return self.y
    
    @property
    def max_x(self) -> float:
        return self.x + self.width
    
    @property
    def max_y(self) -> float:
        return self.y + self.height
        
    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)
    
    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)

    @property
    def top_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y)

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height / 2)

    def pad(self, padding: float) -> 'Rect':
        return Rect(
            x=self.x - padding, 
            y=self.y - padding,
            width=self.width + 2*padding,
            height=self.height + 2*padding
        )
    
    def contains_point(self, point: Point) -> bool:
        return self.min_x < point.x < self.max_x and self.min_y < point.y < self.max_y


# detection utilities


@dataclass
class Detection:
    rect: Rect
    class_id: int
    class_name: str
    confidence: float
    tracker_id: Optional[int] = None
    team_id: int = -1
    

    @classmethod
    def from_results(cls, pred: np.ndarray, names: Dict[int, str]) -> List['Detection']:
        result = []
        for x_min, y_min, x_max, y_max, confidence, class_id in pred:
            class_id=int(class_id)
            result.append(Detection(
                rect=Rect(
                    x=float(x_min),
                    y=float(y_min),
                    width=float(x_max - x_min),
                    height=float(y_max - y_min)
                ),
                class_id=class_id,
                class_name=names[class_id],
                confidence=float(confidence)
            ))
        return result


def filter_detections_by_class(detections: List[Detection], class_name: str) -> List['Detection']:
    return [
        detection
        for detection 
        in detections
        if detection.class_name == class_name
    ]

def filter_detections_by_team(detections: List[Detection], team_name: int) -> List['Detection']:
    return [
        detection
        for detection 
        in detections
        if detection.team_id == team_name
    ]


# draw utilities


@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int
        
    @property
    def bgr_tuple(self) -> Tuple[int, int, int]:
        return self.b, self.g, self.r

    @classmethod
    def from_hex_string(cls, hex_string: str) -> 'Color':
        r, g, b = tuple(int(hex_string[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        return Color(r=r, g=g, b=b)


def draw_rect(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, thickness)
    return image


def draw_filled_rect(image: np.ndarray, rect: Rect, color: Color) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, -1)
    return image


def draw_polygon(image: np.ndarray, countour: np.ndarray, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.drawContours(image, [countour], 0, color.bgr_tuple, thickness)
    return image


def draw_filled_polygon(image: np.ndarray, countour: np.ndarray, color: Color) -> np.ndarray:
    cv2.drawContours(image, [countour], 0, color.bgr_tuple, -1)
    return image


def draw_text(image: np.ndarray, anchor: Point, text: str, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.putText(image, text, anchor.int_xy_tuple, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color.bgr_tuple, thickness, 2, False)
    return image


def draw_ellipse(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.ellipse(
        image,
        center=rect.bottom_center.int_xy_tuple,
        axes=(int(rect.width), int(0.35 * rect.width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color.bgr_tuple,
        thickness=thickness,
        lineType=cv2.LINE_4
    )
    return image


# base annotator
  

@dataclass
class BaseAnnotator:
    colors: List[Color]
    thickness: int

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            annotated_image = draw_ellipse(
                image=image,
                rect=detection.rect,
                color=self.colors[detection.class_id],
                thickness=self.thickness
            )
        return annotated_image



# black
MARKER_CONTOUR_COLOR_HEX = "000000"
MARKER_CONTOUR_COLOR = Color.from_hex_string(MARKER_CONTOUR_COLOR_HEX)
# MARKER_CONTOUR_COLOR = int(f"0x{MARKER_CONTOUR_COLOR_HEX}")


# red
PLAYER_MARKER_FILL_COLOR_HEX = "FF0000"
PLAYER_MARKER_FILL_COLOR = Color.from_hex_string(PLAYER_MARKER_FILL_COLOR_HEX)

# green
BALL_MERKER_FILL_COLOR_HEX = "00FF00"
BALL_MARKER_FILL_COLOR = Color.from_hex_string(BALL_MERKER_FILL_COLOR_HEX)

MARKER_CONTOUR_THICKNESS = 2
MARKER_WIDTH = 20
MARKER_HEIGHT = 20
MARKER_MARGIN = 10

# distance in pixels from the player's bounding box where we consider the ball is in his possession
PLAYER_IN_POSSESSION_PROXIMITY = 30

# calculates coordinates of possession marker
def calculate_marker(anchor: Point) -> np.ndarray:
    x, y = anchor.int_xy_tuple
    return(np.array([
        [x - MARKER_WIDTH // 2, y - MARKER_HEIGHT - MARKER_MARGIN],
        [x, y - MARKER_MARGIN],
        [x + MARKER_WIDTH // 2, y - MARKER_HEIGHT - MARKER_MARGIN]
    ]))


# draw single possession marker
def draw_marker(image: np.ndarray, anchor: Point, color: Color) -> np.ndarray:
    possession_marker_countour = calculate_marker(anchor=anchor)
    image = draw_filled_polygon(
        image=image, 
        countour=possession_marker_countour, 
        color=color)
    image = draw_polygon(
        image=image, 
        countour=possession_marker_countour, 
        color=MARKER_CONTOUR_COLOR,
        thickness=MARKER_CONTOUR_THICKNESS)
    return image


# dedicated annotator to draw possession markers on video frames
@dataclass
class MarkerAnntator:

    color: Color

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            annotated_image = draw_marker(
                image=image, 
                anchor=detection.rect.top_center,
                color=self.color)
        return annotated_image


def get_k_means_center(detections: List[Detection], frame):
    hsv_vectors = []
    IS_HSV = True
    # bgr_vectors = []
    player_detections = filter_detections_by_class(detections=detections, class_name="player")
    for player in player_detections:
        p = player.rect
        newImg = frame[int(p.min_y):int(p.max_y), int(p.min_x):int(p.max_x)]
        hsv = cv2.cvtColor(newImg, cv2.COLOR_BGR2HSV)

        
        if IS_HSV:
            # HSV
            avg_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        else:
            # HS
            hsv = np.delete(hsv, 2, axis=2)
            avg_hsv = np.mean(hsv.reshape(-1, 2), axis=0)

        hsv_vectors.append(avg_hsv)
        # bgr = cv2.split(newImg)
        # avg_bgr = np.mean(bgr.reshape(-1, 3), axis=0)
        # bgr_vectors.append(avg_bgr)
    hsv_array = np.array(hsv_vectors)

    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(hsv_array)

    return kmeans.cluster_centers_




global IS_HSV
IS_HSV = 1  # hsv:1, hs:0

def distance_3d(point1, point2):
    """Calculate the Euclidean distance between two points in 3D space."""
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    distance = (dx**2 + dy**2 + dz**2)
    # distance = math.sqrt(dx**2 + dy**2 + dz**2)
    return distance

def distance_2d(point1, point2):
    """Calculate the Euclidean distance between two points in 3D space."""
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    distance = (dx**2 + dy**2)
    # distance = math.sqrt(dx**2 + dy**2 + dz**2)
    return distance

CLUSTER_CENTERS = None  # inti cluster centers as global
def k_add_team_id(detections: List[Detection], medians_cluster_center, frame):
    global CLUSTER_CENTERS

    hsv_vectors = []
    # bgr_vectors = []
    player_detections = filter_detections_by_class(detections=detections, class_name="player")
    for player in player_detections:
        p = player.rect
        newImg = frame[int(p.min_y):int(p.max_y), int(p.min_x):int(p.max_x)]
        hsv = cv2.cvtColor(newImg, cv2.COLOR_BGR2HSV)

        if IS_HSV:
            # HSV
            avg_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        else:
            # HS
            hsv = np.delete(hsv, 2, axis=2)
            avg_hsv = np.mean(hsv.reshape(-1, 2), axis=0)

        hsv_vectors.append(avg_hsv)

    hsv_array = np.array(hsv_vectors)

    for i in range(len(hsv_array)):
        if IS_HSV: # HSV
            if (distance_3d(medians_cluster_center[0], hsv_array[i]) < distance_3d(medians_cluster_center[1], hsv_array[i])):
                player_detections[i].team_id = 0

            else:
                player_detections[i].team_id = 1

        else: # HS
            if (distance_2d(medians_cluster_center[0], hsv_array[i]) < distance_2d(medians_cluster_center[1], hsv_array[i])):
                player_detections[i].team_id = 0

            else:
                player_detections[i].team_id = 1
            