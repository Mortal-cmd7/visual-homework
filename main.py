import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

# 全局常量与样式配置 
MODEL_PATH = "yolo26n-pose.pt"
DEFAULT_DEVICE = "cpu"
DEFAULT_CONF_THRESH = 0.3
DEFAULT_IOU_THRESH = 0.3
MAX_DET = 20
TRACKER_CONFIG = "bytetrack.yaml"

# UI 样式
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
LINE_THICKNESS = 2
TEXT_BG_ALPHA = 0.6
TRACK_LEN = 30

# COCO 姿态关键点连接关系
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]


@dataclass
class TargetInfo:
    track_id: Optional[int] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    distance: Optional[float] = None


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int
    class_id: int
    confidence: float
    keypoints: Optional[np.ndarray] = None   # (17, 3) 格式


class CalibrationParams:
    def __init__(self, ref_dist_m: float = 2.0, ref_height_px: int = 200):
        self.ref_dist_m = ref_dist_m
        self.ref_height_px = ref_height_px

    def update(self, real_dist_m: float, bbox_height_px: int):
        self.ref_dist_m = real_dist_m
        self.ref_height_px = bbox_height_px

    def estimate_distance(self, bbox_height_px: int) -> Optional[float]:
        if bbox_height_px <= 0 or self.ref_height_px <= 0:
            return None
        return self.ref_dist_m * (self.ref_height_px / bbox_height_px)


class TrackingUI:
    def __init__(self, class_names: Dict[int, str]):
        self.class_names = class_names

    @staticmethod
    def draw_text_with_background(
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        text_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        alpha: float = TEXT_BG_ALPHA
    ) -> None:

        (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        x, y = pos
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y - th - baseline), (x + tw, y + baseline), bg_color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.putText(img, text, (x, y), FONT, FONT_SCALE, text_color, FONT_THICKNESS)

    def draw_info_panel(
        self,
        frame: np.ndarray,
        target: TargetInfo,
        fps: float,
        same_class_count: int,
        is_paused: bool = False,
        is_calibrating: bool = False
    ) -> None:

        h, w = frame.shape[:2]
        panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        if fps > 0:
            fps_text = f"FPS: {fps:.1f}"
            self.draw_text_with_background(frame, fps_text, (10, 25),
                                           text_color=(0, 255, 0), bg_color=(0, 0, 0))
        if is_paused:
            self.draw_text_with_background(frame, "PAUSED", (w - 120, 25),
                                           text_color=(0, 255, 255), bg_color=(0, 0, 0))
        if is_calibrating:
            self.draw_text_with_background(frame, "CALIBRATION MODE", (w - 200, 55),
                                           text_color=(0, 255, 255), bg_color=(0, 0, 255))

        if target.track_id is not None:
            info = f"Tracking ID: {target.track_id} | Class: {target.class_name or 'N/A'}"
            if target.distance is not None:
                info += f" | Distance: {target.distance:.1f}m"
            info += f" | Same class: {same_class_count}"
            self.draw_text_with_background(frame, info, (10, 55),
                                           text_color=(255, 255, 0), bg_color=(0, 0, 0))
        else:
            hint = "Click on object to select | c: clear | p: pause | s: screenshot | k: calibrate | q: quit"
            self.draw_text_with_background(frame, hint, (10, 55),
                                           text_color=(200, 200, 200), bg_color=(0, 0, 0))

    def draw_distance_on_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        distance: Optional[float],
        color: Tuple[int, int, int]
    ) -> None:

        if distance is None or np.isinf(distance):
            text = "Dist: N/A"
        else:
            text = f"{distance:.1f}m"
        x1, y1, x2, y2 = bbox
        text_y = y1 - 5
        if text_y - 20 < 0:
            text_y = y1 + 20
        self.draw_text_with_background(frame, text, (x1, text_y),
                                       text_color=(0, 255, 255), bg_color=color)


class VideoProcessor:
    def __init__(
        self,
        source: str = "0",
        view_img: bool = True,
        save_video: bool = False,
        video_output_path: str = "output.avi",
        conf_thres: float = DEFAULT_CONF_THRESH,
        iou_thres: float = DEFAULT_IOU_THRESH,
        device: str = DEFAULT_DEVICE,
        show_fps: bool = True,
        show_conf: bool = False,
        track_len: int = TRACK_LEN,
    ):
        self.source = source
        self.view_img = view_img
        self.save_video = save_video
        self.video_output_path = video_output_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.show_fps = show_fps
        self.show_conf = show_conf
        self.track_len = track_len

        # 模型与数据
        self.model: Optional[YOLO] = None
        self.class_names: Dict[int, str] = {}
        self.calib_params = CalibrationParams()

        # 状态变量
        self.selected_target = TargetInfo()
        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.latest_detections: List[Detection] = []

        self.paused = False
        self.calibration_mode = False
        self.frame_count = 0
        self.fps = 0.0

        self.ui: Optional[TrackingUI] = None
        self._last_time = time.perf_counter()

    def init_model(self) -> None:
        LOGGER.info(f"Loading pose model {MODEL_PATH} on {self.device}")
        if not Path(MODEL_PATH).exists():
            LOGGER.warning(f"Model file {MODEL_PATH} not found locally. Attempting to download...")
        self.model = YOLO(MODEL_PATH)
        if self.device != "cpu":
            self.model.to(self.device)
        self.class_names = self.model.names
        self.ui = TrackingUI(self.class_names)

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.paused:
            return
        if not self.latest_detections:
            return

        # 寻找鼠标点击命中的最小边界框（避免选中大框内的重叠小物体）
        best_match = None
        min_area = float('inf')
        for det in self.latest_detections:
            if det.x1 <= x <= det.x2 and det.y1 <= y <= det.y2:
                area = (det.x2 - det.x1) * (det.y2 - det.y1)
                if area < min_area:
                    min_area = area
                    best_match = det

        if best_match is None:
            return

        if self.calibration_mode:
            self._handle_calibration(best_match)
        else:
            self._handle_selection(best_match)

    def _handle_selection(self, det: Detection) -> None:
        self.selected_target.track_id = det.track_id
        self.selected_target.class_id = det.class_id
        self.selected_target.class_name = self.class_names[det.class_id]
        LOGGER.info(f"Selected target ID={det.track_id}, Class={self.selected_target.class_name}")

    def _handle_calibration(self, det: Detection) -> None:
        bbox_height = det.y2 - det.y1
        class_name = self.class_names[det.class_id]
        print(f"\n[标定模式] 选中目标：ID={det.track_id}, 类别={class_name}, 像素高度={bbox_height}px")
        try:
            real_dist = float(input("请输入该物体到摄像头的实际距离（单位：米）：").strip())
            if real_dist <= 0:
                print("距离必须大于0，标定取消。")
                self.calibration_mode = False
                return
        except ValueError:
            print("输入无效，标定取消。")
            self.calibration_mode = False
            return

        self.calib_params.update(real_dist, bbox_height)
        print(f" 标定完成，参考距离 = {self.calib_params.ref_dist_m}m, "
              f"参考像素高度 = {self.calib_params.ref_height_px}px")
        self.calibration_mode = False

    @staticmethod
    def get_bbox_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
        return (x1 + x2) // 2, (y1 + y2) // 2

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        #处理单帧图像，返回标注后的帧
        if self.paused:
            return frame

        # 执行跟踪推理
        try:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=MAX_DET,
                tracker=TRACKER_CONFIG,
                verbose=False
            )
        except FileNotFoundError:
            LOGGER.warning("Tracker config file not found, using default tracker.")
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=MAX_DET,
                verbose=False
            )

        # 解析检测结果
        self.latest_detections = self._parse_results(results)

        # 统计同类别数量
        same_class_count = 0
        if self.selected_target.track_id is not None:
            selected_class = self.selected_target.class_id
            same_class_count = sum(1 for d in self.latest_detections if d.class_id == selected_class)

        # 绘制所有检测与轨迹
        annotator = Annotator(frame, line_width=LINE_THICKNESS)
        self.selected_target.bbox = None
        self.selected_target.distance = None

        for det in self.latest_detections:
            color = colors(det.class_id, True)
            is_selected = (self.selected_target.track_id is not None and
                           det.track_id == self.selected_target.track_id)

            # 绘制边界框与标签
            label = f"{self.class_names[det.class_id]} ID:{det.track_id}"
            if self.show_conf:
                label += f" {det.confidence:.2f}"
            annotator.box_label([det.x1, det.y1, det.x2, det.y2], label, color=color)

            # 选中目标的特殊处理
            if is_selected:
                cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 3)

                bbox_height = det.y2 - det.y1
                dist = self.calib_params.estimate_distance(bbox_height)
                self.selected_target.distance = dist
                self.selected_target.bbox = (det.x1, det.y1, det.x2, det.y2)

                self.ui.draw_distance_on_box(frame, (det.x1, det.y1, det.x2, det.y2), dist, color)

                if det.keypoints is not None:
                    self.draw_skeleton(frame, det.keypoints, color=color)

                # 轨迹更新与绘制
                center = self.get_bbox_center(det.x1, det.y1, det.x2, det.y2)
                self.track_history[det.track_id].append(center)
                if len(self.track_history[det.track_id]) > self.track_len:
                    self.track_history[det.track_id].pop(0)
                if len(self.track_history[det.track_id]) > 1:
                    points = np.array(self.track_history[det.track_id], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

        # 计算 FPS
        self._update_fps()

        # 绘制顶部信息面板
        self.ui.draw_info_panel(
            frame, self.selected_target, self.fps if self.show_fps else 0.0,
            same_class_count, self.paused, self.calibration_mode
        )

        return frame

    def _parse_results(self, results) -> List[Detection]:
        detections = []
        if results[0].boxes is None:
            return detections

        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        track_ids = (results[0].boxes.id.int().cpu().numpy()
                     if results[0].boxes.id is not None
                     else [-1] * len(boxes))

        # 获取关键点数据
        kpts_array = None
        if results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
            kpts_array = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)

        for idx, (box, tid, cid, conf) in enumerate(zip(boxes, track_ids, class_ids, confs)):
            if tid < 0:   # 忽略无效跟踪ID
                continue
            x1, y1, x2, y2 = map(int, box[:4])
            kpts = kpts_array[idx] if kpts_array is not None and idx < len(kpts_array) else None
            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                track_id=tid, class_id=cid, confidence=conf,
                keypoints=kpts
            ))
        return detections

    def _update_fps(self) -> None:
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            now = time.perf_counter()
            self.fps = 10.0 / (now - self._last_time)
            self._last_time = now

    @staticmethod
    def draw_skeleton(frame: np.ndarray, kpts: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> None:
        # 绘制关键点
        for x, y, conf in kpts:
            if conf > 0.5:
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
        # 绘制骨架连线
        for i, j in SKELETON:
            if kpts[i][2] > 0.5 and kpts[j][2] > 0.5:
                pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                pt2 = (int(kpts[j][0]), int(kpts[j][1]))
                cv2.line(frame, pt1, pt2, color, 2)

    def run(self) -> None:
        self.init_model()

        # 打开视频源
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {src}")

        # 视频写入器
        video_writer = None
        if self.save_video:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(self.video_output_path, fourcc, fps, (w, h))

        window_name = "YOLO Pose Tracking + Distance"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        self._last_time = time.perf_counter()

        while cap.isOpened():
            if not self.paused:
                success, frame = cap.read()
                if not success:
                    LOGGER.info("Video ended")
                    break
                frame = self.process_frame(frame)

            if self.view_img:
                cv2.imshow(window_name, frame)

            if self.save_video and video_writer is not None and not self.paused:
                video_writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.selected_target = TargetInfo()
                self.track_history.clear()
                LOGGER.info("Cleared selected target")
            elif key == ord('r'):
                self.track_history.clear()
                LOGGER.info("Cleared all trajectories")
            elif key == ord('p'):
                self.paused = not self.paused
                LOGGER.info(f"Paused: {self.paused}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"screenshot_{timestamp}.png"
                cv2.imwrite(screenshot_path, frame)
                LOGGER.info(f"Screenshot saved to {screenshot_path}")
            elif key == ord('k'):
                self.calibration_mode = not self.calibration_mode
                if self.calibration_mode:
                    LOGGER.info("进入标定模式：点击一个物体并输入实际距离")
                else:
                    LOGGER.info("退出标定模式")

        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser(description="YOLO Pose Tracking with Distance Estimation")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (file path, camera index, or RTSP URL)")
    parser.add_argument("--view-img", action="store_true", default=True,
                        help="Display results in a window")
    parser.add_argument("--no-view-img", action="store_false", dest="view_img",
                        help="Do not display results")
    parser.add_argument("--save-video", action="store_true", default=False,
                        help="Save output video")
    parser.add_argument("--output", type=str, default="tracking_output.avi",
                        help="Output video path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESH,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESH,
                        help="IoU threshold")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help="Inference device (cpu, cuda:0, mps, etc.)")
    parser.add_argument("--no-fps", action="store_false", dest="show_fps",
                        help="Hide FPS display")
    parser.add_argument("--show-conf", action="store_true", default=False,
                        help="Show confidence score on labels")
    parser.add_argument("--track-len", type=int, default=TRACK_LEN,
                        help="Maximum length of trajectory history")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    processor = VideoProcessor(
        source=opt.source,
        view_img=opt.view_img,
        save_video=opt.save_video,
        video_output_path=opt.output,
        conf_thres=opt.conf,
        iou_thres=opt.iou,
        device=opt.device,
        show_fps=opt.show_fps,
        show_conf=opt.show_conf,
        track_len=opt.track_len,
    )
    processor.run()
