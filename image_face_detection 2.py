import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque, defaultdict
import logging
import traceback
from datetime import datetime
import recognition_module as rg

CAMERA_INDEX = 0
# Enhanced error handling and debugging system
class SystemLogger:
    def __init__(self):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('object_detection_debug.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.error_count = 0
        self.max_errors = 50

    def log_info(self, message):
        self.logger.info(message)
        print(f"INFO: {message}")

    def log_error(self, message, exception=None):
        self.error_count += 1
        error_msg = f"ERROR #{self.error_count}: {message}"
        if exception:
            error_msg += f" - Exception: {str(exception)}"
            error_msg += f" - Traceback: {traceback.format_exc()}"
        self.logger.error(error_msg)
        print(f"🚨 {error_msg}")
        return self.error_count < self.max_errors

    def log_warning(self, message):
        self.logger.warning(message)
        print(f"⚠️ WARNING: {message}")

    def log_heartbeat(self, component):
        self.logger.debug(f"Heartbeat: {component} - {datetime.now()}")

# Enhanced object database with more detailed information
ENHANCED_OBJECT_DB = {
    # People 
    "person": {"width": 0.5, "min_width": 0.35, "max_width": 0.7, "priority": 1},
}

class RobustCameraCalibration:
    def __init__(self, logger):
        self.logger = logger
        self.focal_length = 650
        self.sensor_width = 6.17
        self.is_calibrated = False
        self.calibration_data = {}

    def auto_calibrate_focal_length(self, frame_width, frame_height):
        """Automatic focal length estimation with error handling"""
        try:
            # Typical focal lengths for different resolutions
            resolution_focal_map = {
                (640, 480): 500,
                (1280, 720): 650,
                (1920, 1080): 950,
                (3840, 2160): 1900
            }

            # Find closest resolution match
            current_res = (frame_width, frame_height)
            closest_res = min(resolution_focal_map.keys(),
                            key=lambda x: abs(x[0] - frame_width) + abs(x[1] - frame_height))

            # Scale focal length based on resolution difference
            scale_factor = frame_width / closest_res[0]
            self.focal_length = resolution_focal_map[closest_res] * scale_factor

            self.logger.log_info(f"Auto-calibrated focal length: {self.focal_length:.1f} for resolution {frame_width}x{frame_height}")
            return self.focal_length

        except Exception as e:
            self.logger.log_error("Error in auto_calibrate_focal_length", e)
            self.focal_length = 650  # Keep default
            return self.focal_length

# Enhanced object tracker with comprehensive error handling
class RobustObjectTracker:
    def __init__(self, logger, max_disappeared=10, max_distance=100):
        self.logger = logger
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, detection):
        """Register a new object with error handling"""
        try:
            self.objects[self.next_object_id] = {
                'centroid': detection.get('center', [0, 0]),
                'bbox': detection.get('bbox', [0, 0, 0, 0]),
                'class': detection.get('class', 'unknown'),
                'confidence': detection.get('confidence', 0.0),
                'distance': detection.get('distance', float('inf')),
                'priority': detection.get('priority', 3),
                'history': [detection.get('center', [0, 0])],
                'last_seen': time.time()
            }

            self.disappeared[self.next_object_id] = 0
            self.next_object_id += 1

        except Exception as e:
            self.logger.log_error("Error registering object", e)

    def deregister(self, object_id):
        """Remove an object from tracking with error handling"""
        try:
            if object_id in self.objects:
                del self.objects[object_id]
            if object_id in self.disappeared:
                del self.disappeared[object_id]
        except Exception as e:
            self.logger.log_error(f"Error deregistering object {object_id}", e)

    def update(self, detections):
        """Update tracked objects with comprehensive error handling"""
        try:
            # Handle empty detections
            if len(detections) == 0:
                for object_id in list(self.disappeared.keys()):
                    try:
                        self.disappeared[object_id] += 1
                        if self.disappeared[object_id] > self.max_disappeared:
                            self.deregister(object_id)
                    except Exception as e:
                        self.logger.log_error(f"Error updating disappeared object {object_id}", e)
                return self.objects

            if len(self.objects) == 0:
                for detection in detections:
                    self.register(detection)
                return self.objects

            # Get current object IDs (stable list)
            object_ids = list(self.objects.keys())

            try:
                object_centroids = []
                for obj_id in object_ids:
                    centroid = self.objects[obj_id].get('centroid', [0, 0])
                    if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
                        object_centroids.append(centroid[:2])
                    else:
                        object_centroids.append([0, 0])

                detection_centroids = []
                for detection in detections:
                    center = detection.get('center', [0, 0])
                    if isinstance(center, (list, tuple)) and len(center) >= 2:
                        detection_centroids.append(center[:2])
                    else:
                        detection_centroids.append([0, 0])

            except Exception as e:
                self.logger.log_error("Error extracting centroids", e)
                return self.objects

            # Verify we have valid data
            if len(object_centroids) == 0 or len(detection_centroids) == 0:
                return self.objects

            try:
                # Calculate distance matrix
                object_centroids_array = np.array(object_centroids)
                detection_centroids_array = np.array(detection_centroids)

                # Compute distance matrix
                D = np.linalg.norm(object_centroids_array[:, np.newaxis] -
                                 detection_centroids_array, axis=2)

                # Find minimum distance assignments
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                used_row_indices = set()
                used_col_indices = set()

                # Update existing objects with bounds checking
                for (row, col) in zip(rows, cols):
                    try:
                        if row in used_row_indices or col in used_col_indices:
                            continue

                        # Bounds checking
                        if row >= len(object_ids) or col >= len(detections):
                            continue

                        if D[row, col] > self.max_distance:
                            continue

                        # Get object ID safely
                        object_id = object_ids[row]
                        detection = detections[col]

                        # Verify object still exists
                        if object_id not in self.objects:
                            continue

                        # Update object with all fields
                        self.objects[object_id]['centroid'] = detection.get('center', [0, 0])
                        self.objects[object_id]['bbox'] = detection.get('bbox', [0, 0, 0, 0])
                        self.objects[object_id]['confidence'] = detection.get('confidence', 0.0)
                        self.objects[object_id]['distance'] = detection.get('distance', float('inf'))
                        self.objects[object_id]['priority'] = detection.get('priority', 3)
                        self.objects[object_id]['history'].append(detection.get('center', [0, 0]))
                        self.objects[object_id]['last_seen'] = time.time()

                        # Keep history limited
                        if len(self.objects[object_id]['history']) > 10:
                            self.objects[object_id]['history'] = self.objects[object_id]['history'][-10:]

                        self.disappeared[object_id] = 0

                        used_row_indices.add(row)
                        used_col_indices.add(col)

                    except Exception as e:
                        self.logger.log_error(f"Error updating object pair ({row}, {col})", e)
                        continue

                # Handle unmatched detections and objects
                unused_row_indices = set(range(0, len(object_ids))).difference(used_row_indices)
                unused_col_indices = set(range(0, len(detections))).difference(used_col_indices)

                # Mark unmatched existing objects as disappeared
                for row in unused_row_indices:
                    try:
                        if row < len(object_ids):
                            object_id = object_ids[row]
                            if object_id in self.disappeared:
                                self.disappeared[object_id] += 1
                                if self.disappeared[object_id] > self.max_disappeared:
                                    self.deregister(object_id)
                    except Exception as e:
                        self.logger.log_error(f"Error handling unmatched row {row}", e)

                # Register new objects
                for col in unused_col_indices:
                    try:
                        if col < len(detections):
                            self.register(detections[col])
                    except Exception as e:
                        self.logger.log_error(f"Error registering new object {col}", e)

            except Exception as e:
                self.logger.log_error("Error in distance matrix calculation", e)
                # Fallback: clear tracking and re-register all detections
                self.objects.clear()
                self.disappeared.clear()
                for detection in detections:
                    self.register(detection)

        except Exception as e:
            self.logger.log_error("Critical error in ObjectTracker.update", e)

        return self.objects

class RobustImagePreprocessor:
    def __init__(self, logger):
        self.logger = logger
        try:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        except Exception as e:
            self.logger.log_error("Error initializing CLAHE", e)
            self.clahe = None

    def enhance_frame(self, frame):
        """Enhanced frame preprocessing with error handling"""
        try:
            if self.clahe is None:
                return frame  # Return original if CLAHE failed to initialize

            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)

            # Apply CLAHE to L channel for better contrast
            l_channel = self.clahe.apply(l_channel)

            # Merge channels and convert back
            enhanced_lab = cv2.merge((l_channel, a, b))
            enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # Apply slight Gaussian blur to reduce noise
            enhanced_frame = cv2.GaussianBlur(enhanced_frame, (3, 3), 0)

            return enhanced_frame

        except Exception as e:
            self.logger.log_error("Error in frame enhancement", e)
            return frame  # Return original frame on error


# Main selective object detection system optimized for mobility
class SelectiveObjectDetectionSystem:
    def __init__(self):
        self.logger = SystemLogger()
        # self.logger.log_info("Initializing Selective Object Detection System with High-Quality TTS...")

        # Initialize system state
        self.running = False
        self.system_errors = 0
        self.max_system_errors = 20
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 10  # seconds

        try:
            self.setup_system()
        except Exception as e:
            self.logger.log_error("Critical error during system setup", e)
            raise

    def setup_system(self):
        """Initialize all system components with error handling"""
        try:
            # Initialize components with HIGH QUALITY TTS
            
            self.calibration = RobustCameraCalibration(self.logger)
            self.tracker = RobustObjectTracker(self.logger)
            self.preprocessor = RobustImagePreprocessor(self.logger)

            # Load enhanced YOLO model
            self.load_enhanced_model()

            # Detection parameters optimized for mobility
            self.min_distance = 0.1
            self.max_distance = 15.0
            self.warning_distance = 1.0  # Not used by selective TTS
            self.safety_distance = 0.5  # Not used by selective TTS

            # Dynamic confidence thresholding
            self.base_confidence = 0.3
            self.high_confidence = 0.6
            self.confidence_history = deque(maxlen=30)

            # Performance tracking
            self.detection_history = defaultdict(list)
            self.fps_history = deque(maxlen=30)

            # self.logger.log_info("High-Quality TTS system initialized successfully!")

        except Exception as e:
            self.logger.log_error("Error in setup_system", e)
            raise

    def load_enhanced_model(self):
        """Load optimized YOLO model with fallback options"""
        model_variants = ['yolov8m.pt', 'yolov8s.pt', 'yolov8n.pt']
        for i, model_name in enumerate(model_variants):
            try:
                self.model = YOLO(model_name)
                size_names = ['medium', 'small', 'nano']
                self.model_size = size_names[i]
                self.logger.log_info(f"YOLOv8 {self.model_size.title()} model loaded successfully!")
                return
            except Exception as e:
                self.logger.log_error(f"Failed to load {model_name}", e)
                if i == len(model_variants) - 1:
                    raise Exception("Failed to load any YOLO model")
                continue

    # # def calculate_dynamic_confidence(self, frame_detections):
    # #     """Dynamic confidence thresholding with error handling"""
    # #     try:
    # #         if not frame_detections:
    # #             return self.base_confidence

    # #         # Calculate scene complexity metrics
    # #         num_detections = len(frame_detections)
    # #         avg_confidence = sum(det.get('confidence', 0) for det in frame_detections) / num_detections

    # #         # Store in history
    # #         self.confidence_history.append(avg_confidence)

    # #         # Calculate adaptive threshold
    # #         if len(self.confidence_history) > 5:
    # #             recent_avg = sum(self.confidence_history) / len(self.confidence_history)

    # #             # Adjust threshold based on scene complexity
    # #             if num_detections > 10:  # Crowded scene
    # #                 dynamic_threshold = max(0.4, recent_avg * 0.8)
    # #             elif num_detections > 5:  # Moderate scene
    # #                 dynamic_threshold = max(0.35, recent_avg * 0.85)
    # #             else:  # Simple scene
    # #                 dynamic_threshold = max(0.3, recent_avg * 0.9)
    # #         else:
    # #             dynamic_threshold = self.base_confidence

    # #         return min(dynamic_threshold, 0.7)  # Cap at 0.7

    #     except Exception as e:
    #         self.logger.log_error("Error calculating dynamic confidence", e)
    #         return self.base_confidence

    def process_detections(self, results, frame_shape): # Currently using
        """Process YOLO detection results with comprehensive error handling"""
        detections = []
        try:
            frame_height, frame_width = frame_shape[:2]

            for result in results:
                try:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes:
                            try:
                                class_id = int(box.cls[0])
                                class_name = self.model.names[class_id]
                                confidence = float(box.conf[0])

                                if class_name in ENHANCED_OBJECT_DB:
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                                    # Calculate detection metrics
                                    bbox = [x1, y1, x2, y2]
                                    center = ((x1 + x2) / 2, (y1 + y2) / 2)

                                    detection = {
                                        'bbox': bbox,
                                        'center': center,
                                        'class': class_name,
                                        'confidence': confidence,
                                        'class_id': class_id,
                                        'priority': ENHANCED_OBJECT_DB[class_name]['priority']
                                    }

                                    detections.append(detection)

                            except Exception as e:
                                self.logger.log_error("Error processing individual detection", e)
                                continue

                except Exception as e:
                    self.logger.log_error("Error processing result", e)
                    continue

        except Exception as e:
            self.logger.log_error("Error in process_detections", e)

        return detections

    def draw_enhanced_visualization(self, frame, tracked_objects):
        """Enhanced visualization with high-quality TTS information"""
        try:
            display_frame = frame.copy()

            # Draw all tracked objects with special marking for announced ones
            for obj_id, obj in tracked_objects.items():
                try:
                    bbox = obj.get('bbox', [0, 0, 0, 0])
                    x1, y1, x2, y2 = map(int, bbox)

                    color = (0, 255, 0)
                    thickness = 1
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

                    # Draw trajectory if available
                    history = obj.get('history', [])
                    if len(history) > 1:
                        try:
                            points = np.array(history[-5:], np.int32)
                            cv2.polylines(display_frame, [points], False, color, 1)
                        except:
                            pass  # Skip trajectory on error

                except Exception as e:
                    self.logger.log_error(f"Error drawing object {obj_id}", e)
                    continue

            # Draw high-quality system information
            try:
                # Draw center reference
                center = (display_frame.shape[1] // 2, display_frame.shape[0] // 2)
                cv2.circle(display_frame, center, 8, (0, 255, 255), -1)

            except Exception as e:
                self.logger.log_error("Error drawing system info", e)

            return display_frame

        except Exception as e:
            self.logger.log_error("Error in draw_enhanced_visualization", e)
            return frame  # Return original frame on error

    def check_heartbeat(self):
        """System heartbeat check"""
        current_time = time.time()
        if current_time - self.last_heartbeat > self.heartbeat_interval:
            # self.logger.log_heartbeat("High-Quality TTS System")
            self.last_heartbeat = current_time

    def run(self):
        """Main execution loop optimized for high-quality announcements"""
        self.running = True
        # self.logger.log_info("Starting High-Quality TTS Object Detection System...")

        # Create display window with error handling
        try:
            cv2.namedWindow("High-Quality TTS Object Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("High-Quality TTS Object Detection", 1280, 720)
        except Exception as e:
            self.logger.log_error("Error creating display window", e)

        # Performance tracking
        image = cv2.imread("/Users/shreyassawant/mydrive/Shreyus_workspace/Semester_VII/CV/attendance_images/frame6.jpg")

        enhanced_frame = self.preprocessor.enhance_frame(image)
        results = self.model(enhanced_frame, verbose=False,
                                            conf=self.base_confidence, iou=0.5)
        detections = self.process_detections(results, image.shape)
        tracked_objects = self.tracker.update(detections)
        display_image = self.draw_enhanced_visualization(image, tracked_objects)
        cv2.imwrite("/Users/shreyassawant/mydrive/Shreyus_workspace/Semester_VII/CV/attendance_images/display_frame_6.png", display_image)
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cropped_img = display_image[int(y1):int(y2), int(x1):int(x2)]
            cv2.imshow("image", cropped_img)
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            rg.recognise(unknown_image=cropped_img)

        self.cleanup()

    def cleanup(self):
        """Comprehensive cleanup with error handling"""
        # self.logger.log_info("Starting high-quality TTS system cleanup...")
        self.running = False

        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                # self.logger.log_info("Camera released")
        except Exception as e:
            self.logger.log_error("Error releasing camera", e)

        try:
            cv2.destroyAllWindows()
            # self.logger.log_info("Windows destroyed")
        except Exception as e:
            self.logger.log_error("Error destroying windows", e)

        # self.logger.log_info("High-Quality TTS Object Detection System stopped")
        print("\n" + "="*60)
        # print("HIGH-QUALITY TTS SYSTEM SHUTDOWN COMPLETE")
        print(f"Total errors encountered: {self.system_errors}")
        print("Check 'object_detection_debug.log' for detailed logs")
        print("="*60)

# Entry point with comprehensive error handling
if __name__ == "__main__":
    system = None
    try:

        system = SelectiveObjectDetectionSystem()
        system.run()

    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user (Ctrl+C)")
        if system:
            system.cleanup()
    except Exception as e:
        print(f"\n🚨 CRITICAL SYSTEM ERROR: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        if system:
            try:
                system.cleanup()
            except:
                print("Cleanup also failed")
        print("\n📋 Check 'object_detection_debug.log' for complete error details")
    finally:
        # print("\n✅ High-Quality TTS Application terminated")
        pass