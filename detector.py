import cv2
import numpy as np
from retinaface import RetinaFace
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FaceForensicsDetector:
    def __init__(self, texture_threshold: int = 1200):
        self.texture_threshold = texture_threshold
        self.face_detector = RetinaFace

    def _calculate_texture(self, face_img: np.ndarray) -> float:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def detect(self, image_path: str) -> Dict:
        result = {
            "path": image_path,
            "faces": [],
            "is_ai_generated": False
        }
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"无法读取图片: {image_path}")
                return result
            faces = self.face_detector.detect_faces(image_path)
            if not faces:
                logger.info(f"未检测到人脸: {image_path}")
                return result
            for idx, (key, face_info) in enumerate(faces.items()):
                bbox = face_info["facial_area"]
                x1, y1, x2, y2 = bbox
                face_img = img[y1:y2, x1:x2]
                texture_var = self._calculate_texture(face_img)
                is_ai = texture_var < self.texture_threshold
                if is_ai:
                    result["is_ai_generated"] = True
                result["faces"].append({
                    "id": idx + 1,
                    "bbox": [x1, y1, x2, y2],
                    "texture_var": texture_var,
                    "is_ai": is_ai
                })
            logger.info(f"检测完成: {image_path}，发现 {len(faces)} 个人脸")
            return result
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return result