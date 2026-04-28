import os
import csv
from detector import FaceForensicsDetector
from config import TEXTURE_THRESHOLD
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def batch_detect(input_dir: str, output_csv: str = "detect_results.csv"):
    detector = FaceForensicsDetector(TEXTURE_THRESHOLD)
    supported_exts = (".jpg", ".jpeg", ".png", ".bmp")
    results = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_exts):
            img_path = os.path.join(input_dir, filename)
            res = detector.detect(img_path)
            results.append(res)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["图片路径", "人脸数", "是否含AI脸", "人脸ID", "纹理方差", "是否AI"])
        for res in results:
            face_count = len(res["faces"])
            has_ai = res["is_ai_generated"]
            for face in res["faces"]:
                writer.writerow([
                    res["path"], face_count, has_ai,
                    face["id"], f"{face['texture_var']:.2f}",
                    "是" if face["is_ai"] else "否"
                ])
    logger.info(f"批量检测完成，结果已保存至: {output_csv}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        logger.info("使用方式: python main.py <图片路径/目录>")
        sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        batch_detect(target)
    else:
        detector = FaceForensicsDetector(TEXTURE_THRESHOLD)
        result = detector.detect(target)
        for face in result["faces"]:
            print(f"人脸 {face['id']}: 纹理方差={face['texture_var']:.2f}, 判定={'AI' if face['is_ai'] else '真实'}")