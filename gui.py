import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
from detector import FaceForensicsDetector
from config import TEXTURE_THRESHOLD
import threading

class FaceDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI人脸真伪检测工具 - 专业版")
        self.geometry("1000x700")
        self.detector = FaceForensicsDetector(TEXTURE_THRESHOLD)
        self.current_image = None
        self.result_image = None
        self._create_widgets()

    def _create_widgets(self):
        top_frame = ttk.Frame(self, padding=10)
        top_frame.pack(fill=tk.X)
        self.btn_select = ttk.Button(top_frame, text="选择图片", command=self.load_image)
        self.btn_select.pack(side=tk.LEFT, padx=5)
        self.btn_detect = ttk.Button(top_frame, text="开始检测", command=self.run_detection)
        self.btn_detect.pack(side=tk.LEFT, padx=5)

        center_frame = ttk.Frame(self, padding=10)
        center_frame.pack(fill=tk.BOTH, expand=True)
        self.left_label = ttk.Label(center_frame, text="原图")
        self.left_label.pack(side=tk.LEFT, padx=5)
        self.canvas_original = tk.Canvas(center_frame, width=450, height=500)
        self.canvas_original.pack(side=tk.LEFT, padx=5)
        self.right_label = ttk.Label(center_frame, text="检测结果")
        self.right_label.pack(side=tk.RIGHT, padx=5)
        self.canvas_result = tk.Canvas(center_frame, width=450, height=500)
        self.canvas_result.pack(side=tk.RIGHT, padx=5)

        self.log_text = tk.Text(self, height=10, state=tk.DISABLED)
        self.log_text.pack(fill=tk.X, padx=10, pady=5)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.current_image = cv2.imread(path)
            self._show_image(self.canvas_original, self.current_image)
            self.log("已加载图片: " + path)

    def run_detection(self):
        if self.current_image is None:
            messagebox.showwarning("警告", "请先选择图片！")
            return
        def task():
            self.log("开始检测...")
            self.btn_detect.config(state=tk.DISABLED)
            temp_path = "temp_detect.jpg"
            cv2.imwrite(temp_path, self.current_image)
            result = self.detector.detect(temp_path)
            self.result_image = self.current_image.copy()
            for face in result["faces"]:
                x1, y1, x2, y2 = face["bbox"]
                color = (0, 0, 255) if face["is_ai"] else (0, 255, 0)
                label = "AI" if face["is_ai"] else "真实"
                text = f"{label} (Var:{face['texture_var']:.0f})"
                cv2.rectangle(self.result_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(self.result_image, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            self._show_image(self.canvas_result, self.result_image)
            self.log(f"检测完成！发现 {len(result['faces'])} 个人脸，含AI脸: {result['is_ai_generated']}")
            self.btn_detect.config(state=tk.NORMAL)
        threading.Thread(target=task, daemon=True).start()

    def _show_image(self, canvas, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        scale = min(450/w, 500/h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        photo = ImageTk.Photoimage(image=Image.fromarray(img_resized))
        canvas.create_image(225, 250, image=photo)
        canvas.image = photo

    def log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = FaceDetectionApp()
    app.mainloop()