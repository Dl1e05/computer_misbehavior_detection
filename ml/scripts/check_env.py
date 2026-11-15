import torch, onnxruntime as ort, cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis

print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("onnxruntime providers:", ort.get_all_providers())
r = YOLO("yolo26n.pt")("https://ultralytics.com/images/bus.jpg", device=0, verbose=False)[0]
print("YOLO нашла объктов:", len(r.boxes))
app = FaceAnalysis(name="buffalo_l", provides=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0)
print("InsightFace OK")