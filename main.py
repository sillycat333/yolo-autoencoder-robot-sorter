import collections
import os
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import serial
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO


class CVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc = self.encoder(x)
        enc = enc.view(x.size(0), -1)
        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)
        z = self.reparameterize(mu, logvar)
        dec_input = self.decoder_input(z)
        dec_input = dec_input.view(x.size(0), 512, 4, 4)
        reconstruction = self.decoder(dec_input)
        return reconstruction, mu, logvar


def get_centroid(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def check_anomaly(crop_img):
    image = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstructed, _, _ = ae_model(image_tensor)
    mse_loss = torch.mean((reconstructed - image_tensor) ** 2).item()
    return mse_loss


def save_anomaly_heatmap(image_tensor, reconstructed, loss_value, obj_id):
    anomaly_map = torch.abs(image_tensor - reconstructed).mean(dim=1, keepdim=True)
    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    anomaly_map = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-8
    )

    orig_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(orig_np)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(recon_np)
    axs[1].set_title("Reconstructed Image")
    axs[1].axis("off")

    axs[2].imshow(orig_np)
    axs[2].imshow(anomaly_map, cmap="jet", alpha=0.5)
    axs[2].set_title(f"Anomaly Heatmap\nLoss: {loss_value:.6f}")
    axs[2].axis("off")

    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/heatmap_{obj_id}_{timestamp}.png"
    plt.savefig(filename)
    plt.close(fig)

    print(f"[Saved] Heatmap image saved to {filename}")


TRANSFORM = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AE_MODEL_PATH = "ae.pth"
YOLO_MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.90
LOSS_BATCH_SIZE = 20
DELAY_SECONDS = 0.005
OBJECT_WARMUP_FRAMES = 5
CENTROID_TOLERANCE = 30
SPIKE_FILTER_THRESHOLD = 0.0099
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
DEFECT_THRESHOLD = 0.007183

print("Loading models...")
device = DEVICE
transform = TRANSFORM
ae_model = CVAE(latent_dim=128).to(device)
ae_model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=device))
ae_model.eval()
yolo_model = YOLO(YOLO_MODEL_PATH)
print(f"Models loaded successfully on {device}.")

print("Initializing camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()
print("Camera initialized.")

arduino = None
try:
    print(f"Connecting to Arduino on {SERIAL_PORT}...")
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("Arduino connected.")
except serial.SerialException as e:
    print(f"WARNING: Could not connect to Arduino. {e}")
    print("Running in SIMULATION MODE. No data will be sent.")

object_history = collections.defaultdict(dict)

print("\nStarting monitoring")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = yolo_model(frame, verbose=False)[0]
        current_time = time.time()

        for box in results.boxes:
            if box.conf[0] < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            centroid = get_centroid(x1, y1, x2, y2)

            for obj_id in list(object_history.keys()):
                if (
                    current_time - object_history[obj_id].get("last_seen", current_time)
                    > 2
                ):
                    print(f"[Cleanup] Removing stale object ID {obj_id}")
                    del object_history[obj_id]

            matched_id = None
            for obj_id, data in object_history.items():
                prev_centroid = data.get("last_centroid")
                if (
                    prev_centroid
                    and abs(prev_centroid[0] - centroid[0]) < CENTROID_TOLERANCE
                    and abs(prev_centroid[1] - centroid[1]) < CENTROID_TOLERANCE
                ):
                    data["seen_frames"] += 1
                    data["last_centroid"] = centroid
                    data["last_seen"] = current_time
                    matched_id = obj_id
                    break

            if matched_id is None:
                matched_id = len(object_history)
                object_history[matched_id] = {
                    "seen_frames": 1,
                    "last_centroid": centroid,
                    "valid_losses": [],
                    "first_sent": False,
                    "last_seen": current_time,
                }
                print(f"[New Object] Assigned ID {matched_id}")

            current_obj = object_history[matched_id]

            if current_obj["seen_frames"] <= OBJECT_WARMUP_FRAMES:
                print(
                    f"[Warmup] Skipping object {matched_id}, frame {current_obj['seen_frames']}"
                )
                continue

            anomaly_score = check_anomaly(crop)

            if anomaly_score > SPIKE_FILTER_THRESHOLD:
                print(f"[Spike Filter] Score {anomaly_score:.6f} too high, ignored.")
                current_obj["valid_losses"].clear()
                continue

            current_obj["valid_losses"].append(anomaly_score)

            if len(current_obj["valid_losses"]) >= LOSS_BATCH_SIZE:
                avg_error = sum(current_obj["valid_losses"]) / len(
                    current_obj["valid_losses"]
                )

                if not current_obj["first_sent"]:
                    print(
                        f"[Skip First Send] Skipping first Arduino send for object {matched_id}"
                    )
                    current_obj["first_sent"] = True
                    current_obj["valid_losses"].clear()
                    continue

                image_tensor = (
                    transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.no_grad():
                    reconstructed, _, _ = ae_model(image_tensor)

                save_anomaly_heatmap(image_tensor, reconstructed, avg_error, matched_id)

                print(f"[*] Object {matched_id} final avg loss: {avg_error:.6f}")
                label = "DEFECT" if avg_error > DEFECT_THRESHOLD else "OK"
                msg = f"{label},{avg_error:.6f}\n"

                if arduino:
                    try:
                        print(f"Sending to Arduino: {msg.strip()}")
                        arduino.write(msg.encode())
                    except serial.SerialException as e:
                        print(f"ERROR: Arduino write failed: {e}. Disconnecting.")
                        arduino.close()
                        arduino = None
                else:
                    print(f"[SIMULATED] Arduino message: {msg.strip()}")

                current_obj["valid_losses"].clear()
            else:
                print(
                    f"[Collecting] Object {matched_id} sample {len(current_obj['valid_losses'])}/{LOSS_BATCH_SIZE}"
                )

        time.sleep(DELAY_SECONDS)

except KeyboardInterrupt:
    print("\nInterruption detected")
finally:
    cap.release()
    print("Camera released.")
    if arduino and arduino.is_open:
        arduino.close()
        print("Arduino connection closed.")
    print("Exited.")
