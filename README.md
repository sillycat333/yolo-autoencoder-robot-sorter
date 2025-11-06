### Convolutional Variational Autoencoder

This repo contains Python code to run an Autoencoder and YOLO model for my arm robot defect sorter project.

The basic workflow looks like this:

1. YOLO starts the camera feed.  
2. YOLO detects objects and crops the bounding box.  
3. The cropped image is passed to the autoencoder to check whether it's defect or no defect.  
4. The classification result is sent to Arduino through serial communication.  
5. The robot arm servos move based on whether the item is defective or not.

---

#### Main Files

- `main.py` — Runs YOLO + Autoencoder and sends classification results to Arduino via serial communication.  
- `main.ipynb` — Notebook for training the autoencoder model.  
- `test_saved.ipynb` — Runs the trained autoencoder using the saved model `ae.pth`.  
- `test_threshold.ipynb` — Helps choose the best threshold using Youden’s J statistic.  
- `plot.ipynb` — Visualizes the training results.  
- `ae.pth` — Saved autoencoder model.  
- `yolo.pth` — Saved YOLO model.

---

#### Sample Data

You can test the model with the provided sample defect/non-defect containers in: assets/*_input.png

Run `test_saved.ipynb` to try it out.

---

#### Notes

This repo only includes the code to train the autoencoder and to run the integrated YOLO–Autoencoder–Serial system. It does not include YOLO training code or the dataset. If you need the dataset, please contact me by email.

---

#### Demonstrations

![Normal](https://github.com/sillycat333/yolo-autoencoder-robot-sorter/blob/main/assets/1.png)

![Defect](https://github.com/sillycat333/yolo-autoencoder-robot-sorter/blob/main/assets/10.png)

- YOLO Detection Demo: uses open source EEZYbotARM Mk1 (3D printed)  

https://github.com/sillycat333/yolo-autoencoder-robot-sorter/blob/main/assets/yolo.avi

- Arm Robot Sorting Demo:  

https://github.com/sillycat333/yolo-autoencoder-robot-sorter/blob/main/assets/robot.mp4

- Screen Recording (Running Model):  

https://github.com/sillycat333/yolo-autoencoder-robot-sorter/blob/main/assets/recording.mkv

---

