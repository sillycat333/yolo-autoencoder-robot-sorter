#### Convolutional Variational Autoencoder

This repo contains Python code to run an **Autoencoder** and **YOLO** model for my **arm robot defect sorter** project.

The basic workflow looks like this:

1. **YOLO** starts the camera feed.  
2. **YOLO** detects objects and crops the **bounding box**.  
3. The **cropped image** is passed to the **autoencoder** to check whether it's *defect* or *no defect*.  
4. The classification result is sent to **Arduino** through **serial communication**.  
5. The **robot arm servos** move based on whether the item is defective or not.

---

##### Main Files

- **`main.py`** — Runs YOLO + Autoencoder and sends classification results to Arduino via serial communication.  
- **`main.ipynb`** — Notebook for **training the autoencoder model**.  
- **`test_saved.ipynb`** — Runs the trained autoencoder using the saved model `ae.pth`.  
- **`test_threshold.ipynb`** — Helps **choose the best threshold** using **Youden’s J statistic**.  
- **`plot.ipynb`** — Visualizes the **training results**.  
- **`ae.pth`** — Saved **autoencoder model**.  
- **`yolo.pth`** — Saved **YOLO model**.

---

##### Sample Data

You can test the model with the provided **sample defect/non-defect containers** in: assets/*_input.png

Run `test_saved.ipynb` to try it out.

---

##### Notes

- The repo includes:
  - Code to **train YOLO**
  - Code to **train and run the autoencoder**
- The **dataset is not included**.  
  If you need the data, please email me.

---

##### Demonstrations

- **YOLO Detection Demo:**  
  *(video)*  

- **Arm Robot Sorting Demo:**  
  *(video)*  

- **Screen Recording (Running Model):**  
  *(video)*  

---
