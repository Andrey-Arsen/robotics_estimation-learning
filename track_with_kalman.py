import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO

model = YOLO(r"C:\Users\ADMIN\OneDrive\Desktop\robot estimation\yolo\best_my_car.pt")

'''def create_kalman():
    kf = KalmanFilter(dim_x=6, dim_z=2)
    
    dt = 1.0

    # Модель движения: учитываем ускорение
    kf.F = np.array([
        [1, 0, dt, 0, 0.5*dt**2, 0],
        [0, 1, 0, dt, 0, 0.5*dt**2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],  # измерения: только x, y
        [0, 1, 0, 0, 0, 0]
    ])

    kf.P *= 1000.
    kf.R = np.eye(2) * 10  # шум измерений
    kf.Q = np.eye(6) * 0.05  # шум процесса (сделали менее доверчивым к модели)
    return kf'''
    
def create_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    
    dt = 1.0  # шаг времени

    # Модель движения: только позиция и скорость
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Матрица наблюдения: измеряем только положение (x, y)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    # Начальная ковариация ошибки
    kf.P *= 1000.

    # Шум измерения (например, шум в координатах камеры)
    kf.R = np.eye(2) * 10

    # Шум процесса (модель не идеальна)
    q = 0.05
    kf.Q = q * np.array([
        [dt**4/4, 0, dt**3/2, 0],
        [0, dt**4/4, 0, dt**3/2],
        [dt**3/2, 0, dt**2, 0],
        [0, dt**3/2, 0, dt**2]
    ])

    return kf

kalman = create_kalman()
found = False
missed_frames = 0
MAX_MISSED = 30
trajectory = []

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()

    kalman.predict()

    if len(detections) > 0:
        x1, y1, x2, y2 = detections[0]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if not found:
            kalman.x[:2] = np.array([[cx], [cy]])
            kalman.x[2:] = 0
            found = True
        else:
            kalman.update([cx, cy])
        
        missed_frames = 0
        color = (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    else:
        missed_frames += 1

    if found:
        pred_x, pred_y = kalman.x[0], kalman.x[1]
        trajectory.append((int(pred_x), int(pred_y)))
        if len(trajectory) > 50:
            trajectory.pop(0)


        color = (0, 255, 255) if missed_frames > 0 else (255, 0, 0)
        cv2.circle(frame, (int(pred_x), int(pred_y)), 8, color, -1)

 
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i-1], trajectory[i], (255, 255, 255), 2)

        if missed_frames > MAX_MISSED:
            found = False
            trajectory.clear()

    cv2.imshow("Improved Kalman Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
