import os
import pickle
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = '.images'

data = []
labels = []
image_paths = []  # قائمة لتخزين مسارات الصور

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img_path_full = os.path.join(DATA_DIR, dir_, img_path)  # المسار الكامل للصورة
        img = cv2.imread(img_path_full)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                    # تحويل النقاط إلى الإحداثيات النسبية لحجم الصورة
                    x_pixel = int(x * img.shape[1])
                    y_pixel = int(y * img.shape[0])

                    # رسم النقاط على الصورة
                    cv2.circle(img_rgb, (x_pixel, y_pixel), 5, (0, 255, 0), -1)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
            image_paths.append(img_path_full)  # إضافة المسار للقائمة

            # عرض الصورة على الشاشة مع النقاط المعترف بها
            plt.imshow(img_rgb)
            plt.title(f'Class: {dir_}')
            plt.axis('off')
            plt.show()
            break  # نظرًا لأننا نريد عرض صورة واحدة فقط لكل مجموعة، سنتوقف هنا

# حفظ البيانات في ملف pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels, 'image_paths': image_paths}, f)