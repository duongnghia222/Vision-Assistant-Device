import cv2
import mediapipe as mp
import time
import numpy as np


class FingersCount:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.medhands = mp.solutions.hands
        # Initialize with max_num_hands set to 2 for detecting two hands
        self.hands = self.medhands.Hands(max_num_hands=2, min_detection_confidence=0.8)
        self.draw = mp.solutions.drawing_utils
        self.img = None

    def infer(self, img):
        self.img = img
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(imgrgb)

        left_hand_count = None
        right_hand_count = None

        if res.multi_hand_landmarks:
            for handlms in res.multi_hand_landmarks:
                h, w, c = self.img.shape
                finger_count, is_left_hand = self.update_finger_list(handlms, h, w)
                if is_left_hand:
                    left_hand_count = finger_count
                else:
                    right_hand_count = finger_count

                self.draw.draw_landmarks(self.img, handlms, self.medhands.HAND_CONNECTIONS,
                                         self.draw.DrawingSpec(color=(0, 255, 204), thickness=2, circle_radius=2),
                                         self.draw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=3))
        return [left_hand_count, right_hand_count]

    def update_finger_list(self, hand_landmarks, h, w):
        landmarks_list = []
        finger_list = []
        finger_tips_id = [4, 8, 12, 16, 20]  # 4 -> thumb tip, 8,12,16,20 -> index,middle,ring,pinky tips
        is_left_hand = True
        # Get all landmarks of a hand
        for i, lm in enumerate(hand_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks_list.append([i, cx, cy])

        # Once we have all 21 landmarks, process them
        if len(landmarks_list) == 21:
            # Improved thumb detection
            finger_list.append(self.is_thumb_up(landmarks_list))
            # print(landmarks_list[0][1])

            # Other fingers
            for i in range(1, 5):
                if landmarks_list[0][1] < self.screen_width // 2:  # this is left hand
                    finger_list.append(int(landmarks_list[finger_tips_id[i]][1] > landmarks_list[finger_tips_id[i] - 2][1]))
                else:
                    is_left_hand = False
                    finger_list.append(int(landmarks_list[finger_tips_id[i]][1] < landmarks_list[finger_tips_id[i] - 2][1]))

        return finger_list.count(1), is_left_hand

    def is_thumb_up(self, landmark_list):
        # Use the vector from the wrist to the base of the index finger as a reference
        vector34 = (landmark_list[3][1] - landmark_list[4][1], landmark_list[3][2] - landmark_list[4][2])
        vector32 = (landmark_list[3][1] - landmark_list[2][1], landmark_list[3][2] - landmark_list[2][2])

        # Calculate angle between vectors
        angle = abs(self.calculate_angle(vector34, vector32))

        # Determine if thumb is up (customize the threshold as needed)
        # is_thumb = (angle > 0 and landmark_list[4][1] < landmark_list[5][1]) or (angle < 0 and landmark_list[4][1] > landmark_list[5][1])
        # print(angle)
        return angle < 40 and landmark_list[4][2] < landmark_list[2][2]

    @staticmethod
    def calculate_angle(v1, v2):
        # Calculate the dot product of v1 and v2
        dot_product = np.dot(v1, v2)

        # Compute the norms (magnitudes) of the vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # Calculate the cosine of the angle (ensure it's within [-1, 1] to avoid numerical issues)
        cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1, 1)

        # Calculate the angle in radians and then convert to degrees
        angle_radians = np.arccos(np.abs(cos_angle))  # Use abs to ensure a positive angle
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees
