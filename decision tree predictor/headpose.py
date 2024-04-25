# head_pose.py
import cv2
import mediapipe as mp
import numpy as np

def estimate_head_pose(image):
    # Initialize mediapipe face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Process the image for face mesh
    results = face_mesh.process(image)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    face_ids = [33, 263, 1, 61, 291, 199]

    X_AXIS_CHEAT = 0
    Y_AXIS_CHEAT = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in face_ids:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360

            # Detect if the user's head is tilting
            if y < -10:
                X_AXIS_CHEAT = 1
            else:
                X_AXIS_CHEAT = 0

            if x < -5:
                Y_AXIS_CHEAT = 1
            else:
                Y_AXIS_CHEAT = 0

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
            cv2.line(image, p1, p2, (255, 0, 0), 2)

            # Add the text on the image
            cv2.putText(image, str(X_AXIS_CHEAT) + "::" + str(Y_AXIS_CHEAT), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

    return image, X_AXIS_CHEAT, Y_AXIS_CHEAT