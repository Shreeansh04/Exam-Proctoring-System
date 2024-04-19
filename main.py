import cv2
import mediapipe as mp
import pandas as pd
import datetime as dt
import numpy as np

# place holders and global variables
x = 0  # X axis head pose
y = 0  # Y axis head pose

X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0

video_file1 = 'Yousef1.avi'
def pose(video_file):
    global X_AXIS_CHEAT, Y_AXIS_CHEAT
    #############################
    print("1\n")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    print("2\n")
    cap = cv2.VideoCapture(video_file)
    mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions

    df = pd.DataFrame(columns=['Timestamp', 'X_AXIS_CHEAT', 'Y_AXIS_CHEAT'])

    while cap.isOpened():
        success, image = cap.read()
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        if not success:
            break
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get the result
        results = face_mesh.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        face_ids = [33, 263, 1, 61, 291, 199]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = None)
                for idx, lm in enumerate(face_landmarks.landmark):
                    # print(lm)
                    if idx in face_ids:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                        # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360

                # See where the user's head is tilting
                if y < -10:
                    X_AXIS_CHEAT = 1
                else:
                    X_AXIS_CHEAT = 0

                if x < -5:
                    Y_AXIS_CHEAT = 1
                else:
                    Y_AXIS_CHEAT = 0

                # Log the values of X_AXIS_CHEAT and Y_AXIS_CHEAT to the dataframe
                #df = df.append(
                #    {'Timestamp': cap.get(cv2.CAP_PROP_POS_MSEC), 'X_AXIS_CHEAT': X_AXIS_CHEAT, 'Y_AXIS_CHEAT': Y_AXIS_CHEAT},
                #    ignore_index=True)
                new_row = {'Timestamp': cap.get(cv2.CAP_PROP_POS_MSEC), 'X_AXIS_CHEAT': X_AXIS_CHEAT, 'Y_AXIS_CHEAT': Y_AXIS_CHEAT}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                cv2.line(image, p1, p2, (255, 0, 0), 2)

                # Add the text on the image
                cv2.putText(image, str(X_AXIS_CHEAT) + "::" + str(Y_AXIS_CHEAT), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

        cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    print("Woah")
    # Save the dataframe to a CSV file
    df.to_csv('head_pose_cheat_detection.csv', index=False)

pose(video_file1)