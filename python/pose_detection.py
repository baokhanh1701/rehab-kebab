import json
import socket
from typing import Optional, Sequence, Literal
import cv2
import pprint
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles


def extract_pose_index(
    pose_type: Sequence,
    pose_landmarks: Sequence
) -> int:
    """
    Extract the index of the pose from the multi-pose list
    """
    pose_type_lms = pose_landmarks.landmark[mp_pose.PoseLandmark[pose_type.upper()]]
    if (pose_type_lms.visibility <= 0.5):
        return -1
    
    return mp_pose.PoseLandmark[pose_type.upper()]

def extract_pose_coords(
    pose_landmarks: Optional[Sequence],
    pose_world_landmarks,
    # segmentation_mask
) -> dict:
    """
    1) "pose_landmarks" field that contains the pose landmarks.
    2) "pose_world_landmarks" field that contains the pose landmarks in
        real-world 3D coordinates that are in meters with the origin at the
        center between hips.
    3) "segmentation_mask" field that contains the segmentation mask if
        "enable_segmentation" is set to true.
    """
    pose_coords = {
        "nose": None,
        "left_eye_inner": None,
        "left_eye": None,
        "left_eye_outer": None,
        "right_eye_inner": None,
        "right_eye": None,
        "right_eye_outer": None,
        "left_ear": None,
        "right_ear": None,
        "mouth_left": None,
        "mouth_right": None,
        "left_shoulder": None,
        "right_shoulder": None,
        "left_elbow": None,
        "right_elbow": None,
        "left_wrist": None,
        "right_wrist": None,
        "left_pinky": None,
        "right_pinky": None,
        "left_index": None,
        "right_index": None,
        "left_thumb": None,
        "right_thumb": None,
        "left_hip": None,
        "right_hip": None,
        "left_knee": None,
        "right_knee": None,
        "left_ankle": None,
        "right_ankle": None,
        "left_heel": None,
        "right_heel": None,
        "left_foot_index": None,
        "right_foot_index": None,
    }
    # print("pose_landmarks: ", pose_landmarks)
    # print("pose_world_landmarks: ", pose_world_landmarks)
    # print("segmentation_mask: ", segmentation_mask)

    if pose_landmarks is None:
        return pose_coords
    
    for pose_type in pose_coords.keys():
        pose_index = extract_pose_index(pose_type, pose_landmarks)

        # Visibility is not clear
        if pose_index == -1:
            continue

        pose_lms = pose_landmarks.landmark[pose_index]
        
        pose_coords[pose_type] = {
            "x": pose_lms.x,
            "y": pose_lms.y,
            "z": pose_lms.z,
            "visibility": pose_lms.visibility
        }
    
    pprint.pprint(pose_coords)
    return pose_coords
    
def run_pose_tracking_server(
    server_ip: str,
    server_port: int,
) -> None:
    """
    Run the pose tracking which sends the hand coordinates via UDP.

    Args:
        server_ip: The IP address of the server
        server_port: The port number of the server
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Open the webcam video feed
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            # outputs=['pose_landmarks', 'pose_world_landmarks', 'segmentation_mask'])

            pose_coords = extract_pose_coords(
                results.pose_landmarks,
                results.pose_world_landmarks,
                # results.segmentation_mask
            )
            
            # Send the hand coordinates to the client
            encoded_coords = json.dumps(pose_coords)
            client_socket.sendto(encoded_coords.encode(),
                                    (server_ip, server_port))

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if results.pose_landmarks:
                # print("poses_landmarks", results.pose_landmarks)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
            cv2.imshow("Pose Tracking", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    run_pose_tracking_server(
        server_ip="127.0.0.1",
        server_port=4242,
    )
