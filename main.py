import os
import cv2
import numpy as np
from scipy.spatial import distance
from itertools import count as while_true
from object_detection import ObjectDetection
from feature_extraction import FeatureExtraction
from helpers import stack_images, new_coordinates_resize
import pandas as pd
import atexit

# creating an empty dataframe for storing event logs
activity_fields = ["Event_Type", "Person_ID","Time","Scene_ID"]
activity_logs = pd.DataFrame(columns=activity_fields)

# This piece of code is meant to run when the execution is stopped abruplty of naturally
def cleanup():
    print("Saving activity logs")
    activity_logs.to_csv('activity.csv',index=False)
atexit.register(cleanup)


# configuration variables
object_detection_threshold = 0.7
yolo_v4_tiny_path = "./pretrained_models/yolov4-tiny.onnx"
yolo_v4_classes = "./pretrained_models/coco.names"
osnet_path = "./pretrained_models/osnet_ain_x1_0_M.onnx"
video_path = "./sample_video/campus_4"
camera_image_size = [640, 480]
feature_extraction_threshold = 0.42
max_number_of_feauture_vectors_per_person = 512
resize_camera_image_by = 0.8

def main():
    global activity_logs # i set it global so that inner codes can store logs in this global dataframe
    # Variable for saved detected persons
    detected_persons = {}
    camera_view = {} # this is to store the set of persons_ids present in each camera scene

    # Initializing object detection module
    object_detection = ObjectDetection(confidence_threshold= object_detection_threshold,onnx_path= yolo_v4_tiny_path,coco_names_path=yolo_v4_classes,device= "cpu")

    # Initializing feature extraction module
    feature_extraction = FeatureExtraction(onnx_path= osnet_path,device= "cpu")

    # Setup camera
    cam = {}
    videos = np.array(os.listdir(video_path))
    total_cam = len(videos)
    for i in range(total_cam):
        cam[f"cam_{i}"] = cv2.VideoCapture(os.path.join(video_path, videos[i]))
        cam[f"cam_{i}"].set(3, camera_image_size[0] ) # setting the width of the captured videostream
        cam[f"cam_{i}"].set(4, camera_image_size[1] ) # setting the height of the captured videostream
        camera_view[f"scene_{i}"] = set()

    id = 0 # for maintaining person ids

    frame_number = 0 # this is used for keeping track of time
    for _ in while_true():  #this loop will repeat untill all the frames are processed
        # ith frame of each camera is processed in ith loop
        # Set up variable
        images = {}
        predicts = {}

        # Get a frame from each camera
        for i in range(total_cam):
            _, images[f"image_{i}"] = cam[f"cam_{i}"].read()

        # Predict persons with object detection for each frame
        for i in range(total_cam):
            predicts[f"image_{i}"] = object_detection.predict_img(images[f"image_{i}"])
            # this kind of information is given stored [{'Person': {'confidence': 0.99, 'bounding_box': [81, 117, 130, 365]}}]

        # Resize image for display in screen
        for i in range(total_cam):
            images[f"image_{i}"] = cv2.resize(
                images[f"image_{i}"],
                camera_image_size,                                                                                                                         # red
            )

        frame_scene = {} # a local data structure for storing the set of persons present in the frame at hand
        # camera_scene is the global data structure across multiple frames for storing the set of persons present in a scene
        # frame scene is used to update the camera scene
        for i in range(total_cam):
            frame_scene[f"scene_{i}"] = set()

        for i in range(total_cam):
            for predict in predicts[f"image_{i}"]:
                cls_name = tuple(predict.keys())[0]
                x1, y1, x2, y2 = predict[cls_name]["bounding_box"]

                # Resize bbox for new size image
                x1, y1 = new_coordinates_resize((object_detection.model_width, object_detection.model_height),camera_image_size,(x1, y1))
                x2, y2 = new_coordinates_resize((object_detection.model_width, object_detection.model_height),camera_image_size,(x2, y2))
                # Person identification
                cropped_image = images[f"image_{i}"][y1:y2, x1:x2]
                extracted_features = feature_extraction.predict_img(cropped_image)[0]

                # re-identification module....
                # Add new person if data is empty
                if not detected_persons: # that is if detected_persons in empty 
                    detected_persons[f"id_{id}"] = {
                        "extracted_features": extracted_features,
                        "id": id,
                        "camera_id": i,
                        "cls_name": cls_name,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": predict[cls_name]["confidence"],
                        "color": np.random.randint(0, 255, size=3),
                    }

                    # new person detected
                    print(f"New Person Detected | Assigned Person_ID: {id} | Video time: {format(frame_number/30,'.2f')} seconds")
                    # adding the event log
                    temp_row = pd.DataFrame([{'Event_Type':'New_Detection','Person_ID':id,'Time': format(frame_number/30,'.2f') , 'Scene_ID': i }])
                    activity_logs = pd.concat([activity_logs, temp_row], ignore_index=True)
                    # updating the global store
                    camera_view[f"scene_{i}"].add(id)
                    # person entered a scene
                    print(f"Person {id} entered scene {i} at time {format(frame_number/30,'.2f')}")
                    # adding the event log 
                    temp_row = pd.DataFrame([{'Event_Type':'Entry','Person_ID':id,'Time': format(frame_number/30,'.2f') , 'Scene_ID': i }])
                    activity_logs = pd.concat([activity_logs, temp_row], ignore_index=True)
                    # updating the local store
                    frame_scene[f"scene_{i}"].add(id)
                    id += 1
                else:
                    # Search best match from already existing persons
                    # calculating cosine distances with all the existing persons
                    top1_person = np.array(
                        [
                            {
                                "id": value["id"],
                                "cls_name": value["cls_name"],
                                "color": value["color"],
                                "score": distance.cosine(
                                    np.expand_dims(
                                        np.mean(value["extracted_features"], axis=0),
                                        axis=0,
                                    )
                                    if len(value["extracted_features"]) > 1
                                    else value["extracted_features"],
                                    extracted_features,
                                ),
                            }
                            for value in detected_persons.values()
                        ]
                    )
                    
                    # selecting the best match, lowest cosine distance
                    top1_person = sorted(
                        top1_person, key=lambda d: d["score"], reverse=False
                    )[0]
                    # Add data for new person or replace new bbox, confidence object detection, feature extraction embedding, and camera id for existing person
                    if top1_person["score"] < feature_extraction_threshold: # top match must have cosine distance less than this threshold for clear re-identification
                        detected_persons[f"id_{top1_person['id']}"] = {
                            "extracted_features": np.vstack(
                                (
                                    detected_persons[f"id_{top1_person['id']}"][
                                        "extracted_features"
                                    ],
                                    extracted_features,
                                )
                            )
                            if detected_persons[f"id_{top1_person['id']}"][
                                "extracted_features"
                            ].shape[0]
                            < max_number_of_feauture_vectors_per_person
                            else np.vstack(
                                (
                                    extracted_features,
                                    detected_persons[f"id_{top1_person['id']}"][
                                        "extracted_features"
                                    ][1:],
                                )
                            ),  # har ek person k liye zyada se zyada ek limit tk he feature vectors save krenge
                            "id": top1_person["id"],
                            "camera_id": i,
                            "cls_name": top1_person["cls_name"],
                            "bbox": (x1, y1, x2, y2),
                            "confidence": predict[cls_name]["confidence"],
                            "color": top1_person["color"],
                        }
                        frame_scene[f"scene_{i}"].add(top1_person['id'])
                        if top1_person["id"] not in camera_view[f'scene_{i}']:
                            camera_view[f'scene_{i}'].add(top1_person["id"])
                            print(f"Person {top1_person['id']} entered scene {i} at time {format(frame_number/30,'.2f')}")
                            temp_row = pd.DataFrame([{'Event_Type':'Entry','Person_ID':top1_person['id'],'Time': format(frame_number/30,'.2f') , 'Scene_ID': i }])
                            activity_logs = pd.concat([activity_logs, temp_row], ignore_index=True)
                    else:
                        # best match not good enough, hence new person detected
                        detected_persons[f"id_{id}"] = {
                            "extracted_features": extracted_features,
                            "id": id,
                            "camera_id": i,
                            "cls_name": cls_name,
                            "bbox": (x1, y1, x2, y2),
                            "confidence": predict[cls_name]["confidence"],
                            "color": np.random.randint(0, 255, size=3),
                        }
                        frame_scene[f"scene_{i}"].add(id)
                        print(f"New Person Detected | Assigned Person_ID: {id} | Video time: {format(frame_number/30,'.2f')} seconds")
                        temp_row = pd.DataFrame([{'Event_Type':'New_Detection','Person_ID':id,'Time': format(frame_number/30,'.2f') , 'Scene_ID': i }])
                        activity_logs = pd.concat([activity_logs, temp_row], ignore_index=True)
                        camera_view[f"scene_{i}"].add(id)
                        print(f"Person {id} entered scene {i} at time {format(frame_number/30,'.2f')}")
                        temp_row = pd.DataFrame([{'Event_Type':'Entry','Person_ID':id,'Time': format(frame_number/30,'.2f') , 'Scene_ID': i }])
                        activity_logs = pd.concat([activity_logs, temp_row], ignore_index=True)
                        id += 1

            # Draw all bbox
            for value in detected_persons.values():
                if value["camera_id"] == i:
                    cv2.rectangle(
                        images[f"image_{value['camera_id']}"],
                        value["bbox"][:2],
                        value["bbox"][2:],
                        value["color"].tolist(),
                        2,
                    )
                    cv2.putText(
                        images[f"image_{value['camera_id']}"],
                        f"{value['cls_name']} {value['id']}: {value['confidence']}",
                        (value["bbox"][0], value["bbox"][1] - 10),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        value["color"].tolist(),
                        2,
                    )

        for i in range(total_cam):
            temp_set = set()
            for pid in camera_view[f"scene_{i}"]:
                if pid not in frame_scene[f"scene_{i}"]:
                    print(f"Person {pid} has exited scene {i} at time {format(frame_number/30,'.2f')} seconds")
                    temp_row = pd.DataFrame([{'Event_Type':'Exit','Person_ID':pid,'Time': format(frame_number/30,'.2f') , 'Scene_ID': i }])
                    activity_logs = pd.concat([activity_logs, temp_row], ignore_index=True)

                else:
                    temp_set.add(pid)
            camera_view[f"scene_{i}"] = temp_set
    
        # Display all cam
        if total_cam % 2 == 0:
            display_image = stack_images(
                resize_camera_image_by,
                (
                    [images[f"image_{i}"] for i in range(0, total_cam // 2)],
                    [images[f"image_{i}"] for i in range(total_cam // 2, total_cam)],
                ),
            )
        else:
            display_image = stack_images(
                resize_camera_image_by,
                ([images[f"image_{i}"] for i in range(total_cam)],),
            )

        cv2.imshow("OUTPUT", display_image)
        if cv2.waitKey(1) == ord("q"):
            break
        frame_number += 1

    # Release all cam
    for i in range(total_cam):
        cam[f"cam_{i}"].release()
    cv2.destroyAllWindows()


main()