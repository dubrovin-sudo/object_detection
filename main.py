import os
import cv2
from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection

execution_path = os.getcwd()  # path to resnet


class Picture:

    def __init__(self, exec_path, path_to_photo, path_to_model):
        self.path_to_photo = path_to_photo
        self.path_to_model = path_to_model
        self.exec_path = exec_path

    def detect(self):
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()  # use retinanet model for objects detection
        detector.setModelPath(os.path.join(self.exec_path, self.path_to_model))
        detector.loadModel()

        data = detector.detectObjectsFromImage(
            input_image=os.path.join(self.exec_path, self. path_to_photo),
            output_image_path=os.path.join(self.exec_path, f'detected_{self.path_to_photo}'),
            minimum_percentage_probability=70,
            display_percentage_probability=True,
            display_object_name=False
        )

        # print(data)


class Video(Picture):

    def __init__(self, exec_path, path_to_photo, path_to_model):
        super().__init__(exec_path, path_to_photo, path_to_model)

    def detect(self):
        camera = cv2.VideoCapture(0)

        detector = VideoObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(self.exec_path , self.path_to_model))
        detector.loadModel()

        video_path = detector.detectObjectsFromVideo(
            input_file_path=os.path.join(self.exec_path, self.path_to_photo),
            output_file_path=os.path.join(self.exec_path, f'{self.path_to_photo}_detected'),
            frames_per_second=20, log_progress=True, minimum_percentage_probability=30)

        print(video_path)


if __name__ == "__main__":
    picture = Picture(execution_path, 'park.jpg', 'resnet50_coco_best_v2.1.0.h5')
    picture.detect()
    # video = Video(execution_path, 'traffic .mp4', 'yolo.h5')
    # video.detect()