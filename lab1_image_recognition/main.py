# установить перед началом работы pip3 install
# https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl

from imageai.Prediction import ImagePrediction
from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection
from imageai.Prediction.Custom import CustomImagePrediction

# lab1 consts
LAB1_INPUT_IMAGE_PATH = "./test_image.jpg"
LAB1_MODEL_PATH = './resnet50_weights_tf_dim_ordering_tf_kernels.h5'

# lab2 consts
LAB2_MODEL_PATH = './resnet50_coco_best_v2.0.1.h5'
LAB2_INPUT_IMAGE_PATH = './test_image.jpg'
LAB2_OUTPUT_IMAGE_PATH = './lab2_output.jpg'

# lab3 consts
LAB3_INPUT_VIDEO_PATH = './traffic-mini.mp4'
LAB3_OUTPUT_VIDEO_PATH = './traffic_mini_detected_1'
LAB3_MODEL_PATH = './yolo.h5'


def lab1_image_recognition():
    prediction = ImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(LAB1_MODEL_PATH)
    prediction.loadModel()

    predictions, percentage_probabilities = prediction.predictImage(LAB1_INPUT_IMAGE_PATH, result_count=5)
    for index in range(len(predictions)):
        print(predictions[index], " : ", percentage_probabilities[index])


def lab2_object_detection():
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(LAB2_MODEL_PATH)
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=LAB2_INPUT_IMAGE_PATH,
                                                 output_image_path=LAB2_OUTPUT_IMAGE_PATH)

    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"])


def lab3_video_detection_and_analysis():
    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(LAB3_MODEL_PATH)
    detector.loadModel()

    video_path = detector.detectObjectsFromVideo(input_file_path=LAB3_INPUT_VIDEO_PATH,
                                                 output_file_path=LAB3_OUTPUT_VIDEO_PATH,
                                                 frames_per_second=29,
                                                 log_progress=True)
    print(video_path)


def lab4_custom_image_prediction():
    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath("idenprof_061-0.7933.h5")
    prediction.setJsonPath("idenprof_model_class.json")
    prediction.loadModel(num_objects=10)

    predictions, probabilities = prediction.predictImage("image.jpg", result_count=3)

    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction, " : ", eachProbability)


if __name__ == '__main__':
    lab2_object_detection()