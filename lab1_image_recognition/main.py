# установить перед началом работы pip3 install
# https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl

from imageai.Prediction import ImagePrediction
from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection
from imageai.Prediction.Custom import CustomImagePrediction
import matplotlib.pyplot as plt

# lab1 consts
LAB1_INPUT_IMAGE_PATH = "./nevsnkiy1.jpg"
LAB1_MODEL_PATH = './resnet50_weights_tf_dim_ordering_tf_kernels.h5'

# lab2 consts
LAB2_MODEL_PATH = './resnet50_coco_best_v2.0.1.h5'
LAB2_INPUT_IMAGE_PATH = './nevsnkiy1.jpg'
LAB2_OUTPUT_IMAGE_PATH = './nevsnkiy1_output.jpg'

# lab3 consts
LAB3_INPUT_VIDEO_PATH = './nevkiy_video_1sec.mp4'
LAB3_OUTPUT_VIDEO_PATH = './nevkiy_video_1sec_detected.mp4'
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
    frame_square_area = []
    prob = []
    annotations = []

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(LAB2_MODEL_PATH)
    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image=LAB2_INPUT_IMAGE_PATH,
                                                 output_image_path=LAB2_OUTPUT_IMAGE_PATH)

    for eachObject in detections:
        img_height = eachObject['box_points'][3] - eachObject['box_points'][1]
        img_width = eachObject['box_points'][2] - eachObject['box_points'][0]
        square_area = img_height * img_width

        prob.append(eachObject['percentage_probability'])
        frame_square_area.append(square_area)
        annotations.append(eachObject["name"])

        print(eachObject["name"], " : ", eachObject["percentage_probability"], ", square area : ", square_area)

    plt.ylabel('probability (%)')
    plt.xlabel('image square area')
    plt.title('Dependence of object size and recognition probability')
    plt.scatter(frame_square_area, prob)
    for i, txt in enumerate(annotations):
        plt.annotate(txt, (frame_square_area[i], prob[i]))
    plt.show()


def per_sec_function(sec, this_second_output_object_array, this_second_counting_array,
                     this_second_counting, detected_copy):
    for i, frame in enumerate(this_second_output_object_array):
        frame_square_area = []
        prob = []
        annotations = []

        for eachObject in frame:
            img_height = eachObject['box_points'][3] - eachObject['box_points'][1]
            img_width = eachObject['box_points'][2] - eachObject['box_points'][0]
            square_area = img_height * img_width

            prob.append(eachObject['percentage_probability'])
            frame_square_area.append(square_area)
            annotations.append(eachObject["name"])

        plt.figure()
        plt.ylabel('probability (%)')
        plt.xlabel('image square area')
        plt.title("Dependence of object size and recognition probability. Second #{} Frame #{}".format(sec, i+1))
        plt.scatter(frame_square_area, prob)

        for j, txt in enumerate(annotations):
            plt.annotate(txt, (frame_square_area[j], prob[j]))

        plt.savefig("frame_{}_sec_{}.png".format(i+1, sec))
        plt.clf()
        plt.cla()

    print(sec, this_second_output_object_array)


def lab3_video_detection_and_analysis():
    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(LAB3_MODEL_PATH)
    detector.loadModel()
    video_path, counting, output_objects_array = detector.detectObjectsFromVideo(
        input_file_path=LAB3_INPUT_VIDEO_PATH,
        output_file_path=LAB3_OUTPUT_VIDEO_PATH,
        frames_per_second=29,
        log_progress=True,
        return_detected_frame=True,
        per_second_function=per_sec_function)

    print(video_path)


def lab4_custom_image_prediction():
    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath("idenprof_061-0.7933.h5")
    prediction.setJsonPath("idenprof_model_class.json")
    prediction.loadModel(num_objects=10)

    predictions, probabilities = prediction.predictImage("firefighter_training_1200.jpg", result_count=3)

    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction, " : ", eachProbability)


if __name__ == '__main__':
    lab3_video_detection_and_analysis()
