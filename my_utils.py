import cv2 as cv
import tensorflow as tf

CLASS_LIST = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
TARGET_CLASS = 'car'
TARGET_CLASS_IDX = CLASS_LIST.index(TARGET_CLASS)
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    """
        Set boxes' coordinates in the original image dimensions
    :param boxes: bounding boxes
    :param image_h: original image height
    :param image_w: original image width
    :param net_h: network input height
    :param net_w: network input width
    """
    new_w, new_h = net_w, net_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def get_boxes(in_boxes, threshold):
    """
        Return the bounding boxes with class score greater than threshold
    :param in_boxes: bounding boxes
    :param threshold: threshold to keep bounding boxes
    :return: boxes with class scores greater than threshold, their labels and scores
    """
    boxes, labels, scores = [], [], []
    for box in in_boxes:
        for i in range(len(CLASS_LIST)):
            if box.classes[i] > threshold:
                boxes.append(box)
                labels.append(i)
                scores.append(box.classes[i])
    return boxes, labels, scores


def draw_box(img, box, class_idx, prob):
    """
        Draws a box in the given image and adds class name anc score
    :param img: image
    :param box: bounding box
    :param class_idx: detected class index
    :param prob: detected class score
    :return:
    """

    # add the rectangle
    y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
    cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # add the class name and score
    offset = 10
    y = y1 - offset if y1 - offset > offset else y1 + offset
    cv.putText(img, f"{CLASS_LIST[class_idx]}: {prob * 100:.2f}%", (x1, y), cv.FONT_HERSHEY_SIMPLEX, \
               0.5, (255, 255, 255), 1)


def draw_centroid(in_img, centroid):
    """
        Adds a circle to represent the centroid in the given image
    :param in_img:
    :param centroid:
    """
    x, y = centroid.x, centroid.y
    cv.circle(in_img, (x, y), 4, (0, 0, 255), -1)
    cv.putText(in_img, f"Speed {centroid.get_avg_speed()} pxs/s, direction: {centroid.direction}", (x - 10, y - 10), \
               cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)


def draw_line(in_img, pos):
    """
        Adds a line in the given image
    :param in_img:
    :param pos: line coordinates as (pt1, pt2)
    """
    x1, y1, x2, y2 = pos
    cv.line(in_img, (x1, y1), (x2, y2), (255, 0, 0), 1)


def draw_counter(in_img, cars_num):
    """
        Adds a textbox in the image
    :param in_img:
    :param cars_num: passed cars number
    """
    cv.putText(in_img, f"Cars passed: {cars_num}", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def prepare_video_frame(in_frame, net_h, net_w):
    """
        Prepares a frame to pass into a dnn
    :param in_frame: video frame
    :param net_h: network's input height
    :param net_w: network's input width
    :return: a tensor with the resized frame
    """

    # resize the original image to the network input size
    frame = cv.resize(in_frame, (net_h, net_w))

    frame = frame.astype('float32') / 255.0
    frame = tf.expand_dims(frame, 0)

    return frame
