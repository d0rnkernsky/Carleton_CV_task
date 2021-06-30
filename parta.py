import numpy as np
import cv2 as cv
import cars_tracker as ct
import utils.utils as utils
from tensorflow import keras
import time
import os
import my_utils
import argparse

INPUT_H, INPUT_W = 416, 416
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default='./yolo.h5', help="path to keras pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.70, help="minimum probability to filter weak detections")
ap.add_argument("-o", "--out", default="./out.avi", help="path to output file")
ap.add_argument("-i", "--in", default="./fixed side of road.mp4", help="path to the input video file")
ap.add_argument("-v", "--verbose", type=bool, help="display frame processing ", default=False)
args = vars(ap.parse_args())


def main():
    """
    :return:
    """
    # load and compile a model to detect target objects
    model = keras.models.load_model(args.get('model'))
    model.compile()

    # delete results from the previous run
    if os.path.exists(args.get('out')):
        os.remove(args.get('out'))

    border_coord = None

    # open the video file
    cap = cv.VideoCapture(args.get('in'))
    if not cap.isOpened():
        raise Exception(f"Couldn't open the video file: {args.get('in')}")

    # get the video's properties
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    out = cv.VideoWriter(args.get('out'), cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    tracker = ct.CarsTracker(fps, max_frames_disappear=30)
    start = time.time()

    # the main loop to process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img = np.copy(frame)
            image_h, image_w = frame.shape[:2]

            # set a border to count passing cars
            if border_coord is None:
                b = int(image_h * 0.5)
                border_coord = 0, b, image_w, b
                tracker.set_border(b)

            frame = my_utils.prepare_video_frame(frame, INPUT_H, INPUT_W)
            pred = model.predict(frame)

            boxes = []
            for i in range(len(pred)):
                # decode the output of the network
                box_cont = utils.decode_netout(pred[i][0], my_utils.anchors[i], args.get('confidence'), INPUT_H,
                                               INPUT_W)

                # filter out all other objects except the target object
                for box in box_cont:
                    if box.get_label() == my_utils.TARGET_CLASS_IDX:
                        boxes.append(box)

            # prepare bounding boxes to be displayed in the original image
            my_utils.correct_yolo_boxes(boxes, image_h, image_w, INPUT_H, INPUT_W)
            utils.do_nms(boxes, 0.5)
            v_boxes, v_labels, v_scores = my_utils.get_boxes(boxes, args.get('confidence'))
            for i in range(len(v_boxes)):
                my_utils.draw_box(img, v_boxes[i], v_labels[i], v_scores[i])

            # update tracked object info
            objects = tracker.update(v_boxes)
            for obj_id, centroid in objects.items():
                my_utils.draw_centroid(img, centroid)

            # add additional objects to track passing cars
            my_utils.draw_line(img, border_coord)
            my_utils.draw_counter(img, tracker.count_passed_cars())

            if args.get('verbose'):
                cv.imshow('Frame', img)
            out.write(img)

            if cv.waitKey(25) == ord('q'):
                break

        else:
            break

    # release the video capture object and close the frames
    cap.release()
    out.release()
    cv.destroyAllWindows()

    print(f'Done in {int(time.time() - start)}s')


if __name__ == '__main__':
    main()
