"""
Sections of this code were taken from:
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
"""
import numpy as np

import os
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
import cv2
import argparse
import datetime
from argparse import ArgumentParser
import csv
import datetime
import os.path
import numpy as np
from timeit import default_timer as timer






# Path to frozen detection graph. This is the actual model that is used
# for the object detection.
# MODEL_NAME = 'persona_inference_graph'
MODEL_NAME = 'faster_rcnn_inception_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')
NUM_CLASSES = 1  # mcsoco 90
sys.path.append("..")


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def create_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return detection_graph, category_index



def detect_in_cam():
    detection_graph, category_index = create_graph()

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object
            # was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class
            # label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            cap = cv2.VideoCapture(0)
            
            
            accum_time = 0
            curr_fps = 0
            fps = "FPS: ??"
            prev_time = timer()
            i=0
            while(cap.isOpened()):
                # Read the frame
                ret, color_frame = cap.read()
                image_np_expanded = np.expand_dims(color_frame, axis=0)
                
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores,
                        detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})


                t = datetime.datetime.now()
                personas = get_number_persons(scores)                

                # Uncoment to write csv every 100 frames
                """
                if i == 100:
                    if(personas>=1):
                      print(personas," Persons detected at ",t)
                      write_csv(str(t).split()[0], personas, t)
                      i=0
                i+=1
                """

                vis_util.visualize_boxes_and_labels_on_image_array(
                    color_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4,
                    min_score_thresh=.85)
                
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0
                cv2.putText(color_frame, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=(255, 0, 0), thickness=2)

                cv2.imshow('object detection', cv2.resize(color_frame, (800,600)))
                #output_rgb = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)


                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    break

            

def get_number_persons(score):
  #np.where(score >= 0.5)
  data = np.where(score >= 0.5)[0]
  return len(data)

def write_csv(video_name, number,time):
    if os.path.isfile(video_name+".csv"):
        with open(video_name+".csv", 'a', newline='') as csvfile:
            fieldnames = ['number_persons', 'time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'number_persons': number, 'time': time})
    else:          
        with open(video_name+".csv", 'a') as csvfile:
            fieldnames = ['number_persons', 'time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'number_persons': number, 'time': time})


def detect_in_video(input_path='sample.mp4', output_path='output/'):
    """if args.output is not None:
        output = args.output
    else:
        output=''"""

    video_name = input_path.split('/')[-1]
    # VideoWriter is the responsible of creating a copy of the video
    # used for the detections but with the detections overlays. Keep in
    # mind the frame size has to be the same as original video.
    out = cv2.VideoWriter(output_path+video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (800, 600))

    detection_graph, category_index = create_graph()

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object
            # was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class
            # label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            cap = cv2.VideoCapture(input_path)

            while(cap.isOpened()):
                # Read the frame
                ret, color_frame = cap.read()
                image_np_expanded = np.expand_dims(color_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores,
                        detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})


                millis = cap.get(cv2.CAP_PROP_POS_MSEC)
                seconds, minutes, hours=(millis/1000)%60, (millis/(1000*60))%60, (millis/(1000*60*60))%24
                t = datetime.time(int(hours), int(minutes), int(seconds),int(millis))
                personas = get_number_persons(scores)

                # Uncoment and refine to write csv
                """
                if(personas>=1):
                    print(personas," Persons detected at ",t)
                    write_csv(video_name, personas, t)
                """

                # Visualization of the results of a detection.
                # note: perform the detections using a higher threshold
                vis_util.visualize_boxes_and_labels_on_image_array(
                    color_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=.50)

                cv2.imshow('object detection', cv2.resize(color_frame, (800,600)))
                #output_rgb = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                out.write(cv2.resize(color_frame,(800,600)))

                if cv2.waitKey(1) == 27:
                    out.release()
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            out.release()
            cap.release()
            cv2.destroyAllWindows()
            


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def predict_single_image(image, output_path='output/'):
  image_name = image.split('/')[-1].split('\\')[-1]
  IMAGE_SIZE = (12, 8)
  detection_graph, category_index = create_graph()
  image = Image.open(image)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)

  personas = get_number_persons(output_dict['detection_scores'])
  if(personas>=1):
    print(personas)

  #plt.figure(figsize=IMAGE_SIZE)
  #plt.imshow(image_np)
  new_img = Image.fromarray(image_np, mode='RGB' )
  new_img.save(output_path+image_name)




def _main_(args):
  input_path   = args.input
  output_path  = args.output
  makedirs(output_path)
  if 'webcam' in input_path:
    detect_in_cam()
  elif input_path[-4:] == '.mp4': # do detection on a video
    detect_in_video(input_path, output_path)
  else:
    predict_single_image(input_path, output_path)

"""
    if args.cam is not None:
      detect_in_cam(args.cam)
    elif args.image is not None:
      predict_single_image(args.image)
    elif args.video is not None:
        detect_in_video(args.video, args.output)
    else:
        detect_in_cam()
"""

if __name__ == '__main__':
  argparser = argparse.ArgumentParser(description='Predict with a trained tensorflow model')
  argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
  argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
  args = argparser.parse_args()
  _main_(args)
  """
    parser = ArgumentParser()
    parser.add_argument("-v", "--video", nargs="?", dest="video", help="Video you want to process")
    parser.add_argument("-o", "--output", nargs="?", dest="output", help="where to save the video")
    parser.add_argument("-c", "--cam", nargs="?", dest="cam", help="where to save the video", type=int)
    parser.add_argument("-i", "--image", nargs="?", dest="image", help="Image to detect")
    args = parser.parse_args()
    """
  