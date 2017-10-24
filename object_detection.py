import cv2, os, sys, tarfile, zipfile, io, contextlib, traceback
import numpy as np
import tensorflow as tf


#from utils import label_map_util
import label_map_util
import visualization_utils as vis_util


# # Model preparation 
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'data/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90

# Camera number. 0 for first device, 1 for second etc..
CAPTURE_DEVICE = 0



# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  
# Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map      = label_map_util.load_labelmap(PATH_TO_LABELS)
categories     = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Needed to suppress error from cv2 in cap.read()
@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:


    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()


# # Detection -------------------------------------------------------------
# Capturing camera with cv2
cap = cv2.VideoCapture(CAPTURE_DEVICE)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True: 
            # Needed to suppress error from cv2
            with stdchannel_redirected(sys.stderr, os.devnull):
                _, image_np = cap.read()
            image_np = cv2.flip(image_np, 1)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            cv2.imshow('Object Detection', cv2.resize(image_np, (1280,960)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              break
