import os
import cv2
import numpy as np
import face_recognition
import pickle
import time
import sys
import tensorflow as tf
import collections
import sys
import argparse
#from utils import label_map_util
#from utils import visualization_utils_color as vis_util

#windowName = "FaceRecognitionDemo"
known_face_encodings = []
known_face_names = []

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Face Recognition on Jetson TX2")
    parser.add_argument("--usb", help="use USB webcam",
                        action="store_true")
    parser.add_argument("--addperson", help="person name.xxx",
                        type=str)
    args = parser.parse_args()
    return args


# for check performance
def add_face(img_path):
    global known_face_encodings, known_face_names
    if not os.path.isfile(img_path):
        print('path {} do not exist!'.format(img_path))
        exit(1)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    image_ = face_recognition.load_image_file(img_path)
    face_encodings = face_recognition.face_encodings(image_,num_jitters=10)
    if face_encodings:
        face_encoding_ = face_encodings[0]
        norm_encoding = face_encoding_ / np.linalg.norm(face_encoding_)
        known_face_encodings.append(norm_encoding)
        known_face_names.append(img_name)
    else:
        print('encoding for new face {} failed'.format(img_path))
        exit(1)
    
#this function will read all persons' training images, detect face from each image and will 
#return two lists of exactly same size, one list of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    global known_face_encodings, known_face_names
    pic_names = [ item for item in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, item)) ]
    for name in pic_names:
        print(name)
        namepath = os.path.join(data_folder_path,name)
        for pic in os.listdir(namepath):
            if pic.startswith("."):
                continue;
            image_path = os.path.join(namepath,pic)
            print(image_path)
            image_ = face_recognition.load_image_file(image_path)
            face_encoding_ = face_recognition.face_encodings(image_,num_jitters=10)[0]
            norm_encoding = face_encoding_ / np.linalg.norm(face_encoding_)
            known_face_encodings.append(norm_encoding)
            known_face_names.append(name)
#    return known_face_names, known_face_encodings
    for enc1 in known_face_encodings:
        for enc2 in known_face_encodings:
            print(np.dot(enc1, enc2)) 
        print('\n')
            
def save_known_faces(data_file):
    global known_face_encodings, known_face_names
    with open(data_file, 'w') as f:
       pickle.dump([known_face_names, known_face_encodings], f)
    print ('Known faces saved to '+data_file)

def load_known_faces(data_file):
    global known_face_encodings, known_face_names
    with open(data_file) as f:
        known_face_names, known_face_encodings = pickle.load(f)
    print ('Known faces loaded from '+data_file)
#    return known_face_names, known_face_encodings

def find_face_location_on_image_array(image, boxes, scores, min_score_thresh=.7):
  """
  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    min_score_thresh: minimum score threshold for a box to be visualized
  """
  face_locations = []
  im_height, im_width, _ = image.shape
  for i in range(boxes.shape[1]):
    if scores is None or scores[0][i] >= min_score_thresh:
      ymin, xmin, ymax, xmax = tuple(boxes[0][i].tolist())
      (top, right, bottom, left) = (int(ymin * im_height), int(xmax * im_width), int(ymax * im_height), int(xmin * im_width))
      face_locations.append((top, right, bottom, left))
    elif scores[0][i] < min_score_thresh:
      break
  return face_locations


def reg_faces(cap):
    global known_face_encodings, known_face_names
#    cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    #cap = cv2.VideoCapture ("v4l2src device=/dev/video0 ! video/x-raw, width=1920, height=1080, format=(string)RGB ! videoconvert ! appsink")
#    cap = cv2.VideoCapture(0)
    if cap.isOpened():        
        windowName = "Face Detection"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,1280,720)
        #cv2.setWindowTitle(windowName,"Face Detection")
        font = cv2.FONT_HERSHEY_DUPLEX
        process_this_frame = True
        
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
        # List of the strings that is used to add correct label for each box.
#        PATH_TO_LABELS = './protos/face_label_map.pbtxt'
#        NUM_CLASSES = 2
#        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#        category_index = label_map_util.create_category_index(categories)

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(graph=detection_graph, config=config) as sess:
                while True:
                    if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
                        break;
                    start_time = time.time()
                    ret_val, frame = cap.read();
                    
                    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
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
                    
                    face_locations = find_face_location_on_image_array(frame, boxes, scores, min_score_thresh=.7)
                    
                    if face_locations is None:
                        continue
                    face_encodings = face_recognition.face_encodings(frame, face_locations, num_jitters=10)
            
                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        norm_encoding = face_encoding / np.linalg.norm(face_encoding)
                        dist_list = face_recognition.face_distance(known_face_encodings, norm_encoding)
                        dist_min = min(dist_list)
                        if dist_min < 0.32:
                            name = known_face_names[np.argmin(dist_list)]
                        else:
                            name = ""
                        face_names.append(name)
                    # Display the results
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        # Draw a box around the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        if top > 22:
                            cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (0, 255, 0), 1)
                        else:
                            cv2.putText(frame, name, (left + 6, bottom + 22), font, 1.0, (0, 255, 0), 1)
                    
                    cv2.putText(frame,"FPS:{:.2f}".format(1.0 / (time.time() - start_time)),(10,30), font, 1.0, (0,0,0),1) 
                    cv2.imshow(windowName, frame)
                
                    process_this_frame = not process_this_frame
                    key = cv2.waitKey(10)
                    if key == ord('Q') or key == ord('q'): # quit program
                        break
 
    else:
        print ("camera open failed")
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    #prepare_training_data('/Users/xzhang/Pictures/FaceRec2')
    load_known_faces('known_faces.pkl')
    args = parse_args()
    if args.addperson:
        add_face(args.addperson)
        save_known_faces('known_faces.pkl')
    else:
        if args.use:
            cap = cv2.VideoCapture(0)
        else: # by default, use the Jetson onboard camera
            cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,"
            "format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx"
            " ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
        reg_faces(cap)
    
    

