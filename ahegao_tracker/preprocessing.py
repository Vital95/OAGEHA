from helper import *

import tensorflow as tf


class Preprocess:
    def __init__(self, args, graph=None):
        self.args=args
        if args.type_model == 'tensorflow':
                sess=tf.Session()
                sess.graph.as_default()
                tf.import_graph_def(graph, name='')
                self.sess = sess
        self.CONF_THRESHOLD = args.threshold_conf
        self.NMS_THRESHOLD = args.threshold_nms
        self.IMG_WIDTH = 416
        self.IMG_HEIGHT = 416

        # Default colors
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_BLACK=(0,0,0)
    def tf_detect(self, img, classes):

        faces_box = []
        people_box = []
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv2.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]
        out = self.sess.run([self.sess.graph.get_tensor_by_name('num_detections:0'),
                             self.sess.graph.get_tensor_by_name('detection_scores:0'),
                             self.sess.graph.get_tensor_by_name('detection_boxes:0'),
                             self.sess.graph.get_tensor_by_name('detection_classes:0')],
                            feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]

            if score > self.CONF_THRESHOLD:

                x = int(bbox[1] * cols)
                y = int(bbox[0] * rows)
                right = int(bbox[3] * cols)
                bottom = int(bbox[2] * rows)
                boxes = [x,y, right, bottom]
                pred = classes[classId - 1]
                if pred == 'face':
                    faces_box.append(boxes)
                else:
                    people_box.append(boxes)
                self.draw_predict(img, score, x, y, right, bottom, x, y, pred)
        return faces_box, people_box

    def preprocess_img(self, eyes):
        left_eye_x = eyes[0][0] + eyes[0][2] / 2
        left_eye_y = eyes[0][1] + eyes[0][3] / 2
        right_eye_x = eyes[1][0] + eyes[1][2] / 2
        right_eye_y = eyes[1][1] + eyes[1][3] / 2
        deg = (left_eye_y - right_eye_y) / (left_eye_x - right_eye_x)

        deg = atan(deg)
        deg = math.degrees(deg)

        return deg

    def get_outputs_names(self, net):
        # Get the names of all the layers in the network
        layers_names = net.getLayerNames()

        # Get the names of the output layers, i.e. the layers with unconnected
        # outputs
        return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box

    def draw_predict(self, frame, conf, left, top, right, bottom, center_y, center_x, prediction):
        # Draw a bounding box.

        if prediction == 'face' and self.args.type_model!='tensorflow':
             rad = right - center_x
             rad_fin = rad if rad > 0 else -rad
             cv2.circle(frame, center=(center_x, center_y), radius=int(rad_fin*1.3), color=self.COLOR_RED,
                        thickness=3)
        #
        elif prediction=='face':
            cv2.rectangle(frame, (left, top), (right, bottom), self.COLOR_RED, 3)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), self.COLOR_YELLOW, 3)
        text = '{:.2f}'.format(conf)

        # Display the label at the top of the bounding box
        label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        top = max(top, label_size[1])
        cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    self.COLOR_BLACK, 1)

    def post_process(self, frame, outs, classes):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only
        # the ones with high confidence scores. Assign the box's class label as the
        # class with the highest score.
        confidences = []
        boxes = []
        face_boxes = []
        people_boxes = []
        center = []
        class_ids = []
        if True:

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > self.CONF_THRESHOLD:
                        center_x = int(detection[0] * frame_width)
                        center_y = int(detection[1] * frame_height)
                        width = int(detection[2] * frame_width)
                        height = int(detection[3] * frame_height)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
                        center.append([center_x, center_y])
                        class_ids.append(class_id)

        # Perform non maximum suppression to eliminate redundant
        # overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONF_THRESHOLD,
                                   self.NMS_THRESHOLD)

        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            center_x = center[i][0]
            center_y = center[i][1]

            prediction = classes[class_ids[i]]
            if prediction == 'face':
                face_boxes.append(box)
            else:
                people_boxes.append(box)
            left, top, right, bottom = self.refined_box(left, top, width, height)

            self.draw_predict(frame, confidences[i], left, top, right, bottom, center_y, center_x, prediction)

        return face_boxes, people_boxes, class_ids, indices

    def refined_box(self, left, top, width, height):
        right = left + width
        bottom = top + height

        original_vert_height = bottom - top
        top = int(top + original_vert_height * 0.15)
        bottom = int(bottom - original_vert_height * 0.05)

        margin = ((bottom - top) - (right - left)) // 2
        left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

        right = right + margin

        return left, top, right, bottom
