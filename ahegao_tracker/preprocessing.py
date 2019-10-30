from helper import *
import dlib
import tensorflow.compat.v1 as tf


class Preprocess:
    def __init__(self, args, graph=None):
        self.args = args
        if args.type_model == 'tensorflow':
            sess = tf.Session()
            sess.graph.as_default()
            tf.import_graph_def(graph, name='')
            self.sess = sess
        self.CONF_THRESHOLD = args.threshold_conf
        self.NMS_THRESHOLD = args.threshold_nms
        self.IMG_WIDTH = 416
        self.IMG_HEIGHT = 416
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
        # Default colors
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        # TODO : make scale age in more data science way
        self.scale_age = [1, 0.8, 0.5, 0.3]

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
                boxes = [x, y, right, bottom]
                pred = classes[classId - 1]
                if pred == 'face':
                    faces_box.append(boxes)
                else:
                    people_box.append(boxes)
                self.draw_predict(img, score, x, y, right, bottom, x, y, pred)
        return faces_box, people_box

    def adjust_gamma(self, image, gamma=1.0):

        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)

    def get_eyes_nose_dlib(self,shape):
        nose = shape[4][1]
        left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
        left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
        right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
        right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
        return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)

    def shape_to_normal(self,shape):
        shape_normal = []
        for i in range(0, 5):
            shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
        return shape_normal

    def rotate_point(self, origin, point, angle):
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def is_between(self, point1, point2, point3, extra_point):
        c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (
                extra_point[0] - point1[0])
        c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (
                extra_point[0] - point2[0])
        c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (
                extra_point[0] - point3[0])
        if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
            return True
        else:
            return False

    def distance(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def cosine_formula(self, length_line1, length_line2, length_line3):
        cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
        return cos_a

    def face_alignment(self,img):
        h, w, _ =img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        if len(rects) > 0:
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = self.shape_to_normal(shape)
                nose, left_eye, right_eye = self.get_eyes_nose_dlib(shape)
                center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
                center_pred = (nose[0]+5,-h)
                length_line1 = self.distance(center_of_forehead, nose)
                length_line2 = self.distance(center_pred, nose)
                length_line3 = self.distance(center_pred, center_of_forehead)
                cos_a = self.cosine_formula(length_line1, length_line2, length_line3)
                angle = np.arccos(cos_a)
                rotated_point = self.rotate_point(nose, center_of_forehead, angle)
                rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
                if self.is_between(nose, center_of_forehead, center_pred, rotated_point):
                    angle = np.degrees(-angle)
                else:
                    angle = np.degrees(angle)
                img = Image.fromarray(img)
                img = np.array(img.rotate(angle))
            return img
        else:
            print("Couldn't determine face")
            return img

    def estimate_brightness(self, img):
        im = Image.fromarray(img).convert('L')
        stat = ImageStat.Stat(im)
        return stat.rms[0]

    def get_outputs_names(self, net):
        layers_names = net.getLayerNames()
        return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]



    def draw_predict(self, frame, conf, left, top, right, bottom, center_y, center_x, prediction):
        # Draw a bounding box.

        if prediction == 'face' and self.args.type_model != 'tensorflow':
            rad = right - center_x
            rad_fin = rad if rad > 0 else -rad
            cv2.circle(frame, center=(center_x, center_y), radius=int(rad_fin * 1.3), color=self.COLOR_RED,
                       thickness=3)
        #
        elif prediction == 'face':
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
