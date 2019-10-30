from helper import *


class Ahegao:
    def __init__(self, args):
        self.args = args
        self.graph = None
        if self.args.type_model == 'tensorflow':
            self.graph = self.read_tensorflow()
        else:
            self.net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.qua_conf = args.qua_conf
        self.preproc = prep.Preprocess(args, self.graph)
        self.model, self.model_emotions = load_models()
        self.prediction = None
        self.predicted_gender = None
        self.predicted_age = None
        self.prediction_gender = []
        self.prediction_age = []
        self.prediction_emo = []
        self.average_pred_emo = []
        self.average_pred_age = []
        self.average_pred_gender = []
        self.final_predictions = []
        self.past_coordinates = []
        self.conf_counter = 0
        self.classes_emo = {0: 'Ahegao', 1: 'Angry', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
        self.IMG_HEIGHT = 128
        self.IMG_WIDTH = 128
        self.dim = (self.IMG_WIDTH, self.IMG_HEIGHT, 3)

    def read_tensorflow(self):
        with tf.gfile.FastGFile('ssd2.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    def make_inputs(self):

        inputs_lis = []
        if self.args.type_model != 'tensorflow':
            crop_faces = [(self.faces[i][0], self.faces[i][1], self.faces[i][2] + self.faces[i][0],
                           self.faces[i][3] + self.faces[i][1]) for i in
                          range(len(self.faces))]
        else:
            crop_faces = [(self.faces[i][0], self.faces[i][1], self.faces[i][2],
                           self.faces[i][3]) for i in
                          range(len(self.faces))]

        for i in range(len(crop_faces)):
            img = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            img_ = img[crop_faces[i][1]:crop_faces[i][3], crop_faces[i][0]:crop_faces[i][2], :]

            img=self.preproc.face_alignment(img_)
            brightness=self.preproc.estimate_brightness(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if brightness<=self.args.brightness_threshold:
               img=self.preproc.adjust_gamma(img,1.5)
            inputs_lis.append(img)
        return inputs_lis, crop_faces

    def predict_age_gender_emotions(self, i):

        img = i

        np_image = cv2.resize(img, dsize=(128, 128))

        np_image = np.array(np_image).astype('float32') / 255
        np_image = transform.resize(np_image, (128, 128, 3))
        np_image = np.expand_dims(np_image, axis=0)

        prediction = self.model_emotions.predict(np_image)

        self.prediction_emo.append(prediction)
        prediction = self.model.predict(np_image)
        self.prediction_gender.append(prediction[0][0])
        ages = np.arange(0, 21).reshape(21, 1)
        predicted_age = prediction[1].dot(ages).flatten()
        self.prediction_age.append(predicted_age * 4.76)

    def labels_func(self):
        for i in self.overall_prediction:
            font = cv2.FONT_HERSHEY_DUPLEX
            bottomLeftCornerOfText = (i[3][0] - 20, i[3][1] - 20)
            fontScale = 0.5
            fontColor = self.preproc.COLOR_GREEN
            lineType = 2
            prediction_ = 'sex : {} age :{} emotion: {} {}%'.format(i[2], i[1],
                                                                    i[0], i[4])
            cv2.putText(self.frame, prediction_,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

    def put_labels(self, crop_faces,  average_emo, average_age, average_gender):

        if self.conf_counter % self.qua_conf == 0:
            self.get_average_prediction(crop_faces, average_emo, average_age, average_gender)

        self.labels_func()

    def get_average_prediction(self, crop_faces, average_pred_emo, average_pred_age, average_pred_gender,constant=0.8):
        self.overall_prediction = []
        tmp_emo = np.zeros(average_pred_emo[0][0].shape)
        tmp_gender = np.zeros(average_pred_gender[0][0].shape)
        tmp_age = np.zeros(average_pred_age[0][0].shape)
        if self.conf_counter - self.qua_conf < 0:
            qua_conf = 1
        else:
            qua_conf = self.qua_conf
        for counter1, tmp in enumerate(crop_faces):
            for i in range(len(average_pred_emo)):
                tmp_emo = np.add(average_pred_emo[i][counter1], tmp_emo)
                tmp_gender = np.add(average_pred_gender[i][counter1], tmp_gender)
                tmp_age = np.add(average_pred_age[i][counter1], tmp_age)
            tmp_emo = tmp_emo / qua_conf
            tmp_age = tmp_age / (qua_conf *constant)
            tmp_gender = tmp_gender / qua_conf
            score = round((np.max(tmp_emo, axis=1) / np.sum(tmp_emo, axis=1))[0] * 100, 2)
            predicted_emo = self.classes_emo[np.argmax(tmp_emo, axis=1)[0]]
            predicted_gender = 'female' if np.argmax(tmp_gender) == 0 else 'male'
            predicted_age = int(tmp_age[0])
            past_coordinates = tmp
            self.overall_prediction.append((predicted_emo, predicted_age, predicted_gender, past_coordinates, score))

    def end_to_end(self):

        self.prediction_age = []
        self.prediction_gender = []
        self.prediction_emo = []

        inputs_lis, crop_faces = self.make_inputs()
        for i in inputs_lis:
            self.predict_age_gender_emotions(i)
        self.average_pred_age.append(self.prediction_age)
        self.average_pred_gender.append(self.prediction_gender)
        self.average_pred_emo.append(self.prediction_emo)
        self.put_labels(crop_faces, self.average_pred_emo.copy(), self.average_pred_age.copy(), self.average_pred_gender.copy())
        inputs_lis.clear()
        self.average_pred_emo.clear()
        self.average_pred_gender.clear()
        self.average_pred_age.clear()
        self.conf_counter += 1

    def run(self):

        wind_name = 'Ahegao'
        cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

        self.class_list = get_classes(self.args)

        if self.args.video:
            if not os.path.isfile(self.args.video):
                print("[!] ==> Input video file {} doesn't exist".format(self.args.video))
                sys.exit(1)
            cap = cv2.VideoCapture(self.args.video)

        else:

            cap = cv2.VideoCapture(self.args.src)

        n = 0
        while True:

            has_frame, self.frame = cap.read()

            if not has_frame:
                print('[i] ==> Done processing!!!')
                cv2.waitKey(1000)
                break
            if self.args.skip != n:
                n += 1
                continue
            n = 0
            if self.args.type_model == 'tensorflow':
                self.faces, people = self.preproc.tf_detect(img=self.frame, classes=self.class_list)
            else:
                blob = cv2.dnn.blobFromImage(self.frame, 1 / 255, (self.IMG_WIDTH, self.IMG_HEIGHT),
                                             [0, 0, 0], 1, crop=False)

                self.net.setInput(blob)

                outs = self.net.forward(self.preproc.get_outputs_names(self.net))

                self.faces, people, ids, indices = self.preproc.post_process(self.frame, outs, self.class_list,
                                                                             )

            if self.faces:
                self.end_to_end()

            print('[i] ==> # detected  objects: {}'.format(len(self.faces) + len(people)))
            print('#' * 60)

            info = [
                ('number of objects detected', '{}'.format(len(self.faces) + len(people)))
            ]

            for (i, (txt, val)) in enumerate(info):
                text = '{}: {}'.format(txt, val)
                cv2.putText(self.frame, text, (10, (i * 20) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.preproc.COLOR_RED, 2)

            cv2.imshow(wind_name, self.frame)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                print('[i] ==> Interrupted by user!')
                break

        cap.release()
        cv2.destroyAllWindows()

        print('==> All done!')
        print('***********************************************************')
