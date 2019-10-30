import keras
from sklearn.metrics import classification_report
from settings import *
import tensorflow as tf

class AdditionalMetrics(keras.callbacks.Callback):
    def __init__(self, pred_datagen, samples, names, valid_df, dict_logs):

        super().__init__()
        self.dict_logs = dict_logs
        self.valid_data = pred_datagen.flow_from_dataframe(dataframe=valid_df,
                                                           target_size=(dim[0], dim[1]), color_mode='rgb',
                                                           batch_size=samples,
                                                           x_col='path',
                                                           y_col='label',
                                                           class_mode='categorical',
                                                           shuffle=False,
                                                           seed=random_state,
                                                           )
        self.names = names

    def on_epoch_end(self, epoch, logs):
        report = None
        logs.update(self.dict_logs)
        for i in range(len(self.valid_data)):
            x_test_batch, y_test_batch = self.valid_data.__getitem__(i)
            val_predict = (np.asarray(self.model.predict(x_test_batch))).round()
            val_targ = y_test_batch

            report = classification_report(y_true=val_targ, y_pred=val_predict, output_dict=True,
                                           target_names=self.names)

        for i in report.items():
            for z in i[1].items():
                logs[i[0] + '_' + z[0]] = z[1]
                tf.summary.scalar(name=str(i[0] + '_' + z[0]), data=z[1],step=epoch)
                print(str(i[0] + '_' + z[0]) + ' : {}'.format(z[1]))
