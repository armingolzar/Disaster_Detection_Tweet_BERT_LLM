import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from transformers import TFBertModel
import config

class Disaster_Detection_BERT_Model():

    def __init__(self):
        pass

    
    @staticmethod
    def building_network():

        bert_model = TFBertModel.from_pretrained(config.BERT_MODEL_NAME)

        for layer in bert_model.layers:
            layer.trainable = False

        input_tokens = Input(shape=(config.SEN_LENGTH,), dtype=tf.int64, name="input_tokens")
        input_masks = Input(shape=(config.SEN_LENGTH,), dtype=tf.int64, name="input_masks")

        bert_embedding = bert_model(input_tokens, attention_mask=input_masks) # with [1] at the end of this line we could get cls_embedding
        cls_embedding = bert_embedding.pooler_output

        drop1 = Dropout(0.1, name="drop1")(cls_embedding)
        dense1 = Dense(256, activation="relu", name="dense1")(drop1)
        drop2 = Dropout(0.1, name="drop2")(dense1)
        dense2 = Dense(128, activation="relu", name="dense2")(drop2)
        drop3 = Dropout(0.1, name="drop3")(dense2)
        dense3 = Dense(32, activation="relu", name="dense3")(drop3)

        output = Dense(1, activation="sigmoid", name="output")(dense3)

        disaster_detection_model = Model(inputs=[input_tokens, input_masks], outputs=output)
        disaster_detection_model.summary()

        return disaster_detection_model
    


    @staticmethod
    def compil_model(model, optimizer="Adam", learning_rate=1e-3):

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)
            optimizer.learning_rate = learning_rate

        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model
    

    @staticmethod
    def training_model(model, train_data, epochs, validation_data):

        history = model.fit(train_data, epochs=epochs, validation_data=validation_data)
    
        return model, history
    





