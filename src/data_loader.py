import tensorflow as tf 
from transformers import AutoTokenizer
import pandas as pd
import config 

def preparing_bert_training_datasets(train_path):

    train_data = pd.read_csv(train_path, usecols=["id", "text", "target"])

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    train_tokenized = tokenizer(train_data["text"].to_list(), padding="max_length", truncation=True, max_length=config.SEN_LENGTH, return_tensors="tf")

    labels = train_data["target"].values


    base_dataset = tf.data.Dataset.from_tensor_slices((
                            {"input_tokens" : train_tokenized["input_ids"], "input_masks" : train_tokenized["attention_mask"]},
                            labels))


    train_size = int(0.85 * len(train_data))
    val_size = len(train_data) - train_size

    train_dataset = base_dataset.take(train_size)
    val_dataset = base_dataset.skip(train_size)

    train_dataset = train_dataset.shuffle(1000).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset



def preparing_bert_inference_datasets(test_path):

    test_data = pd.read_csv(test_path, usecols=["id", "text"])
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    test_tokenized = tokenizer(test_data["text"].to_list(), padding="max_length", truncation=True, max_length=config.SEN_LENGTH, return_tensors="tf")

    test_dataset = tf.data.Dataset.from_tensor_slices(
                    {"input_tokens" : test_tokenized["input_ids"], "input_masks" : test_tokenized["attention_mask"]})
    
    test_dataset = test_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return test_dataset