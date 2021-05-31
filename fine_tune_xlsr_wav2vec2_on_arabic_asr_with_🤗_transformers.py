from datasets import load_dataset, load_metric
from datasets import ClassLabel
import random
import pandas as pd
import re
import pandas as pd
from datasets import Dataset
from lang_trans.arabic import buckwalter
import json
#from google.colab import drive
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
import librosa
import numpy as np
import torchaudio
import IPython.display as ipd
import numpy as np
import random
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import TrainingArguments
from transformers import Wav2Vec2ForCTC
from transformers import Trainer


chars_to_delete = ['t','☭','ۖ', 'e','g','چ','ڨ']
chars_to_ignore_regex = '[\,\.\!\-\;\:\"\“\%\‘\”\�\;\—\؛\_\'ْ\،\'ُ\؟\ـ\?\'ۚ\'ٌ\'َ\'ّ\'ۖ\»\«]'

no_rows_drop  = 4000
def preporcess_text(dataset):
    
    global no_rows_drop

    df = dataset.to_pandas()
   
    df = df.drop(df.index[no_rows_drop:])

    no_rows_drop = 1000

    for idx , row in df.iterrows():

        df.at[idx,'sentence'] = normalize(row["sentence"]) 

        for c in chars_to_delete:
            if c in row["sentence"]:
                df.drop(idx,inplace=True)
                break
    
    return Dataset.from_pandas(df)


def transliterate(batch):

    batch["sentence"] = buckwalter.transliterate(batch["sentence"])     
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  
  if 'ٰ' in vocab:
    vocab.remove('ٰ')
  
  if 'ﻻ' in vocab:
    vocab.remove('ﻻ')
  
  return {"vocab": [vocab], "all_text": [all_text]}

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch

def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch


def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

def compute_metrics(pred):
    pred_logits = pred.predictions

    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)

    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)


    return {"wer": wer}

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


chars_to_ignore_regex = '[\,\.\!\-\;\:\"\“\%\‘\”\�\;\—\؛\_\'ْ\،\'ُ\؟\ـ\?\'ۚ\'ٌ\'َ\'ّ\'ۖ\»\«]'



arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)




def normalize_arabic(text):
    text = re.sub("[إأﺃآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub("ک", "ك", text)
    text = re.sub("ھ", "ه", text)
    text = re.sub("ی", "ي", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    text = re.sub(chars_to_ignore_regex,'',text)
    return text

def normalize(text):
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    return text





if __name__ == "__main__":

    #drive.mount('/content/gdrive/')
    #out_dir = "/content/gdrive/MyDrive/wav2vec2-large-xlsr-arabic-demo-v5"


    out_dir = "/home/mohammed/spereco/model"

    # load dataset
    common_voice_train = load_dataset("common_voice", "ar", split="train")
    common_voice_test = load_dataset("common_voice", "ar", split="test")


    columns_to_remove = ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"]
    common_voice_train = common_voice_train.remove_columns(columns_to_remove)
    common_voice_test = common_voice_test.remove_columns(columns_to_remove)

    # normalize arabic text
    
    print("train shape : " + str(common_voice_train.to_pandas().shape))
    print("test shape : " + str(common_voice_test.to_pandas().shape))



    common_voice_train = preporcess_text(common_voice_train)
    common_voice_test = preporcess_text(common_voice_test)

    
    print("train shape : " + str(common_voice_train.to_pandas().shape))
    print("test shape : " + str(common_voice_test.to_pandas().shape))



    common_voice_train = common_voice_train.map(transliterate)
    common_voice_test = common_voice_test.map(transliterate)


    vocab_train = extract_all_chars(common_voice_train)
    vocab_test = extract_all_chars(common_voice_test)


    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict['|'] = vocab_dict[' ']
    del vocab_dict[' ']

    vocab_dict['[UNK]'] = len(vocab_dict)
    vocab_dict['[PAD]'] = len(vocab_dict)


    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]",
    pad_token="[PAD]", word_delimiter_token="|")
    
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000,
    padding_value=0.0, do_normalize=True, return_attention_mask=True)

    wer_metric = load_metric("wer")
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(out_dir)


    common_voice_train = common_voice_train.map(speech_file_to_array_fn,
    remove_columns=common_voice_train.column_names)
    
    common_voice_test = common_voice_test.map(speech_file_to_array_fn,
    remove_columns=common_voice_test.column_names)


    common_voice_train = common_voice_train.map(resample, num_proc=4)
    common_voice_test = common_voice_test.map(resample, num_proc=4)


    common_voice_train = common_voice_train.map(prepare_dataset,
    remove_columns=common_voice_train.column_names,
    batch_size=8, num_proc=4, batched=True)


    common_voice_test = common_voice_test.map(prepare_dataset,
    remove_columns=common_voice_test.column_names,
    batch_size=8, num_proc=4, batched=True)


    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", 
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )

    model.freeze_feature_extractor()


    training_args = TrainingArguments(
        output_dir=out_dir,
        group_by_length=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=False,
        save_steps=5000,
        eval_steps=500,
        logging_steps=400,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=1,
    )
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=common_voice_train,
        eval_dataset=common_voice_test,
        tokenizer=processor.feature_extractor,
    )


    trainer.train()


    trainer.save_model(out_dir)
