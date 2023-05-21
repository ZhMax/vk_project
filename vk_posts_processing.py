import pandas as pd
import numpy as np
import re

import os
import json

import torch
from transformers import BertTokenizer

FOLDER = './vk_groups'
MODEL_PATH = './models'

group_names = {'Мой город Пермь': 'vikiperm', '59.RU': 'news59ru', 'BusinessNews': 'gazetabc'}
name_of_loaded_model = 'model_v5_3.pt'

class NewsClassification(object):
    
    regex = re.compile("[А-Яа-я.,-]+")
    
    def __init__(self, df, model_name='DeepPavlov/rubert-base-cased-sentence'):
        self.df = df.copy()
        
        self.texts = None

        self.model = None
        self.cat_to_idx = None
        self.device = None
        self.model_name = model_name



    def transform(self):
        """
        Метод, который удаляет из текстов статей все лишние символы и выполняет их токенезацию
        """
        
        self.df['text'] = self.df['text'].apply(lambda x: self._words_only(x))
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.texts = [self.tokenizer(text, 
                               padding='max_length', max_length = 150, truncation=True,
                               return_tensors='pt') for text in self.df['text']]

    def load_model(self, path, device='cpu'):
        """
        Метод для загрузки модели
        """

        self.model = torch.load(path)
        self.device = device
        self.model = self.model.to(device)
        self.cat_to_idx = pd.read_csv(f'{MODEL_PATH}/categories_and_indexes.csv',
                                      header=None, names=['topic', 'topic_id'])

    def get_posts_topic(self):
        """
        Метод для определения темы поста
        """

        #Формирование входных данных
        mask = []
        input_id = []
        for data_input in self.texts:
            mask.append(data_input['attention_mask'])
            input_id.append(data_input['input_ids']) 

        input_id = torch.vstack(input_id)
        mask = torch.vstack(mask)

        #Применение модели к определению темы поста
        self.model.eval()
        output = self.model(input_id, mask)
        topics_proba, topics_id = torch.softmax(output[0].to('cpu'), dim=1).max(dim=1)
        topics_proba = topics_proba.detach().numpy()
        topics_id = topics_id.detach().numpy()

        #Добавление столбцов с темой постов и вероятностью, с которой модель их определила
        self.df['topic_id'] = topics_id
        self.df.reset_index(inplace=True)
        self.df = self.df.merge(self.cat_to_idx, on='topic_id', how='inner')
        self.df['topic_proba'] = topics_proba
        self.df.set_index('index', inplace=True)
        self.df.index.name = None

    def _words_only(self, text):
        """
        Исключение лишних символов из документа
        """

        try:
            res = self.regex.findall(text)
            res = ' '.join(res)
            return res
        except:
            return []
        

def get_vk_group_posts(group_name, folder=FOLDER):
    """
    Процедура для считывания постов
    """

    if os.path.exists(f'{folder}/{group_name}'):
        path = f'{folder}/{group_name}/{group_name}_posts.json'
        df = pd.read_json(path, orient='index', convert_dates=False)
        return df
    else:
        print('Директория {group_name} не существует')


def write_posts_with_topic(df, group_name, folder=FOLDER):
    """
    Процедура для записи постов после определения их темы
    """

    if os.path.exists(f'{folder}/{group_name}'):
        path = f'{folder}/{group_name}/{group_name}_posts_with_topic.csv'
        df.to_csv(path)
    else:
        print('Директория {group_name} не существует')


def main():
    
    for group_name in group_names.values():
        df_posts = get_vk_group_posts(group_name)

        news_classifier = NewsClassification(df_posts)
        news_classifier.transform()
        news_classifier.load_model(f'{MODEL_PATH}/{name_of_loaded_model}')
        news_classifier.get_posts_topic()

        df_posts[['topic', 'topic_proba']] = news_classifier.df[['topic', 'topic_proba']]
        write_posts_with_topic(df_posts, group_name)

if __name__ == '__main__':
    main()

