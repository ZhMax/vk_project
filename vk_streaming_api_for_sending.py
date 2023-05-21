import requests
import json
import os

TOKEN = '...' #Необходимо подставить токен доступа в vk
FOLDER = './vk_groups/'

#Доменные имена групп и количество записей
group_names = {'Мой город Пермь': 'vikiperm', '59.RU': 'news59ru', 'BusinessNews': 'gazetabc'}
items_count = 100

def get_wall_posts(group_name):
    """
    Функция для сбора постов со страницы группа group_name и сохранения их в .json файлы
    """

    url = f'https://api.vk.com/method/wall.get?domain={group_name}&count={items_count}&access_token={TOKEN}&v=5.131'

    #cчитываем посты с группы
    req = requests.get(url)
    src = req.json()
    posts = src['response']['items']

    #проверяем существует ли директория с именем группы
    if os.path.exists(f'{FOLDER}/{group_name}'):
        print('Директория {group_name} уже существует')
    else:
        os.makedirs(f'{FOLDER}/{group_name}')

    #собираем ID новых постов
    new_posts_dict = {}
    for post in posts:
        new_post_id = post['id']
        new_post_date = post['date']
        new_post_text = post['text']

        if len(new_post_text) > 0:
            new_posts_dict[new_post_id] = {'date': new_post_date, 'text': new_post_text}

    #создаем файл для хранения новостей или добавляем новые посты в имеющийся файл 
    if not os.path.exists(f'{FOLDER}/{group_name}/{group_name}_posts.json'):
        print('Создаем файл с постами')
        
        with open(f'{FOLDER}/{group_name}/{group_name}_posts.json', 'w', encoding='utf-8') as f:
            json.dump(new_posts_dict, f, indent=4, ensure_ascii=False)

    else:
        print('Добавляем посты в файл')
        
        with open(f'{FOLDER}/{group_name}/{group_name}_posts.json', 'r') as f:
            #извлекаем имеющиеся посты
            existing_posts = json.load(f)
            existing_posts_id = [int(id) for id in existing_posts]

        #выбираем новые посты
        posts_to_add = {}
        for item in new_posts_dict:
            if item not in existing_posts_id:
                posts_to_add[item] = new_posts_dict[item]
        
        with open(f'{FOLDER}/{group_name}/{group_name}_posts.json', 'w') as f:
            #добавляем записи в файл
            posts_to_add.update(existing_posts)
            json.dump(posts_to_add, f, indent=4, ensure_ascii=False)


def main():
    for group_name in group_names.values():
        get_wall_posts(group_name)


if __name__ == '__main__':
    main()