# !/usr/bin/python
# -*- coding: utf-8 -*-

# загружаем нужные библиотеки

import requests
from urllib.parse import urlencode
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np
import math as mth
from scipy import stats as st

# получаем данные с Яндекс.Диска

# подготавливаем ссылки
base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.by/i/ucbEk8NydnwLgw'

# получаем загрузочную ссылку
final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']
download_response = requests.get(download_url)

# открываем файл
with open('Тестовое данные.xlsx', 'wb') as f:
    f.write(download_response.content)

# читаем файл и записываем в переменную
data = pd.read_excel('Тестовое данные.xlsx')

# приводим наименование колонок к нижнему регистру для удобства

data.columns = data.columns.str.lower()

# поскольку данные представляют только целые числа, то изменим тип с float на int

data.loc[:, data.columns != 'ab_cohort'] = (
    data.loc[:, data.columns != 'ab_cohort'].astype('int')
)

# создаем функцию для первичного обзора данных

def first_check(dataset):

    print('Первые 10 строк таблицы \n')
    print(dataset.head(10))

    print('-'*50)
    print('Последние 10 строк таблицы \n')
    print(dataset.tail(10))

    print('-'*50)
    print('Информация о таблице \n')   
    print(dataset.info())

    print('-'*50)
    print('Характеристики чисел в таблице \n')  
    print(dataset.describe())

    print('-'*50)
    print('Наименование колонок \n')  
    print(dataset.columns)
    
    print('-'*50)
    print('Количество дубликатов: {}\n'.format(dataset.duplicated().sum()))

if __name__ == '__main__':
    first_check(data)

# проверяем равномерность распределения пользователей по когортам
if __name__ == '__main__':
    print(data.groupby('ab_cohort').agg({'user_id' : 'nunique'}))

# добавим колонку с обозначением времени на восстановление жизни

data['heart_time'] = np.where(data['ab_cohort'] == "A", 30, 3)

# поскольку данные о стартах и победах уровней представляют собой накопленные
# данные, создадим отдельную таблицу с количеством пройденных уровней

levels = (
    data.groupby(['user_id', 'ab_cohort'])
    .agg({
        'maxlevelpassed' : 'max',
        'countallstart' : 'max',
        'countallfinish' : 'max',
        'countcleanstart' : 'max',
        'countcleanfinish' : 'max',
        'heart_time' : 'max'})
    .reset_index()
)

# добавляем столбцы с количеством стартов и побед с применением бонусов
levels['notcleanstart'] = levels['countallstart'] - levels['countcleanstart']
levels['notcleanfinish'] = levels['countallfinish'] - levels['countcleanfinish']

if __name__ == '__main__':
    print(levels.head()) # смотрим результат

# проверим, нет ли записей, в которых количество стартов равно нулю,
# а количество пройденных уровней больше нуля

if __name__ == '__main__':
  
    print('Количество ошибочных записей:\n{}'.format(
        levels.loc[
            (levels['countallstart'] == 0) & (levels['maxlevelpassed'] != 0)
            ]['ab_cohort'].value_counts()
            )
    )

    print()

    for group in list(levels['ab_cohort'].unique()):
        print('Процент ошибочных записей в группе {}: {:.2%}'.format(
            group,
            len(levels.loc[
                (levels['countallstart'] == 0)
                & (levels['maxlevelpassed'] != 0)
                & (levels['ab_cohort'] == group)])
            / len(levels.query('ab_cohort == @group'))
            )
        )

# возможно, в записях ошибка;
# в данном случае замена на меданные значения может исказить реальные результаты,
# поэтому таких пользователей следует удалить;
# создаем список пользователей на удаление

users_for_delete = levels.loc[
    (levels['countallstart'] == 0) & (levels['maxlevelpassed'] != 0)
    ][['user_id', 'ab_cohort']]

# определяем пользователей, которые должны быть сохранены для анализа

levels = levels.merge(
    users_for_delete,
    how='left',
    left_on=['user_id', 'ab_cohort'],
    right_on=['user_id', 'ab_cohort'],
    indicator='i'
    )

# удаляем лишних пользователей и колонку с индикатором

levels = (
    levels.query('i == "left_only"').copy()
    .drop(columns='i').sort_values(by='ab_cohort')
)

# смотрим, сколько пользователей осталось в каждой группе

if __name__ == '__main__':
    print(levels['ab_cohort'].value_counts())

# проверяем, нет ли ошибки в записи данных, когда количество завершений
# уровней превышает количество стартов

if __name__ == '__main__':
    print('Превышение всех побед над всеми стартами: {}'.format(len(
        levels.loc[levels['countallstart'] < levels['countallfinish']]
        )
    ))

    print('Превышение чистых побед на чистыми стартами: {}'.format(len(
        levels.loc[levels['countcleanstart'] < levels['countcleanfinish']]
        )
    ))

    print('Превышение побед с бонусами над стартами с бонусами: {}'.format(len(
        levels.loc[levels['notcleanstart'] < levels['notcleanfinish']]
        )
    ))

# заменим неверные данные о победах с бонусами на количество стартов с бонусами

levels['notcleanfinish'] = np.where(
    levels['notcleanstart'] < levels['notcleanfinish'],
    levels['notcleanstart'],
    levels['notcleanfinish']
    )

# проверяем

if __name__ == '__main__':
    print('Превышение побед с бонусами над стартами с бонусами: {}'.format(len(
        levels.loc[levels['notcleanstart'] < levels['notcleanfinish']]
        )
    ))

# добавим колонки с долей стартов и побед с бонусами для каждого пользователя

levels['notcleanstart_percent'] = round(
    levels['notcleanstart'] / levels['countallstart'] * 100, 2).fillna(0)

levels['notcleanfinish_percent'] = round(
    levels['notcleanfinish'] / levels['countallfinish'] * 100, 2).fillna(0)

# создадим датасет для анализа получения и трат золота;
# поскольку данные о получении золота за прохождение уровней отсутствует,
# проверка превышения сумм трат над суммой получения не требуется

gold = (
    data.groupby(['user_id', 'ab_cohort'])
    .agg({
        'get_ads' : 'sum',
        'get_chapter' : 'sum',
        'get_buy' : 'sum',
        'get_faceb' : 'sum',
        'get_teaml' : 'sum',
        'get_teamt' : 'sum',
        'spend_bonlives' : 'sum',
        'spend_bonus' : 'sum',
        'spend_boost' : 'sum',
        'spend_lives' : 'sum',
        'spend_moves' : 'sum',
        'spend_teamc' : 'sum',
        'heart_time' : 'max'})
    .reset_index()
)

# определяем пользователей, которые должны быть сохранены для анализа

gold = gold.merge(
    users_for_delete,
    how='left',
    left_on=['user_id', 'ab_cohort'],
    right_on=['user_id', 'ab_cohort'],
    indicator='i'
    )

# удаляем лишних пользователей и колонку с индикатором

gold = (
    gold.query('i == "left_only"').copy()
    .drop(columns='i').sort_values(by='ab_cohort')
)

if __name__ == '__main__':
    print(gold.head()) #смотрим результат

# создаем датасет для анализа бизнес-показателей

profiles = data[['user_id', 'ab_cohort', 'retention', 'countbuy', 'sumrevenue']]

# определяем пользователей, которые должны быть сохранены для анализа

profiles = profiles.merge(
    users_for_delete,
    how='left',
    left_on=['user_id', 'ab_cohort'],
    right_on=['user_id', 'ab_cohort'],
    indicator='i'
    )

# удаляем лишних пользователей и колонку с индикатором

profiles = (
    profiles.query('i == "left_only"').copy()
    .drop(columns='i').sort_values(by='ab_cohort')
)

# определяем пользователей, совершивших покупки

payers = (
    profiles.groupby(['user_id', 'ab_cohort'])
    .agg({'countbuy' : 'max'})
    .query('countbuy != 0')
    .reset_index()[['user_id', 'ab_cohort']]
)

# добавляем информацию о платящих пользователях в таблицу с профайлами

profiles = profiles.merge(
    payers,
    how='left',
    left_on=['user_id', 'ab_cohort'],
    right_on=['user_id', 'ab_cohort'],
    indicator='i'
    )

profiles['payer'] = np.where(profiles['i'] == 'left_only', False, True)

profiles = profiles.drop(columns = 'i')

if __name__ == '__main__':
    print(profiles.head()) # смотрим результат

# проверим, нет ли ошибок в данных

if __name__ == '__main__':
    print(
        'Количество записей, где количество покупок равно 0, а сумма покупок не равна нулю: {}'
        .format(len(profiles.query('countbuy == 0 & sumrevenue != 0')))
    )

    print(
        'Количество записей, где количество покупок не равно 0, а сумма покупок равна нулю: {}'
        .format(len(profiles.query('countbuy != 0 & sumrevenue == 0')))
    )

# создаем функцию для анализа прохождения уровней в обеих группах

def boxplot(x_col, data=levels):

# выводим "ящик с усами" для каждой группы

    plt.figure(figsize=(15, 7))

    sns.boxplot(
        data=data,
        x=data[x_col],
        y=data['ab_cohort'],
        palette='gist_rainbow')

    plt.title('Распределение {} по группам'.format(x_col))

    plt.show()
    print()

# выводим максимальные, минимальные, средние и медианные значения для каждой группы

    for group in sorted(list(data['ab_cohort'].unique())):
        print('Показатели {} в группе {}'.format(x_col, group))
        print()
        print(
            data.loc[data['ab_cohort'] == group, x_col]
            .agg(['mean', 'median', 'min', 'max'])
            )
        print()

# выводим количество нулевых значений для каждой группы

        print('Количество нулевых значений колонки {} группы {}: {}'.format(
            x_col, group,
            len(data.loc[(data['ab_cohort'] == group) & (data[x_col] == 0)])
        ))
        
        print('Процент нулевых значений колонки {} группы {}: {:.2%}'.format(
            x_col, group, (
                len(data.loc[(data['ab_cohort'] == group) & (data[x_col] == 0)])
                / len(data.loc[data['ab_cohort'] == group])
                )
            )
        )
        print()

        print('_'*50)
        print()

# выводим процентную разницу медианных значений групп

    print('Процентное изменение медианных значений в {}: {:.2%}'.format(
        x_col, (
            (data.query('ab_cohort == "B"')[x_col].median()
            - data.query('ab_cohort == "A"')[x_col].median()
            ) / data.query('ab_cohort == "A"')[x_col].median()
        )
        )
    ) 

    print('*'*50)
    print()

# применяем функцию для всех данных со стартами и победами

if __name__ == '__main__':
    for col in levels.loc[:,
                          (levels.columns != 'user_id')
                          & (levels.columns != 'ab_cohort')
                          & (levels.columns != 'heart_time')
                          & (levels.columns != 'notcleanstart_percent')
                          & (levels.columns != 'notcleanfinish_percent')
                          ].columns:
        boxplot(col)

# посмотрим, сколько требуется попыток для прохождения одного уровня
# в каждой группе;
# создадим для этого функцию

def start_finish(start_col, finish_col, data=levels):

    for group in list(data['ab_cohort'].unique()):

        df = data.query('ab_cohort == @group').copy()
        
        df['mean'] = df[start_col] / df[finish_col]

        print(
            'Медианное количество прохождений уровней до победы в группе {}: {}'
            .format(group, round(df['mean'].median(), 2))
        )

# применяем функцию ко всем стартам и победам
if __name__ == '__main__':
    start_finish('countallstart', 'countallfinish')

# посмотрим, сколько требуется попыток для прохождения одного уровня
# без преимуществ в каждой группе
if __name__ == '__main__':
    start_finish('countcleanstart', 'countcleanfinish')

# посмотрим, сколько требуется попыток для прохождения одного уровня
# без преимуществ в каждой группе
if __name__ == '__main__':
    start_finish('notcleanstart', 'notcleanfinish')

# определяем средний процент стартов и побед с преимуществами для каждой группы
# только для тех пользователей, которые начали проходить хотя бы один уровень 
if __name__ == '__main__':
    for group in list(levels['ab_cohort'].unique()):

        print(
            'Медианный процент стартов с преимуществами от общего числа стартов в группе {}: {}'
            .format(group, round(
                levels.query('countallstart != 0')
                ['notcleanstart_percent'].median(), 2
                ))
            )
        
        print(
            'Медианный процент побед с преимуществами от общего числа побед в группе {}: {}'
            .format(group, round(
                levels.query('countallstart != 0')
                ['notcleanfinish_percent'].median(), 2
                ))
        )
            
        print()

# посмотрим, коррелирует ли время восстановления жизни
# с показателями прохождения уровней
if __name__ == '__main__':
    print(levels.corr())

# создаем функцию для анализа получения и трат золота в обеих группах

def histplot(x_col, data=gold, hist=False, bins=10):

# выводим максимальные, минимальные, средние и медианные значения для каждой группы
# без нулевых значений

    for group in list(data['ab_cohort'].unique()):
        print(
            'Показатели {} без нулевых значений в группе {}'
            .format(x_col, group)
            )
        print()
        print(data.loc[
            (data['ab_cohort'] == group) & (data[x_col] != 0), x_col]
            .agg(['mean', 'median', 'min', 'max']))
        print()

# выводим количество нулевых значений для каждой группы

        print('Количество нулевых значений колонки {} группы {}: {}'.format(
            x_col, group,
            len(data.loc[(data['ab_cohort'] == group) & (data[x_col] == 0)])
        ))
        
        print('Процент нулевых значений колонки {} группы {}: {:.2%}'.format(
            x_col, group, (
                len(data.loc[(data['ab_cohort'] == group) & (data[x_col] == 0)])
                / len(data.loc[data['ab_cohort'] == group])
                )
            )
        )
        print()

        print('_'*50)
        print()

# выводим процентную разницу медианных значений групп без нулевых значений

    print(
        'Процентное изменение медианных значений без нулевых значений в {}: {:.2%}'
        .format(x_col, (
            data.loc[
                (data['ab_cohort'] == "B") & (data[x_col] != 0), x_col
                ].median()
            - data.loc[
                (data['ab_cohort'] == "A") & (data[x_col] != 0), x_col
                ].median()
            ) / data.loc[
                (data['ab_cohort'] == "A") & (data[x_col] != 0), x_col
                ].median()
                )
        )
    
    print()

    if hist == True:

    # выводим графики распределения для каждой группы

        plt.figure(figsize=(15, 5))

        ax1 = plt.subplot(1, 2, 1)
        sns.histplot(data=data.query('ab_cohort == "A"')[x_col], bins=bins,
                    color=sns.color_palette('gist_rainbow')[-4])
        plt.title('Распределение {} в группе А'.format(x_col))
        plt.xlabel('Сумма {}'.format(x_col))
        plt.ylabel('Количество пользователей')

        ax2 = plt.subplot(1, 2, 2, sharey=ax1)
        sns.histplot(data=data.query('ab_cohort == "B"')[x_col], bins=bins,
                    color=sns.color_palette('gist_rainbow')[-2])
        plt.title('Распределение {} в группе В'.format(x_col))
        plt.xlabel('Суммарное {}'.format(x_col))
        plt.ylabel('Количество пользователей')

# применяем функцию для получения золота за просмотр рекламы
if __name__ == '__main__':
    histplot('get_ads', hist=True)

# применяем функцию для получения золота за прохождение глав
if __name__ == '__main__':
    histplot('get_chapter', bins=8, hist=True)

# применяем функцию для получения золота из покупки
if __name__ == '__main__':
    histplot('get_buy')

# применяем функцию для получения золота за логин в Facebook
if __name__ == '__main__':
    histplot('get_faceb')

# применяем функцию для получения золота за отправку жизней в команде
if __name__ == '__main__':
    histplot('get_teaml', bins=8, hist=True)

# применяем функцию для получения золота за прохождение туториала для команды
if __name__ == '__main__':
    histplot('get_teamt')

# применяем функцию для трат золота на покупку жизней для бонусных глав
if __name__ == '__main__':
    histplot('spend_bonlives')

# применяем функцию для трат золота на покупку бонусов
if __name__ == '__main__':
    histplot('spend_bonus')

# применяем функцию для трат золота на покупку бустеров
if __name__ == '__main__':
    histplot('spend_boost')

# применяем функцию для трат золота на покупку жизней
if __name__ == '__main__':
    histplot('spend_lives')

# применяем функцию для трат золота на покупку ходов
if __name__ == '__main__':
    histplot('spend_moves')

# применяем функцию для трат золота на создание команды
if __name__ == '__main__':
    histplot('spend_teamc')

# посмотрим, коррелирует ли время восстановления жизни
# с показателями получения и трат золота

if __name__ == '__main__':
    print(gold.corr())

# создаем таблицу с удержанием платящих и неплатящих пользователей в каждой группе

retention = profiles.pivot_table(
    index=['payer', 'ab_cohort'],
    columns='retention',
    values='user_id',
    aggfunc='nunique')

# находим размер каждой когорты

cohort_size = profiles.pivot_table(
    index=['payer', 'ab_cohort'],
    values='user_id',
    aggfunc='nunique')

# рассчитываем удержание

retention_rate = retention.div(cohort_size['user_id'], axis=0)

# добавляем размеры когорт

retention_rate = cohort_size.join(retention_rate)

if __name__ == '__main__':
    print(retention_rate)

# строим график удержания платящих и неплатящих пользователей в обеих группах

if __name__ == '__main__':
    plt.figure(figsize=(15, 5))

    ax1 = plt.subplot(1, 2, 1)

    sns.lineplot(
        data=(
            retention_rate.drop(columns=['user_id', 0])
            .query('payer == True')
            .droplevel('payer').T
            ),
        palette='gist_rainbow'
        )

    plt.ylim([0, 1])
    plt.xlabel('Лайфтайм')
    plt.title('Удержание платящих пользователей')

    ax2 = plt.subplot(1, 2, 2, sharey=ax1)

    sns.lineplot(
        data=(
            retention_rate.drop(columns=['user_id', 0])
            .query('payer == False')
            .droplevel('payer').T),
            palette='gist_rainbow'
            )

    plt.xlabel('Лайфтайм')
    plt.title('Удержание неплатящих пользователей')

    plt.show()

# рассчитываем отскок

churn_rate = cohort_size.join(retention)
churn_rate = churn_rate.div(churn_rate.shift(periods=1, axis=1))
churn_rate = (1 - churn_rate) * 100
churn_rate['user_id'] = cohort_size

# строим тепловую карту
if __name__ == '__main__':
    plt.figure(figsize = (8, 5))

    sns.heatmap(
        churn_rate.drop(columns=['user_id', 0]),
        cbar_kws={'format': '%.0f%%'},
        annot=True,
        cmap="YlGnBu",
        linewidths=.5
        )

    plt.xlabel('Лайфтайм')
    plt.ylabel('Отношение к группе и покупкам')
    plt.title('Отскок пользователей в обеих группах')

    plt.show()

# определим общий процент отскока за 7 дней лайфтайма

common_churn_rate = cohort_size.join(retention[7])

common_churn_rate['percent'] = round(
    (common_churn_rate['user_id'] - common_churn_rate[7])
    / common_churn_rate['user_id'] * 100,
    2)

if __name__ == '__main__':
    print(common_churn_rate)

# находим день лайфтайма, когда была совершена первая покупка, для каждого пользователя;
# формируем таблицу конверсии из неплатящих пользователей в платящие

conversion = (
    (profiles.query('countbuy != 0')
    .sort_values(by=['user_id', 'ab_cohort', 'retention'])
    .groupby(['user_id', 'ab_cohort'])
    .agg({'retention' : 'first'})
    .reset_index())
    .pivot_table(
        index='ab_cohort',
        columns='retention',
        values='user_id',
        aggfunc='nunique'
    ).fillna(0)
)

# преобразовываем таблицу конверсии в таблицу с накоплением

conversion = np.cumsum(conversion, axis=1)

# определяем размер когорт по группам 

cohort_size_group = profiles.pivot_table(
    index='ab_cohort',
    values='user_id',
    aggfunc='nunique')

# находим процент конверсии

conversion = conversion.div(cohort_size_group['user_id'], axis=0)

# добавляем размер когорт

conversion = cohort_size_group.join(conversion)

if __name__ == '__main__':
    print(conversion)

# строим график конверсии
if __name__ == '__main__':
    plt.figure(figsize=(15, 5))

    sns.lineplot(data=(conversion.drop(columns='user_id').T), palette='gist_rainbow')

    plt.ylim([0, .1])
    plt.xlabel('Лайфтайм')
    plt.title('Конверсия из неплатящих в платящих пользователей')

    plt.show()

# находим суммарный доход на каждый день лайфтайма

ltv = profiles.pivot_table(
    index='ab_cohort',
    columns='retention',
    values='sumrevenue',
    aggfunc='sum'
    )

# преобразовываем таблицу ltv в таблицу с накоплением

ltv = np.cumsum(ltv, axis=1)

# находим средний доход от одного пользователя с накоплением

ltv = round(ltv.div(cohort_size_group['user_id'], axis=0), 2)

# добавляем размер когорт

ltv = cohort_size_group.join(ltv)

if __name__ == '__main__':
    print(ltv)

# строим график ltv
if __name__ == '__main__':
    plt.figure(figsize=(15, 5))

    sns.lineplot(data=(ltv.drop(columns='user_id').T), palette='gist_rainbow')

    plt.xlabel('Лайфтайм')
    plt.title('Накопленный доход от одного пользователя')

    plt.show()

# находим суммарное количество покупок на каждый день лайфтайма

count_buy = profiles.query('payer == True').pivot_table(
    index='ab_cohort',
    columns='retention',
    values='countbuy',
    aggfunc='sum'
    )

# находим размер когорт для платящих пользователей

cohort_size_payers = profiles.query('payer == True').pivot_table(
    index='ab_cohort',
    values='user_id',
    aggfunc='nunique')

# преобразовываем таблицу в таблицу с накоплением

count_buy = np.cumsum(count_buy, axis=1)

# находим средний среднее количество покупок одного пользователя с накоплением

count_buy = round(count_buy.div(cohort_size_payers['user_id'], axis=0), 2)

# добавляем размер когорт

count_buy = cohort_size_payers.join(count_buy)

if __name__ == '__main__':
    print(count_buy)

# строим график с накопленным количеством покупок на одного пользователя
if __name__ == '__main__':
    plt.figure(figsize=(15, 5))

    sns.lineplot(data=(count_buy.drop(columns='user_id').T), palette='gist_rainbow')

    plt.xlabel('Лайфтайм')
    plt.title('Накопленное количество покупок на одного платящего пользователя')

    plt.show()

# создаем функцию, которая будет проводить тест манна-уитни
# данный тест выбран как устойчивый к выбросам и малым выборкам

def test(col, data=data, alt='two-sided', alpha = 0.05):

    alpha = alpha

    U, p_value = mannwhitneyu(
        data.query('ab_cohort == "A"')[col],
        data.query('ab_cohort == "B"')[col],
        alternative=alt)

    print('Уровень статистической значимости: ', round(p_value, 5))
            
    if p_value < alpha:
      print('Отвергаем нулевую гипотезу: разница статистически значима')
    else:
      print('Не получилось отвергнуть нулевую гипотезу')

# будет совершено 3 теста, поэтому критическое значение уровня значимости будет
# установлено с поправкой Бонферрони

# проверяем, отличается ли количество пройденных уровней пользователями
# в группе А и в группе В

if __name__ == '__main__':
    test('maxlevelpassed', data=levels, alpha = 0.05/3)

# проверяем, отличается ли общее количество стартов уровней
# в группе А и в группе В
if __name__ == '__main__':
    test('countallstart', data=levels, alpha = 0.05/3)

# проверяем, отличается ли полученный от пользователей доход в группе А и в группе В
if __name__ == '__main__':
    test(
        'sumrevenue',
        data=(
            profiles
            .groupby(['user_id', 'ab_cohort'])
            .agg({'sumrevenue' : 'sum'})
            .reset_index()),
        alpha = 0.05/3
        )