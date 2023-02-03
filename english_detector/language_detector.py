# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import re
from tqdm import tqdm


path = r'~\OneDrive - Brand Delta\General - BAT\data\modelled_data\english\Workflow_output\latest_output'
df = pd.read_parquet(path + '\english_data_tagged_2023_01_13_04_20_20_PM_age18.parquet')

def language_predictions_rules(df, column='cleaned_messge'):
    dataframe = df.copy().reset_index(drop=True)
    dataframe['language'] = 'Unknown'

    EN_UK = ['aerial', 'angry', 'anywhere', 'autumn', 'bank note', 'barrister',
             'solicitor', 'biscuit', 'bonnet', 'boot', 'braces', 'caretaker',
             "chemist's", 'chips', 'the cinema', 'condom', 'constable', 'cooker',
             'corn', 'wheat', 'cot', 'cotton', 'crash', 'crossroads', 'curtains',
             'draughts', 'drawing pin', 'dual carriageway', 'dummy', 'dustbin',
             'rubbish-bin', 'dustman', 'dynamo', 'engine', 'engine driver', 'film',
             'flat', 'flyover', 'garden', 'gear-lever', 'gear lever', 'graduate',
             'grill', 'ground floor', 'gumshoes', 'wellington boots', 'gym shoes',
             'tennis-shoes', 'tennis shoes', 'handbag', 'hoarding',  'holiday',
             'hoover', 'ill', 'interval', 'jersey', 'jumper', 'pullover', 'sweater',
             'jug', 'lift', 'lorry', 'luggage', 'mackintosh', 'raincoat', 'mad',
             'main road', 'maize', 'maths', 'mean', 'motorway', 'nappy', 'nasty',
             'nowhere', 'nursing home', 'optician', 'off-license', 'off license',
             'paraffin', 'pavement', 'peep', 'petrol', 'post', 'postbox', 'postman',
             'potato crisps', 'crisps', 'pram', 'pub', 'public toilet', 'puncture',
             'push-chair', 'push chair',  'queue', 'railway', 'railway carriage',
             'reel of cotton', 'return ticket', 'reverse charges', 'rise in salary',
             'road surface', 'roundabout', 'rubber', 'rubbish', 'saloon', 'sellotape',
             'shop', 'silencer', 'single ticket', 'somewhere', 'spanner',
             'staff', 'sump', 'sweet', 'sweets', 'tap', 'tap outdoors', 'taxi',
             'tea-towel', 'tea towel', 'term', 'tights', 'timetable', 'tin',
             'toll motorway', 'torch', 'tramp', 'trousers', 'turn-ups', 'turn ups',
             'underground railway', 'underpants','verge of road', 'waistcoat',
             'wardrobe', 'wash your hands', 'windscreen', 'wing', 'zip',
             'university', 'fizzy drink', 'football']
    EN_UK = re.compile("|".join(EN_UK))

    EN_US = ['antenna', 'mad', 'anyplace', 'fall', 'bill', 'attorney', 'cookie',
             'hood', 'trunk', 'suspenders', 'janitor', 'drug store', 'french fries',
             'the movies', 'rubber', 'patrolman', 'stove', 'wheat', 'crib', 'thread',
             'wreck', 'intersection', 'drapes', 'checkers', 'thumbtack', 'divided highway',
             'pacifier', 'trashcan', 'garbage can', 'garbage collector', 'generator',
             'motor', 'engineer', 'movie', 'apartment', 'overpass', 'yard', 'gear-shift',
             'gear shift', 'alumnus', 'boiler', 'first floor', 'rubbers', 'sneakers',
             'purse', 'billboard', 'vacation', 'vacuum cleaner', 'sick', 'intermission',
             'sweater', 'pitcher', 'elevator', 'truck', 'baggage', 'raincoat',
             'crazy', 'highway', 'corn', 'math', 'stingy', 'freeway', 'diaper',
             'vicious, mean', 'noplace', 'private hospital', 'optometrist',
             'liquor store', 'kerosene', 'sidewalk', 'peek', 'gasoline',
             'mail', 'mailbox', 'mailman', 'mail carrier', 'potato chips', 'chips',
             'baby carriage', 'bar', 'restroom', 'blow-out', 'blow out', 'stroller',
             'line', 'railroad', 'railway car', 'spool of thread', 'round trip',
             'call collect', 'raise', 'pavement', 'traffic circle', 'eraser',
             'garbage', 'trash', 'sedan', 'scotch tape', 'store', 'muffler',
             'one-way', 'one way', 'someplace', 'wrench', 'faculty', 'oil pan',
             'dessert', 'candy', 'faucet', 'spigot', 'cab', 'dish-towel', 'dish towel',
             'semester', 'pantyhose', 'schedule', 'can', 'turnpike', 'flashlight',
             'hobo', 'pants', 'cuffs', 'subway', 'shorts', 'shoulder of road',
             'vest', 'closet', 'wash up', 'windshield', 'fender', 'zipper',
             'college', 'soda', 'drugstore', 'soccer']
    EN_US = re.compile("|".join(EN_US))

    for index in tqdm(range(0, len(dataframe))):
        conv_stream = str(dataframe.loc[index, column]).lower()
        keywords_EN_UK = len(re.findall(EN_UK, conv_stream))
        keywords_EN_US = len(re.findall(EN_US, conv_stream))
        if keywords_EN_UK > keywords_EN_US:
            dataframe.loc[index, 'language'] = 'EN_UK'
        if keywords_EN_US > keywords_EN_UK:
            dataframe.loc[index, 'language'] = 'EN_US'
    res = dataframe['language']
    return res

def language_predictions_rules_2(df, column='cleaned_messge'):
    dataframe = df.copy().reset_index(drop=True)
    dataframe['language'] = 'Unknown'

    EN_UK = ['aerial', 'angry', 'anywhere', 'autumn', 'bank note', 'barrister',
             'solicitor', 'biscuit', 'bonnet', 'boot', 'braces', 'caretaker',
             "chemist's", 'chips', 'the cinema', 'condom', 'constable', 'cooker',
             'corn', 'wheat', 'cot', 'cotton', 'crash', 'crossroads', 'curtains',
             'draughts', 'drawing pin', 'dual carriageway', 'dummy', 'dustbin',
             'rubbish-bin', 'dustman', 'dynamo', 'engine', 'engine driver', 'film',
             'flat', 'flyover', 'garden', 'gear-lever', 'gear lever', 'graduate',
             'grill', 'ground floor', 'gumshoes', 'wellington boots', 'gym shoes',
             'tennis-shoes', 'tennis shoes', 'handbag', 'hoarding',  'holiday',
             'hoover', 'ill', 'interval', 'jersey', 'jumper', 'pullover', 'sweater',
             'jug', 'lift', 'lorry', 'luggage', 'mackintosh', 'raincoat', 'mad',
             'main road', 'maize', 'maths', 'mean', 'motorway', 'nappy', 'nasty',
             'nowhere', 'nursing home', 'optician', 'off-license', 'off license',
             'paraffin', 'pavement', 'peep', 'petrol', 'post', 'postbox', 'postman',
             'potato crisps', 'crisps', 'pram', 'pub', 'public toilet', 'puncture',
             'push-chair', 'push chair',  'queue', 'railway', 'railway carriage',
             'reel of cotton', 'return ticket', 'reverse charges', 'rise in salary',
             'road surface', 'roundabout', 'rubber', 'rubbish', 'saloon', 'sellotape',
             'shop', 'silencer', 'single ticket', 'somewhere', 'spanner',
             'staff', 'sump', 'sweet', 'sweets', 'tap', 'tap outdoors', 'taxi',
             'tea-towel', 'tea towel', 'term', 'tights', 'timetable', 'tin',
             'toll motorway', 'torch', 'tramp', 'trousers', 'turn-ups', 'turn ups',
             'underground railway', 'underpants','verge of road', 'waistcoat',
             'wardrobe', 'wash your hands', 'windscreen', 'wing', 'zip',
             'university', 'fizzy drink', 'football']
    EN_UK = re.compile("|".join(EN_UK))

    EN_US = ['antenna', 'mad', 'anyplace', 'fall', 'bill', 'attorney', 'cookie',
             'hood', 'trunk', 'suspenders', 'janitor', 'drug store', 'french fries',
             'the movies', 'rubber', 'patrolman', 'stove', 'wheat', 'crib', 'thread',
             'wreck', 'intersection', 'drapes', 'checkers', 'thumbtack', 'divided highway',
             'pacifier', 'trashcan', 'garbage can', 'garbage collector', 'generator',
             'motor', 'engineer', 'movie', 'apartment', 'overpass', 'yard', 'gear-shift',
             'gear shift', 'alumnus', 'boiler', 'first floor', 'rubbers', 'sneakers',
             'purse', 'billboard', 'vacation', 'vacuum cleaner', 'sick', 'intermission',
             'sweater', 'pitcher', 'elevator', 'truck', 'baggage', 'raincoat',
             'crazy', 'highway', 'corn', 'math', 'stingy', 'freeway', 'diaper',
             'vicious, mean', 'noplace', 'private hospital', 'optometrist',
             'liquor store', 'kerosene', 'sidewalk', 'peek', 'gasoline',
             'mail', 'mailbox', 'mailman', 'mail carrier', 'potato chips', 'chips',
             'baby carriage', 'bar', 'restroom', 'blow-out', 'blow out', 'stroller',
             'line', 'railroad', 'railway car', 'spool of thread', 'round trip',
             'call collect', 'raise', 'pavement', 'traffic circle', 'eraser',
             'garbage', 'trash', 'sedan', 'scotch tape', 'store', 'muffler',
             'one-way', 'one way', 'someplace', 'wrench', 'faculty', 'oil pan',
             'dessert', 'candy', 'faucet', 'spigot', 'cab', 'dish-towel', 'dish towel',
             'semester', 'pantyhose', 'schedule', 'can', 'turnpike', 'flashlight',
             'hobo', 'pants', 'cuffs', 'subway', 'shorts', 'shoulder of road',
             'vest', 'closet', 'wash up', 'windshield', 'fender', 'zipper',
             'college', 'soda', 'drugstore', 'soccer']
    EN_US = re.compile("|".join(EN_US))

    EN_AU = ['a good lurk', 'reckon', 'prang', 'arvo', 'aggro', 'grog', 'booze',
             'yank', 'berko', 'aussie', 'strine', 'nana', 'amber', 'middy', 'pot',
             'bickie', 'damper', 'duco', 'plonk', 'bushranger', 'chook',  'chokkie',
             'chrissie', 'wharfie', 'pissed', 'donk', 'pom', 'gum tree', 'tea',
             'good oil', 'ace', 'back of beyond', 'station', 'ringer', 'paddock',
             'thongs', 'tucker', 'brave', 'shove off', "g'day mate", 'gday mate',
             'gday', 'mate', 'neddies', 'dill', 'drongo', 'oil', "she's apple",
             'roo', 'loo', 'dunny', 'bottle shop', 'back of bourke', 'deli',
             'milko', 'lolly', 'mozzie', 'kiwi', 'enzedder', 'piffle', 'earbush',
             'postie', 'counter meal',  'never-never', 'never never', 'beef road',
             'cut lunch', 'snag', 'flake', 'jumbuck', 'gummy', 'kelpie', 'woolgrower',
             'belt up', 'digger', 'gibber', 'alf', 'sunnies', 'bathers', 'num-nums',
             'num nums', 'billie', 'chalkie','grizzle', 'gander', 'fossick',
             'comfort station', 'dead horse', 'jackaroo', 'daks', 'strides', 'jocks',
             'mortician', 'vegemite', 'vegies', 'billabong', 'good on ya']
    EN_AU = re.compile("|".join(EN_AU))

    # for index in tqdm(range(0, len(dataframe))):
    #     conv_stream = str(dataframe.loc[index, column]).lower()
    #     keywords_EN_UK = len(re.findall(EN_UK, conv_stream))
    #     keywords_EN_US = len(re.findall(EN_US, conv_stream))
    #     keywords_EN_AU = len(re.findall(EN_AU, conv_stream))
    #     if keywords_EN_UK > (keywords_EN_US and keywords_EN_AU):
    #         dataframe.loc[index, 'language'] = 'EN_UK'
    #     if keywords_EN_US > (keywords_EN_UK and keywords_EN_AU):
    #         dataframe.loc[index, 'language'] = 'EN_US'
    #     if keywords_EN_AU > (keywords_EN_UK and keywords_EN_US):
    #         dataframe.loc[index, 'language'] = 'EN_AU'

    for index in tqdm(range(0, len(dataframe))):
        conv_stream = str(dataframe.loc[index, column]).lower()
        keywords_dict = {'EN_UK': len(re.findall(EN_UK, conv_stream)),
                         'EN_US': len(re.findall(EN_US, conv_stream)),
                         'EN_AU': len(re.findall(EN_AU, conv_stream))}
        max_value = max(keywords_dict.values())
        max_value = [key for key, value in keywords_dict.items() if value == max_value]
        if len(max_value) == 1:
            dataframe.loc[index, 'language'] = max_value[0]

    res = dataframe['language']
    return res

results1 = language_predictions_rules(df, column='conversation_stream')
results2 = language_predictions_rules_2(df, column='conversation_stream')
