# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import re
from tqdm import tqdm

#from chatgpt
uk_words = ['biscuit', 'lorry', 'torch', 'chips', 'lift', 'queue', 'rubbish', 
            'postcode', 'flat', 'bonnet', 'boot', 'mobile', 'trousers', 'braces', 'brolly', 
            'chuffed', 'cheeky', 'knackered', 'loo', 'mum', 'plaster', 'pram', 'telly', 'trolley', 
            'windscreen', 'banger', 'bobby', 'cuppa', 'crisps', 'holiday', 'jam', 'lift', 
            'scone', 'tarmac', 'till', 'toilet', 'whinge', 'zebra crossing', 'zed', 'diversion', 
            'draught', 'dummy', 'chemist', 'dustbin', 'garden', 'nappy', 'pavement', 'petrol', 
            'pub', 'shop', 'shopkeeper', 'trainer']

us_words = ['cookie', 'truck', 'flashlight', 'fries', 'elevator', 'line', 'garbage', 
            'zip code', 'apartment', 'hood', 'cellphone', 'pants', 'suspenders', 'umbrella', 
            'delighted', 'sassy', 'exhausted', 'bathroom', 'mom', 'band-aid', 'stroller', 'TV', 
            'shopping cart', 'windshield', 'hot dog', 'police officer', 'cup of coffee', 
            'potato chips', 'vacation', 'jelly', 'elevator', 'biscuit', 'asphalt', 'cash register', 
            'restroom', 'whine', 'crosswalk', 'zee', 'detour', 'draft', 'pacifier', 'pharmacy',
            'garbage can', 'yard', 'diaper', 'sidewalk', 'gas', 'bar',  'store', 'storekeeper', 'sneaker']

aus_words = [    'biscuit',    'ute',    'torch',    'chips',    'lift',    'queue',    'rubbish',    'postcode',    'flat',    'bonnet',    'boot',    'mobile',    'trousers',    'braces',    'brolly',    'chuffed',    'cheeky',    'knackered',    'loo',    'mum',    'plaster',    'pram',    'telly',    'trolley',    'windscreen',    'barbie',    'cop',    'brekkie',    'esky',    'maccas',    'servo',    'thongs',    'arvo',    'ute',    'keen',    'trackie dacks',    'pokies',    'espresso',    'snag',    'ta',    'bloke',    'dunny',    'lollies',    'footy',    'cuppa',    'chook',    'fair dinkum',    'fossick',    'heaps',    'mug',    'no worries',    'schooner',    'stubbie',    'tinnie',    'woop woop']

sa_words = [    'braai',    'robot',    'bakkie',    'chop-chop',    'dagga',    'doek',    'bokkie',    'kiff',    'larny',    'naartjie',    'ou',    'sarmie',    'howzit',    'ja',    'ubuntu',    'voetsek',    'veld',    'lekker',    'babelaas',    'boerewors',    'kombi',    'muti',    'pap',    'mealie',    'tsotsi',    'shebeen',    'tsonga',    'kasi',    'madiba',    'bliksem',    'snoek',    'tik',    'dagga',    'donga',    'eina',    'mielie',    'robot',    'kloof',    'koppie',    'bra',    'dop',    'jol',    'kief',    'skollie',    'sosatie',    'veldskoen',    'ouma',    'oupa']

nz_words = [    'bach',    'jandals',    'chilly bin',    'togs',    'biscuit',    'brekkie',    'dairy',    'haka',    'kiwi',    'puku',    'tiki',    'wop-wops',    'judder bar',    'chur',    'sweet as',    'welly',    'sheep station',    'suss',    'tramping',    'wahine',    'hongi',    'pakeha',    'whanau',    'hobbit',    'rugby',    'coromandel',    'hokey pokey',    'bach',    'kiwifruit',    'wharfie',    'pounamu',    'kiore',    'kia ora',    'taonga',    'tiki',    'jandal',    'puku',    'haka',    'wop-wops',    'pom',    'ute',    'kiwi',    'jafa',    'tiki tour',    'chocka',    'tramping',    'pakeha',    'whanau',    'wai',    'iwi',    'whare',    'mate',    'fush and chups']

canadian_words = [    'toque',    'loonie',    'toonie',    'poutine',    'double-double',    'tuque',    'keener',    'chesterfield',    'runners',    'serviette',    'hydro',    'mickey',    'kerfuffle',    'loonie',    'backbacon',    'chesterfield',    'two-four',    'washroom',    'touque',    'all-dressed',    'hydro',    'serviette',    'loonie',    'gongshow',    'mickey',    'kerfuffle',    'runners',    'tuque',    'double-double',    'poutine',    'beaver tail',    'keener',    'washroom',    'two-four',    'backbacon',    'zed',    'runners',    'tuque',    'kerfuffle',    'toque',    'loonie',    'keener',    'hydro',    'mickey',    'homo milk',    'chesterfield',    'double-double',    'pogie',    'gonger',    'serviette',    'zing',    'kerfuffle',    'loonie',    'toonie',    'runners',    'tuque',    'backbacon',    'hydro',    'mickey',    'poutine',    'chesterfield',    'two-four',    'washroom',    'zed',    'keener',    'beaver tail',    'zing',    'double-double',    'homo milk',    'kerfuffle',    'loonie',    'gongshow',    'tuque',    'runners',    'backbacon',    'pogie',    'serviette',    'mickey',    'two-four',    'beaver tail',    'zing',    'washroom',    'double-double',    'chesterfield',    'toonie']



def language_predictions_rules(df, column='messge'):
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
