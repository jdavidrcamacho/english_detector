# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import re
from tqdm import tqdm


def count_language_words(text):
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
    au_words = ['biscuit',    'ute',    'torch',    'chips',    'lift',    'queue',    
                'rubbish',    'postcode',    'flat',    'bonnet',    'boot',    'mobile',    'trousers',   
                'braces',    'brolly',    'chuffed',    'cheeky',    'knackered',    'loo',    'mum',    
                'plaster',    'pram',    'telly',    'trolley',    'windscreen',    'barbie',    'cop',    
                'brekkie',    'esky',    'maccas',    'servo',    'thongs',    'arvo',    'ute',    'keen',   
                'trackie dacks',    'pokies',    'espresso',    'snag',    'ta',    'bloke',    'dunny',    
                'lollies',    'footy',    'cuppa',    'chook',    'fair dinkum',    'fossick',    'heaps',   
                'mug',    'no worries',    'schooner',    'stubbie',    'tinnie',    'woop woop']

    sa_words = ['braai',    'robot',    'bakkie',    'chop-chop',    'dagga',    'doek',    
                'bokkie',    'kiff',    'larny',    'naartjie',    'ou',    'sarmie',    'howzit',    
                'ja',    'ubuntu',    'voetsek',    'veld',    'lekker',    'babelaas',    'boerewors',    
                'kombi',    'muti',    'pap',    'mealie',    'tsotsi',    'shebeen',    'tsonga',    
                'kasi',    'madiba',    'bliksem',    'snoek',    'tik',    'dagga',    'donga',    
                'eina',    'mielie',    'robot',    'kloof',    'koppie',    'bra',    'dop',    
                'jol',    'kief',    'skollie',    'sosatie',    'veldskoen',    'ouma',    'oupa']
    nz_words = ['bach',    'jandals',    'chilly bin',    'togs',    'biscuit',    'brekkie',    
                'dairy',    'haka',    'kiwi',    'puku',    'tiki',    'wop-wops',    'judder bar',    'chur',    
                'sweet as',    'welly',    'sheep station',    'suss',    'tramping',    'wahine',    'hongi',    
                'pakeha',    'whanau',    'hobbit',    'rugby',    'coromandel',    'hokey pokey',    'bach',    
                'kiwifruit',    'wharfie',    'pounamu',    'kiore',    'kia ora',    'taonga',    'tiki',    
                'jandal',    'puku',    'haka',    'wop-wops',    'pom',    'ute',    'kiwi',    'jafa',    
                'tiki tour',    'chocka',    'tramping',    'pakeha',    'whanau',    'wai',    'iwi',    
                'whare',    'mate',    'fush and chups']
    ca_words = ['toque',    'loonie',    'toonie',    'poutine',    'double-double',    'tuque',    
                'keener',    'chesterfield',    'runners',    'serviette',    'hydro',    'mickey',    'kerfuffle',    
                'loonie',    'backbacon',    'chesterfield',    'two-four',    'washroom',    'touque',    'all-dressed',    
                'hydro',    'serviette',    'loonie',    'gongshow',    'mickey',    'kerfuffle',    'runners',    
                'tuque',    'double-double',    'poutine',    'beaver tail',    'keener',    'washroom',    'two-four',    
                'backbacon',    'zed',    'runners',    'tuque',    'kerfuffle',    'toque',    'loonie',    'keener',    
                'hydro',    'mickey',    'homo milk',    'chesterfield',    'double-double',    'pogie',    'gonger',    
                'serviette',    'zing',    'kerfuffle',    'loonie',    'toonie',    'runners',    'tuque',    
                'backbacon',    'hydro',    'mickey',    'poutine',    'chesterfield',    'two-four',    'washroom',    
                'zed',    'keener',    'beaver tail',    'zing',    'double-double',    'homo milk',    'kerfuffle',    
                'loonie',    'gongshow',    'tuque',    'runners',    'backbacon',    'pogie',    'serviette',    'mickey',    
                'two-four',    'beaver tail',    'zing',    'washroom',    'double-double',    'chesterfield',    'toonie']

    word_count = {
        'UK English': 0,
        'American English': 0,
        'Australian English': 0,
        'South African English': 0,
        'New Zealand English': 0,
        'Canadian English': 0
    }

    # Split the text into words
    words = text.lower().split()

    # Count the occurrences of each type of word
    for word in words:
        if word in uk_words:
            word_count['UK English'] += 1
        if word in us_words:
            word_count['American English'] += 1
        if word in au_words:
            word_count['Australian English'] += 1
        if word in sa_words:
            word_count['South African English'] += 1
        if word in nz_words:
            word_count['New Zealand English'] += 1
        if word in ca_words:
            word_count['Canadian English'] += 1

    return word_count

