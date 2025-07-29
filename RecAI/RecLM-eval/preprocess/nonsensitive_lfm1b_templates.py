'''
Recommendation Tasks 
'''

all_tasks = {}

# retrieval without candidates
# =====================================================
# Task Subgroup 1 -- Retrieval
# =====================================================

template = {}

template['source'] = """You are a music artist recommender. If a user has listened to the following artists, listed chronologically from earliest to latest:\n{},\nand considering that the user listened to the artist {} times respectively, then you should recommend the following artists: {}. Now that the user has listened to the recommended artists, you should recommend 10 other artists, who has released songs prior to 2017, that the user is most likely to listen to next. You should order them by probability and compact them in one line split by commas. Do not output the probability. Do not re-recommend artists that the user has listened to. Do not output other words.""" 
template['source_no_val'] = """You are a music artist recommender. If a user has listened to the following artists, listed chronologically from earliest to latest:\n{}\nThe user listened to the artist {} times respectively. You should recommend 10 other artists, who has released songs prior to 2017, that the user is most likely to listen to next. You should order them by probability and compact them in one line split by commas. Do not output the probability. Do not re-recommend artists that the user has listened to. Do not output other words.""" 
template['target'] = "{}"
template['task'] = "retrieval"
template['id'] = "1-1"

all_tasks["retrieval"] = template