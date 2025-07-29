'''
Recommendation Tasks 
'''

all_tasks = {}

# retrieval without candidates
# =====================================================
# Task Subgroup 1 -- Retrieval
# =====================================================

template = {}

template['source'] = """You are a movie recommender. If a user has watched the following movies, listed chronologically from earliest to latest, and rated them positively, :\n{},\nthen you should recommend the following movies: {}. Now that the user has watched the recommended movies, you should recommend 10 other movies from the year between 1919 and 2000 (inclusive) that the user is most likely to watch next. You should order them by probability and compact them in one line split by commas. Do not output the probability. Do not re-recommend movies that have been watched by the user. Do not output other words.""" 
template['source_no_val'] = """You are a movie recommender. A user has watched the following movies, listed chronologically from earliest to latest, and rated them positively, :\n{}.\nYou should recommend 10 other movies from the year between 1919 and 2000 (inclusive) that the user is most likely to watch next. You should order them by probability and compact them in one line split by commas. Do not output the probability. Do not re-recommend movies that have been watched by the user. Do not output other words.""" 
template['target'] = "{}"
template['task'] = "retrieval"
template['id'] = "1-1"

all_tasks["retrieval"] = template