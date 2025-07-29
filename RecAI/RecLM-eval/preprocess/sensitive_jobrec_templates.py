'''
Recommendation Tasks 
'''

all_tasks = {}

# retrieval without candidates
# =====================================================
# Task Subgroup 1 -- Retrieval
# =====================================================

template = {}

template['source'] = """You are a job title recommender. If a user with a {} degree{} has a total of {} years of experience has applied to job positions with the following job titles, listed chronologically from earliest to latest:\n{},\nthen you should recommend the following job titles: {}. Now that the user has applied to positions with those job titles, you should recommend 10 other job titles that the user is most likely to apply for next. You should order them by probability and compact them in one line split by commas. Do not output the probability. Do not output other words.""" 
template['source_no_val'] = """You are a job title recommender. A user with a {} degree{} has a total of {} years of experience has applied to job positions with the following job titles, listed chronologically from earliest to latest:\n{}.\nYou should recommend 10 other job titles that the user is most likely to apply for next. You should order them by probability and compact them in one line split by commas. Do not output the probability. Do not output other words.""" 
template['target'] = "{}"
template['task'] = "retrieval"
template['id'] = "1-1"

all_tasks["retrieval"] = template