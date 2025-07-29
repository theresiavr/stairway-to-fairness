import pandas as pd
import numpy as np

from collections import OrderedDict

from scipy.spatial.distance import pdist
from scipy.stats import entropy

from typing import Optional
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

map_columns = {
                #datasets
                "ml-1m":"ML-1M",
               "jobrec":"JobRec",
               "lfm-1b":"LFM-1B",

                #models
               "UserKNN":"UKNN",
               "SUKNN":"S-UKNN",
               "NeuMF":"NMF",
               "SNeuMF":"S-NMF",
               }

def measure_type_multiple_group(name):
    count_ = name.count("-")

    if "Within" in name:
        count_ -= 1

    if "Ind" in name:
        return "Ind"

    elif count_ == 3:
        return "Grp (2)"
    
    elif count_ == 2:
        return "Grp (1)"
    
    else:
        return "Grp (3)"
    
def rotate_index(indices):
    indices = "\\rotatebox[origin=c]{90}{" + indices + "}"
    return indices
    
def fill_attr(group_table):

    group_table.loc[(group_table.data=='ml-1m') & (group_table.Fair =='Grp (3)'), "Attr."] =  "Gender-Age-Occupation"
    group_table.loc[(group_table.data=='jobrec') & (group_table.Fair =='Grp (3)'), "Attr."] =  "Degree-Experience-Major"
    group_table.loc[(group_table.data=='lfm-1b') & (group_table.Fair =='Grp (3)'), "Attr."] = "Gender-Age-Continent"

    group_table.index.names = [None]
    group_table.columns.names = [None]

    group_table["data"] = group_table["data"].apply(rotate_index) 
    group_table["Attr."] = group_table["Attr."].apply(lambda x: "-" if len(x) ==0 else x)
    return group_table

def print_group_table(group_table):

    printed_group_table = group_table.set_index(["data", "Fair", "Attr."])
    printed_group_table = printed_group_table\
                                    .loc[:,["$\downarrow$ SD", "$\downarrow$ Gini",  "$\downarrow$ Atk",]]

    styler_group = printed_group_table.round(3).style

    styler_group.format(formatter="{:.3f}")

    latex_code = styler_group.to_latex(
            hrules=True, 
            clines="skip-last;data",
    )

    last_cline_starts = latex_code.find("\\cline", -75,-1)
    last_cline_ends = latex_code.find("\\bottomrule")
    latex_code = latex_code[:last_cline_starts] + latex_code[last_cline_ends:]

    for k, v in map_columns.items():
        latex_code = latex_code.replace(k, v)

    print(latex_code)
    return printed_group_table



def prep_df_for_lineplot(printed_group_table):
    group_plot = printed_group_table.reset_index()
    group_plot["data"] = group_plot["data"].str.extract("\}\{(.*)\}")
    group_plot["data"] = group_plot["data"].map(map_columns)

    group_plot = group_plot.melt(id_vars=["data","Fair", "Attr."],value_vars=["$\downarrow$ SD",	"$\downarrow$ Gini","$\downarrow$ Atk"])
    group_plot.rename(columns={"variable":"measure", "value":"$\downarrow$Unfairness", "Fair":"Fairness Type"}, inplace=True)

    group_plot["measure"] = group_plot["measure"].str.replace(" ","")
    group_plot["measure"] = group_plot["measure"].str.replace("$\\downarrow$","")
    return group_plot

def plot_line(group_plot, exp_type="", save=False):
    lineplot = sns.relplot(group_plot,kind="line", x="#groups", y="$\downarrow$Unfairness", hue="measure", style="measure", col="data", ci=None, 
                           aspect=0.9, height=2.4, palette="colorblind",
                           facet_kws=dict(sharey=False, sharex=False)
                           )
    
    lineplot.set_titles(col_template = '{col_name}')
    sns.move_legend(lineplot, loc="upper center", ncols=3, bbox_to_anchor=(.5, 1.1), title=None)
    plt.tight_layout()

    if save:
        time = Utils().timenow()
        plt.savefig(f'multiple_groups/temp_{time}_different_ways_of_grouping_{exp_type}.pdf', bbox_inches='tight')


def compute_group_scores(results, data, model, df_user, selected_cols, sensitive_cols, base_score, agg_type="", fairness=None):
    per_group_score = df_user[selected_cols]\
                            .groupby(sensitive_cols)\
                            .mean()[base_score]\
                            .dropna()
    
    per_group_count = df_user[selected_cols]\
                            .groupby(sensitive_cols)\
                            .count()[base_score]
    per_group_count = per_group_count[per_group_count>0]
    
    results[data][model][f"Min-{agg_type}{base_score}"] = fairness.score_worst(per_group_score)
    results[data][model][f"Range-{agg_type}{base_score}"] = fairness.score_range(per_group_score)
    results[data][model][f"SD-{agg_type}{base_score}"] = fairness.score_std(per_group_score)
    results[data][model][f"CV-{agg_type}{base_score}"] = fairness.score_cov(per_group_score)

    results[data][model][f"MAD-{agg_type}{base_score}"] = fairness.MAD(per_group_score)

    results[data][model][f"Gini-{agg_type}{base_score}"] = fairness.gini(per_group_score)
    results[data][model][f"FStat-{agg_type}{base_score}"] = fairness.fstat(df_user, base_score, per_group_score, per_group_count)
    results[data][model][f"KL-{agg_type}{base_score}"] = fairness.KL(per_group_score, per_group_count)
    results[data][model][f"GCE-{agg_type}{base_score}"] = fairness.GCE(per_group_score)

    #=== Atkinson ===
    # within

    list_ede = []
    atk_within = []
    gini_within = []
    sd_within = []

    tot_base_score = df_user[base_score].sum()

    for group in per_group_score.index:

        user_score_in_group = df_user.set_index(per_group_score.index.names).loc[group][base_score]
        list_ede.append(fairness.ede(user_score_in_group.values))

        share = user_score_in_group.sum()/tot_base_score

        atk_within.append(share * fairness.atk(user_score_in_group))
        gini_within.append(share * fairness.gini(user_score_in_group))
        sd_within.append(share * fairness.score_std(user_score_in_group.values))

    atk_within = list(filter(lambda x: not np.isnan(x), atk_within))
    gini_within = list(filter(lambda x: not np.isnan(x), gini_within))
    sd_within = list(filter(lambda x: not np.isnan(x), sd_within))
    
    atk_within = sum(atk_within)
    gini_within = sum(gini_within)
    sd_within = sum(sd_within)

    results[data][model][f"Atk-Within-{agg_type}{base_score}"] = atk_within
    results[data][model][f"Gini-Within-{agg_type}{base_score}"] = gini_within
    results[data][model][f"SD-Within-{agg_type}{base_score}"] = sd_within

    # between
    atk_b = fairness.atk(np.asarray(list_ede), weights=per_group_count/per_group_count.sum())
    results[data][model][f"Atk-{agg_type}{base_score}"] = atk_b

    return per_group_score, per_group_count, atk_b, atk_within


class Utils():

    def clean_col(self,col):
        return col.replace("map_","").title()

    def load_df_user(self,data):
        df_user = pd.read_pickle(f"../cleaned_data/{data}_user.pkl")
        #filter user_id to be only for the ones in test 
        df_user = df_user[df_user.in_test]

        #filter only the necessary fields
        if data == "ml-1m":
            df_user = df_user[["user_id","gender","map_age", "map_occupation"]]

        elif "lfm" in data:
            df_user = df_user[["user_id","gender","map_age","map_continent"]]

        elif "jobrec" in data:
            df_user = df_user[["user_id","map_degree","map_experience","map_major"]]

        return df_user
    
    def flatten_dict(self, nested_dict):
        res = OrderedDict()
        if isinstance(nested_dict, dict):
            for k in nested_dict:
                flattened_dict = self.flatten_dict(nested_dict[k])
                for key, val in flattened_dict.items():
                    key = list(key)
                    key.insert(-1, k)
                    res[tuple(key)] = val
        else:
            res[()] = nested_dict
        return res
    


    def timenow(self):
        now = datetime.now()
        time = str(now.strftime("%Y-%m-%d_%H%M%S"))
        return time

class Effectiveness():

    def hit(self, top_k_rel):
        return top_k_rel.any(1).long()
    
    def mrr(self, top_k_rel):
        res = np.zeros(top_k_rel.shape[0])

        for idx_u, u in enumerate(top_k_rel):
            if not u.any():
                continue
            for idx_i, item in enumerate(u):
                if item:
                    res[idx_u] = 1/(idx_i+1)
                break
        return res


    def prec(self, top_k_rel, k):
        return top_k_rel.sum(1) / k
    

    def ndcg(self, top_k_rel, num_rel, k):
        # from recbole
        len_rank = np.full_like(num_rel, top_k_rel.shape[1])
        idcg_len = np.where(num_rel > len_rank, len_rank, num_rel)

        iranks = np.zeros_like(top_k_rel, dtype=float)
        iranks[:, :] = np.arange(1, top_k_rel.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        try:
            for row, idx in enumerate(idcg_len):
                idcg[row, idx:] = idcg[row, idx - 1]
        except:
            for row, idx in enumerate(idcg_len):
                idcg[row, idx[0]:] = idcg[row, idx - 1]
        ranks = np.zeros_like(top_k_rel, dtype=float)
        ranks[:, :] = np.arange(1, top_k_rel.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(top_k_rel, dcg, 0), axis=1)

        result = dcg / idcg
        result_at_k = result[:,k - 1]

        return result_at_k
    
class Fairness():
    def score_range(self, scores: pd.Series) -> float:
        #lower is better
        return scores.max() - scores.min()

    def score_std(self, scores:pd.Series) -> float:
        #lower is better
        return scores.std(ddof=0)

    def score_cov(self, scores:pd.Series) -> float:
        return scores.std(ddof=0) / scores.mean()

    def score_worst(self, scores:pd.Series, worst_pct=0.25) -> float:
        #higher is better
        thresh = scores.quantile(worst_pct)
        worst_scores = scores[scores<=thresh]
        avg_worst_scores = worst_scores.mean()

        return avg_worst_scores

    def gini(self, scores:pd.Series) -> float:
        #lower is better
        sorted_values_at_k = scores.sort_values()
        num_instance = scores.shape[0]

        idx = np.arange(1, num_instance + 1)
        gini_index = np.sum((2 * idx - num_instance - 1) * sorted_values_at_k) / num_instance
        gini_index /= scores.sum()
        return gini_index


    def fstat(self, df_user, metric, per_group_score, per_group_count) -> float:
        # Ratio of Between-Groups to Within-Groups Variances
        # lower is better
        # small values = the differences in group means are comparable to the variation within the groups
        # higher fstat = higher disparity of between- and within-group 

        num_user = df_user.shape[0]
        mean_score = df_user[metric].mean()

        #=== BETWEEN GROUP === 
        term_inside_sum = per_group_count * (per_group_score - mean_score)**2
        v_market = term_inside_sum.sum() / num_user 

        #=== WITHIN GROUP ===
        u_market = 0

        for group in per_group_score.index:
            mean_score_this_group = per_group_score.loc[group]
            
            user_score_in_group = df_user.set_index(per_group_score.index.names).loc[group][metric]

            squared_diff = (user_score_in_group - mean_score_this_group)**2
            sum_squared_diff = squared_diff.sum()
            u_market += sum_squared_diff

        u_market /= num_user
        num_intersectional_groups = per_group_score.shape[0]
        fstat = (v_market/ (num_intersectional_groups - 1)) / (u_market / (num_user - num_intersectional_groups))
        
        return fstat 
    
    def GCE(self, per_group_score, B=2):
        """"
        "this is the non-negative version of the measure in Deldjoo, Y., Anelli, V.W., Zamani, H., Bellogín, A., & Noia, T.D. (2021). "
        "A flexible framework for evaluating user and item fairness in recommender systems. "
        "User Modeling and User-Adapted Interaction, 31, 457 - 511."
        """
        # version: parity
        num_intersectional_groups = per_group_score.shape[0]
        pf = np.asarray([1/num_intersectional_groups for i in range(num_intersectional_groups)])
        pm = per_group_score.values / per_group_score.sum()

        # === smoothing with Jeliner-Mercer method ===
        lmbda = 0.95
        pc = 0.0001 #different than the original

        pm = lmbda * pm + (1-lmbda) * pc
        Z = pm.sum()

        pm = pm/Z

        score = (pf**B) * pm**(1-B)

        score = score.sum() -1
        score = score / (B*(1-B))

        return -score



    def ede(self, scores, eps=0.5, weights: Optional[np.array] = None):
            
        if eps > 0 and eps !=1:
            if weights is not None:
                res = weights * (scores**(1-eps))
                res = res.sum() / weights.sum()
                if eps > 1:
                    res =  1 / res ** (1/(eps-1))
                else:
                    res = res ** (1/(1-eps))
            
            else:
                if eps > 1:
                    res = 1/(scores**(eps-1))
                    res  = res.mean()
                    res = 1 / res ** (1/(eps-1))
                else:
                    res = scores**(1-eps)
                    res = res.mean()
                    res = res ** (1/(1-eps))
            return res

        elif eps ==1:
            if not weights:
                return np.prod(scores)**(1/scores.shape[0])
            else:
                return "not implemented"
            
        else:
            return "error!"
        
    def atk(self, scores, eps=0.5, weights=None):
        return 1 - self.ede(scores, eps,weights=weights)/np.average(scores, weights=weights)


    def MAD(self, per_group_score):
        return pdist(per_group_score.values.reshape(-1,1), lambda u, v: abs(u-v)).mean()
    

    def KL(self, per_group_score, per_group_count) -> float:
        # version: proportional to group size, base 2 from https://github.com/abellogin/FairnessFramework4RecSys/blob/main/results/synthetic_experiment/simulation_and_eval.pl
        p = per_group_score / per_group_score.sum() 
        q = per_group_count / per_group_count.sum()
        return entropy(p, q, base=2)
    
