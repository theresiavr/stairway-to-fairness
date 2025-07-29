import glob
import pandas as pd

def clean(df, f):

    if "llama" in f:
        df["answer"] = df["answer"]\
                                    .str.split("\n\n").apply(lambda x: x[-1])

    df["answer"] =  df["answer"]\
                                .str.replace(",)",")")\
                                .str.replace("\,\(", ", (", regex=True)\
                                .str.replace("\(0\.\d*\) ?", "", regex=True)\
                                .str.replace(", \.\d*\)","", regex=True)\
                                .str.replace(" ,","")\
                                .str.replace(", (", ", ")\
                                .str.strip("(")\
                                .str.replace(",",", ")\
                                .str.replace("  "," ")\
                                .str.replace("_"," ")\
                                .str.replace("(, \()?0?\.\d*\, \)","", regex=True)\
                                .str.replace("\( ?","", regex=True)\
                                .str.replace("0\.\d*\, ","", regex=True)\
                                .str.replace(" ,",",")\
                                .str.replace("), ", ", ")
    
    if "ml-1m" in f:
        df["answer"] = df["answer"]\
                                .str.replace(" . "," ")\
                                .str.replace(")-", " - ")\
                                .str.replace(".", " ")\
                                .str.replace(":",": ")
        
    df["result"] = df["answer"].str.split(",")
    
    df["result"] = df["result"].apply(lambda x: [el.strip(" ")  if len(el) > 0 and el[0]==" " else el for el in x ])
    return df

parent_folder = "RecAI/RecLM-eval/output"
cand = glob.glob(f"{parent_folder}/*/*/*")

for f in cand:
    if "steam" in f or "cleaned" in f:
        continue
    print(f)
    df = pd.read_json(f, lines=True)

    new_df = clean(df, f)

    new_df.to_json(f.replace("retrieval", "cleaned_retrieval"), index=False, lines=True, orient="records")
    