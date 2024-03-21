from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import openai
import time
openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Data/superu_catalogue.txt", sep="/n")

df["c0_name"] = df["content"].apply(lambda x: x.split(" > ")[0])
df["c1_name"] = df["content"].apply(lambda x: x.split(" > ")[1] if len(x.split(" > ")) > 1 else "")
df["c2_name"] = df["content"].apply(lambda x: x.split(" > ")[2] if len(x.split(" > ")) > 2 else "")
df["c3_name"] = df["content"].apply(lambda x: x.split(" > ")[3] if len(x.split(" > ")) > 3 else "")

c0_list = [name for name in df["c0_name"].unique().tolist() if name != ""]
c1_list = [name for name in df["c1_name"].unique().tolist() if name != ""]
c2_list = [name for name in df["c2_name"].unique().tolist() if name != ""]
c3_list = [name for name in df["c3_name"].unique().tolist() if name != ""]

def create_embeddings(txt):
    try:
        emb = openai.Embedding.create(input = txt, engine = "")
    except: #Assuming if rate limit exceesdes then wait for 2 minutes and retry again
        time.sleep(120)
        emb = openai.Embedding.create(input = txt, engine = "")
    emb_main = emb["data"][0]["embedding"]
    return txt, emb_main

c0_list_emb_df = pd.DataFrame()
with ThreadPoolExecutor() as executor:
    for txt, embeddings in executor.map(create_embeddings, c0_list):
        append_df = pd.DataFrame(embeddings).T
        append_df.insert(0, "Catalogue", [txt])
        c0_list_emb_df = pd.concat([c0_list_emb_df, append_df], ignore_index = True)
c0_list_emb_df.to_csv("Data/c0_List_Embeddings_DF.csv", index = False)


c1_list_emb_df = pd.DataFrame()
with ThreadPoolExecutor() as executor:
    for txt, embeddings in executor.map(create_embeddings, c1_list):
        append_df = pd.DataFrame(embeddings).T
        append_df.insert(0, "Catalogue", [txt])
        c1_list_emb_df = pd.concat([c1_list_emb_df, append_df], ignore_index = True)
c1_list_emb_df.to_csv("Data/c1_List_Embeddings_DF.csv", index = False)


c2_list_emb_df = pd.DataFrame()
with ThreadPoolExecutor() as executor:
    for txt, embeddings in executor.map(create_embeddings, c2_list):
        append_df = pd.DataFrame(embeddings).T
        append_df.insert(0, "Catalogue", [txt])
        c2_list_emb_df = pd.concat([c2_list_emb_df, append_df], ignore_index = True)
c2_list_emb_df.to_csv("Data/c2_List_Embeddings_DF.csv", index = False)


c3_list_emb_df = pd.DataFrame()
with ThreadPoolExecutor() as executor:
    for txt, embeddings in executor.map(create_embeddings, c3_list):
        append_df = pd.DataFrame(embeddings).T
        append_df.insert(0, "Catalogue", [txt])
        c3_list_emb_df = pd.concat([c3_list_emb_df, append_df], ignore_index = True)
c3_list_emb_df.to_csv("Data/c3_List_Embeddings_DF.csv", index = False)