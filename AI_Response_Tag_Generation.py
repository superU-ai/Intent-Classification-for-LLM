from sentence_transformers import util
import pandas as pd
import random
import openai
import torch
import time
import re

class Tag_generator:
    import warnings
    warnings.filterwarnings("ignore")

    def __init__(self , openai):
        self.c0_list_emb_df = pd.read_csv("Data/c0_List_Embeddings_DF.csv")
        self.c0_list = self.c0_list_emb_df.iloc[:, 0].tolist()
        self.c0_list_emb_lst = self.c0_list_emb_df.iloc[:, 1:].values.tolist()

        self.c1_list_emb_df = pd.read_csv("Data/c1_List_Embeddings_DF.csv")
        self.c1_list = self.c1_list_emb_df.iloc[:, 0].tolist()
        self.c1_list_emb_lst = self.c1_list_emb_df.iloc[:, 1:].values.tolist()

        self.c2_list_emb_df = pd.read_csv("Data/c2_List_Embeddings_DF.csv")
        self.c2_list = self.c2_list_emb_df.iloc[:, 0].tolist()
        self.c2_list_emb_lst = self.c2_list_emb_df.iloc[:, 1:].values.tolist()

        self.c3_list_emb_df = pd.read_csv("Data/c3_List_Embeddings_DF.csv")
        self.c3_list = self.c3_list_emb_df.iloc[:, 0].tolist()
        self.c3_list_emb_lst = self.c3_list_emb_df.iloc[:, 1:].values.tolist()

        self.df = pd.read_csv("superu_catalogue.txt", sep="/n")
        self.df["c0_name"] = self.df["content"].apply(lambda x: x.split(" > ")[0])
        self.df["c1_name"] = self.df["content"].apply(lambda x: x.split(" > ")[1] if len(x.split(" > ")) > 1 else "")
        self.df["c2_name"] = self.df["content"].apply(lambda x: x.split(" > ")[2] if len(x.split(" > ")) > 2 else "")
        self.df["c3_name"] = self.df["content"].apply(lambda x: x.split(" > ")[3] if len(x.split(" > ")) > 3 else "")
        
        self.openai = openai


    def get_top_tags(self , AI_Response, catalogue_names, catalogue_embeddings):
        emb_ques = self.openai.Embedding.create(input = AI_Response, engine = "")
        emb_ques_main = emb_ques["data"][0]["embedding"]

        cos_scores = util.cos_sim(emb_ques_main, catalogue_embeddings)[0]
        topk = min(100, len(catalogue_embeddings))
        top_results = torch.topk(cos_scores, k=topk)

        top_index = top_results[1].tolist()

        top_tags = [catalogue_names[i] for i in top_index]
        top_tags_openai = [f'{i}. ' + top_tags[i] for i in range (len(top_tags))]

        return top_tags, top_tags_openai
    
    def c0_c1_pipeline(self, c0):
        c1_available = self.df.groupby("c0_name").agg(c1_list_wrt_c0 = pd.NamedAgg(column = 'c1_name', aggfunc = lambda x: [name for name in x.unique().tolist() if name != ""])).reset_index()
        c1_list_c0 = c1_available[c1_available["c0_name"] == c0].c1_list_wrt_c0.values.tolist()[0]
        idx_list = [self.c1_list.index(i) for i in c1_list_c0]
        c1_c0_list_emb_lst = [self.c1_list_emb_lst[i] for i in idx_list]
        return c1_list_c0, c1_c0_list_emb_lst
    
    def c1_c2_pipeline(self, c0, c1):
        c1_available = self.df.groupby(["c0_name", "c1_name"]).agg(c1_list_wrt_c0 = pd.NamedAgg(column = 'c2_name', aggfunc = lambda x: [name for name in x.unique().tolist() if name != ""])).reset_index()
        c1_list_c0 = c1_available[(c1_available["c0_name"] == c0) & (c1_available["c1_name"] == c1)].c1_list_wrt_c0.values.tolist()[0]
        idx_list = [self.c2_list.index(i) for i in c1_list_c0]
        c1_c0_list_emb_lst = [self.c2_list_emb_lst[i] for i in idx_list]
        return c1_list_c0, c1_c0_list_emb_lst
    
    def c2_c3_pipeline(self, c0, c1, c2):
        c1_available = self.df.groupby(["c0_name", "c1_name", "c2_name"]).agg(c1_list_wrt_c0 = pd.NamedAgg(column = 'c3_name', aggfunc = lambda x: [name for name in x.unique().tolist() if name != ""])).reset_index()
        c1_list_c0 = c1_available[(c1_available["c0_name"] == c0) & (c1_available["c1_name"] == c1) & (c1_available["c2_name"] == c2)].c1_list_wrt_c0.values.tolist()[0]
        idx_list = [self.c3_list.index(i) for i in c1_list_c0]
        c1_c0_list_emb_lst = [self.c3_list_emb_lst[i] for i in idx_list]
        return c1_list_c0, c1_c0_list_emb_lst

    def extract_integer_from_string(self , string):
        integers = re.findall(r'-?\d+', string)
        integers = [int(num) for num in integers]
        try:
            integer_return = integers[0]
        except Exception as e:   #Assuming OpenAI does not return a number, we rely on the cosine similarity
            print("No integer present in OpenAI response. Refering to cosine similarity scores. \nError message: ", e)
            integer_return = 0 
        return integer_return

    def get_openai_prompt(self , AI_response, top_tags_openai):
        random.shuffle(top_tags_openai)
        prompt = f"""You are given with an AI response.

    AI generated response - {AI_response}

    Your task is to help me decide the catalogue which this response belongs to:

    Make sure to chose only one appropriate catalogue set from my catalogue list given - {top_tags_openai}
    List is of the form "ID. Catalogue"

    Give me the ID of the appropriate catalogue present in ths provided list in the form ["ID"]. Ignore all other information, just return this array in the same format.
    If none of the categories match the given AI Response then just give me ["-1"]"""
        return prompt

    def openai_response(self , prompt):
        messages = [{"role": "system", "content":"You are a helpful assistant."}]
        message_r = {"role": "user", "content": f"{prompt}"}
        messages.append(message_r)
        try:
            response = self.openai.ChatCompletion.create(engine="", messages=messages)
        except Exception as e:
            print("OpenAI response generation error. Retrying after 2 minutes. \nError Message: ", e)
            time.sleep(120)
            response = self.openai.ChatCompletion.create(engine="", messages=messages)
        response_message = response["choices"][0]["message"]["content"]
        print(response_message)
        index = self.extract_integer_from_string(response_message)
        return index

    def tag_generation(self , AI_Response, catalogue_names, catalogue_embeddings):
        top_tags, top_tags_openai = self.get_top_tags(AI_Response, catalogue_names, catalogue_embeddings)
        prompt = self.get_openai_prompt(AI_Response, top_tags_openai)
        index = self.openai_response(prompt)
        if index == -1:
            return "", top_tags
        try:
            final_tag = top_tags[index]
        except Exception as e:
            print("OpenAI gave wrong index")
            final_tag = top_tags[0]
        return final_tag, top_tags
    
    def get_productnames_features(self, AI_Response):
        prompt = f"""You are provided with an AI generated statement.
    
# Statement: {AI_Response}

Instruction : You are a product recommendation engine whose primary job is to infer products required by the customers. Please find out 
all the products, product types and Product features of the products to be recommended in the statement

Give me the output in a single json format """ + """[{"products": [list of products], "product types": [list of product types], "product features": [list of features]}].\nPlease follow the exact same format."""

        messages = [{"role": "system", "content":"You are a smart and helpful assistant."}]
        message_r = {"role": "user", "content": f"{prompt}"}
        messages.append(message_r)
        try:
            response = self.openai.ChatCompletion.create(engine="", messages=messages)
        except Exception as e:
            print("OpenAI product response generation error. Retrying after 2 minutes \nError Message: ", e)
            time.sleep(120)
            response = self.openai.ChatCompletion.create(engine="", messages=messages)
        response_message = response["choices"][0]["message"]["content"]
        return response_message

    def main(self , AI_Response):
        c0_tag, top_tags0 = self.tag_generation(AI_Response, self.c0_list, self.c0_list_emb_lst)
        
        c1_list_c0, c1_c0_list_emb_lst = self.c0_c1_pipeline(c0_tag)
        try:
            c1_tag, top_tags1 = self.tag_generation(AI_Response, c1_list_c0, c1_c0_list_emb_lst)         # c1_list, c1_list_emb_lst
        except Exception as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                c1_tag = ""
                top_tags1 = []
            
        c2_list_c1, c2_c1_list_emb_lst = self.c1_c2_pipeline(c0_tag, c1_tag)
        try:
            c2_tag, top_tags2 = self.tag_generation(AI_Response, c2_list_c1, c2_c1_list_emb_lst)         # c2_list, c2_list_emb_lst
        except Exception as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                c2_tag = ""
                top_tags2 = []
        
        c3_list_c2, c3_c2_list_emb_lst = self.c2_c3_pipeline(c0_tag, c1_tag, c2_tag)
        try:
            c3_tag, top_tags3 = self.tag_generation(AI_Response, c3_list_c2, c3_c2_list_emb_lst)         # c3_list, c3_list_emb_lst
        except Exception as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                c3_tag = ""
                top_tags3 = []
                
        productnames_features = self.get_productnames_features(AI_Response)
        
        return c0_tag, c1_tag, c2_tag, c3_tag, top_tags0, top_tags1, top_tags2, top_tags3, productnames_features