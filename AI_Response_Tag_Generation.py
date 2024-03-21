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

    def extract_integer_from_string(self , string):
        integers = re.findall(r'\d+', string)
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

    Give me the ID of the appropriate catalogue present in ths provided list in the form ["ID"]. Ignore all other information, just return this array in the same format."""
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
        final_tag = top_tags[index]
        return final_tag, top_tags

    def main(self , AI_Response):
        c0_tag, top_tags0 = self.tag_generation(AI_Response, self.c0_list, self.c0_list_emb_lst)
        c1_tag, top_tags1 = self.tag_generation(AI_Response, self.c1_list, self.c1_list_emb_lst)
        c2_tag, top_tags2 = self.tag_generation(AI_Response, self.c2_list, self.c2_list_emb_lst)
        c3_tag, top_tags3 = self.tag_generation(AI_Response, self.c3_list, self.c3_list_emb_lst)
        return c0_tag, c1_tag, c2_tag, c3_tag, top_tags0, top_tags1, top_tags2, top_tags3