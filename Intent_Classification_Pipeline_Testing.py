import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification


class Intent_Classifier:
    def __init__(self):
        #Initialize tokenizer and base model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model_binary = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.model_conversation_contd = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

        #Specify device for importing model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        #Import saved model
        self.model_binary_path = "binary_model.pth"
        self.model_binary.load_state_dict(torch.load(self.model_binary_path, map_location=torch.device(self.device)))
        self.model_conversation_contd_path = "no_api_model.pth"
        self.model_conversation_contd.load_state_dict(torch.load(self.model_conversation_contd_path, map_location=torch.device(self.device)))

    def get_intent(self, user_question):
# #Build a continuous loop
# while True:
#     user_question = input("Question: ")

#     #Break condition
#     if user_question.lower() == 'exit':
#         break

            #Tokenize user question
            user_question_encoding = self.tokenizer(user_question, truncation=True, padding=True, return_tensors='pt')

            #Generate model output
            with torch.no_grad():
                input_ids = user_question_encoding['input_ids'].to(self.device)
                attention_mask = user_question_encoding['attention_mask'].to(self.device)
                output = self.model_binary(input_ids, attention_mask=attention_mask)
                predicted_label_id_binary = torch.argmax(output.logits, dim=1).item()

            # If user query
            if predicted_label_id_binary == 1:
                with torch.no_grad():
                    input_ids = user_question_encoding['input_ids'].to(self.device)
                    attention_mask = user_question_encoding['attention_mask'].to(self.device)
                    output = self.model_conversation_contd(input_ids, attention_mask=attention_mask)
                    predicted_label_id_no = torch.argmax(output.logits, dim=1).item()
                # print(predicted_label_id_no)
                if predicted_label_id_no == 0:
                    # print("Emotional Support")
                    return "Support"
                elif predicted_label_id_no == 1:
                    # print("FAQ")
                    return "FAQ"
                elif predicted_label_id_no == 2:
                    # print("Information")
                    return "Information"
                elif predicted_label_id_no == 3:
                    # print("Support")
                    return "Support"
                
            else:
                # print("Purchase Intent")
                return "Transactional"