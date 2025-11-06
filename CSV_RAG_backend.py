from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from CSV_reader import DataGetter
import os
import pandas as pd

class CSVRagChatbot():
    def __init__(_self):
        _self.llm = ChatGroq(
            api_key="YOUR API KEY",
            model="openai/gpt-oss-120b"
        )
        _self.prompt = ChatPromptTemplate.from_template(
            """
                You are an assistant that has only one role and that is to assess the given dataframe 
                and answer questions strictly on the context given to you.
                The Dataframe as your context is:
                
                <context>
                {context}
                </context>
                
                You're role is to just give answers which shall help the person get to know about the attendance report of various classes and different dates, 
                thus if anyone asks a question which doesn't have a connection to your context, then simply deny them service,
                and say "Sorry, I cannot help you with that Please ask another question" 
                and tell them to ask another question which is related to your purpose.
                Any out of context questions are meant to be denied.

                Question: {input}
            """
        )
        _self.chain = _self.prompt | _self.llm

    def get_response(_self, text, query):
        response = _self.chain.invoke({"context": text, "input": query})
        return response
    
# if __name__ == '__main__':
#     csv_files = []
#     DATA_PATH = "/Users/shreyassawant/mydrive/Shreyus_workspace/Semester_VII/CV/project/attendance"
#     for file in os.listdir(DATA_PATH):
#         if file.endswith(".csv"):
#             csv_files.append(os.path.join(DATA_PATH, file))
    
#     dg = DataGetter()
#     dg.get_data(filepaths=csv_files, type="multi")
#     query = "what is % attendance of shreyas?"
    # docs = dg.database.similarity_search(query=query, k=5)
    # context = "\n\n".join([d.page_content for d in docs])

    # bot = CSVRagChatbot()
    # response = bot.get_response(text=context, query=query)
    # print(response.content)