import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class DataGetter:
    def __init__(self):
        self.csv_data = pd.DataFrame()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def get_data(self, file: str = None, filepaths: list = None, type: str = "single"):
        if type.lower() == "single":
            data = pd.read_csv(file)
            self.csv_data = pd.concat([self.csv_data, data], ignore_index=True)
            self.csv_data["Date"] = pd.to_datetime(self.csv_data["Date"], format="mixed")
            self.csv_data = self.csv_data.sort_values(by = "Date", ascending=True)

            chunks = []
            chunk_size = 50
            for i in range(0, len(self.csv_data), chunk_size):
                chunk_df = self.csv_data.iloc[i:i+chunk_size]
                chunks.append(chunk_df.to_csv(index=False))

            self.database = FAISS.from_texts(chunks, embedding=self.embeddings)
            
        elif type.lower() == "multi":
            for file in filepaths:
                data = pd.read_csv(file)
                self.csv_data = pd.concat([self.csv_data, data], ignore_index=True, axis = 0)
            self.csv_data["Date"] = pd.to_datetime(self.csv_data["Date"], format="mixed")
            self.csv_data = self.csv_data.sort_values(by = "Date", ascending=True)

            chunks = []
            chunk_size = 50
            for i in range(0, len(self.csv_data), chunk_size):
                chunk_df = self.csv_data.iloc[i:i+chunk_size]
                chunks.append(chunk_df.to_csv(index=False))

            self.database = FAISS.from_texts(chunks, embedding=self.embeddings)