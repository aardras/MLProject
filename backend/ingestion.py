import pandas as pd
from pathlib import Path
import collections

'''
steps of ingestion
load file
infer data type.
do null count sum
infer col name as well
store all this in a dictionary and return
'''

class Ingest():
    def __init__(self):
        self.df:pd.DataFrame = None
        self.filepath:str = None
        self.schema:dict = {}
    
    def ingest(self, filepath: str):

        self.filepath = filepath
        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Given file: {filepath} does not exist")
        
        self.df = pd.read_csv(path)

        #colnames
        colnames = self.df.columns.tolist()

        #nullcounts
        nullcounts = collections.defaultdict(int)
        for col in colnames:
            nulls = self.df[col].isnull().sum()
            nullcounts[col] = int(nulls)
        
        #datatypes
        datatypes = {}
        for col in colnames:
            datatypes[col] = str(self.df[col].dtype)
        
        #rowcount
        rowcount = len(self.df)

        #full schema
        self.schema = {
            'data': self.df,
            'filepath': path,
            'colnames': colnames,
            'datatypes': datatypes,
            'null counts': nullcounts,
            'total entries': rowcount,
        }

        return self.schema
            

