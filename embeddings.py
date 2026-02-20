'''
python3 watch_file.py -p1 python3 embeddings.py -d .
'''
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
from typing import List,Union
import numpy as np
import tiktoken

load_dotenv()

#############
# CONSTANTS #
#############

OPENAI_KEY=os.environ['OPENAI_KEY']
OPENAI_EMBEDDING_MODEL=os.environ['OPENAI_EMBEDDING_MODEL']
OPENAI_EMBEDDING_DIMENSIONS=int(os.environ['OPENAI_EMBEDDING_DIMENSIONS'])
TIKTOKEN_MODEL=os.environ['TIKTOKEN_MODEL']
TIKTOKEN_ENCODING=tiktoken.get_encoding(TIKTOKEN_MODEL)


def get_client(key=OPENAI_KEY) -> openai.OpenAI:

    return OpenAI(api_key=key)


CLIENT=get_client()


#############
# FUNCTIONS #
#############

def get_embeddings(strings: str|List[str], 
                   model:str=OPENAI_EMBEDDING_MODEL,
                   client:openai.OpenAI=CLIENT,
                   dimensions:int=OPENAI_EMBEDDING_DIMENSIONS,
                  ) -> np.array:
    '''
    Gathers embeddings and postprocesses them:

    1) Ensures a list of strings is submitted.
    2) Sends a response to the OpenAI API.
    3) Postprocesses the embeddings, ensuring they are L2-normalized.
    '''
    if not isinstance(strings,list): # (1)

        strings=[strings]


    response=client.embeddings.create(model=model, # (2)
                                      input=strings,
                                      encoding_format="float", 
                                      dimensions=dimensions,
                                     )
    
    embeddings=[] # (3)

    for data in response.data:

        v=data.embedding
        v/=np.linalg.norm(v)

        embeddings.append(v)

    return np.array(embeddings)

    
def num_tokens(st,encoding=TIKTOKEN_ENCODING):

    return len(encoding.encode(st))


if __name__=='__main__':

    s1='''
    I haven’t set it (PYTHONPATH) before; what I am doing just go with command prompt and type CMD anywhere (since python.exe is in my shell PATH). If I try to access Window ENVIRONMENT variable, it gives mapped value but the problem with Python ENVIRONMENT variable like; PYTHONPATH and PYTHONHOME.
    '''
    s2='''
    I try to access the environment.
    '''
    embeddings=get_embeddings([s1,s2])
    
    
    print(embeddings.shape)
