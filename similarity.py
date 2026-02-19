'''
python3 watch_file.py -p1 python3 similarity.py -d .
'''
from embeddings import get_embeddings
from typing import List,Union,Tuple
import numpy as np
import tiktoken
from operator import itemgetter
import pprint as pp

ENCODER=tiktoken.get_encoding("cl100k_base")

def vectors_and_lengths(strings:List[str],
                        encoder:tiktoken.core.Encoding=ENCODER,
                       ) -> Tuple[np.array,np.array]:

    vecs=get_embeddings(strings)
    lengths=np.log([len(encoder.encode(st)) for st in strings])

    return lengths,vecs


def submit_query(q:str,
                 strings:List[str],
                 lengths:np.array=None,
                 precomputedEmbeddings:np.array=None,
                ):

    if precomputedEmbeddings is None or lengths is None:

        lengths,embeddings=vectors_and_lengths(strings)

    queryLength,queryEmbedding=vectors_and_lengths([q])
    queryLength=queryLength[0]
    queryEmbedding=queryEmbedding[0]
    
    diffs=lengths-queryLength
    similarities=np.dot(embeddings,queryEmbedding)
    similarities*=diffs
    distancesAndStrings=[{'string':st,
                          'similarity':sim,
                         } for (st,sim) in zip(strings,similarities)]
    distancesAndStrings=sorted(distancesAndStrings,
                               key=itemgetter('similarity'),
                               reverse=True,
                              )

    return distancesAndStrings
    

if __name__=='__main__':

    query='love dog'
    strings=['i love you',
             'i love my dog',
             'i love my cat',
            ]
    distancesAndStrings=submit_query(query,strings)

    pp.pprint(distancesAndStrings)
