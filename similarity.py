'''
python3 watch_file.py -p1 python3 similarity.py -d .
'''
from embeddings import get_embeddings
from typing import List,Union,Tuple,List,Dict,Set
import numpy as np
import tiktoken
from operator import itemgetter
import pprint as pp

ENCODER=tiktoken.get_encoding("cl100k_base")

def vectors_and_lengths(strings:List[str],
                        encoder:tiktoken.core.Encoding=ENCODER,
                       ) -> Tuple[np.array,np.array]:

    vecs=get_embeddings(strings)
    lengths=np.array([np.log(len(encoder.encode(st))) for st in strings])

    return lengths,vecs


def strong_recommendations(ranking:np.array,
                           similarities:np.array,
                           diffs:np.array,
                           strings:List[str]) -> Set[str]:

    recommendations={}

    for (rank,similarity,diff,st) in zip(ranking,similarities,diffs,strings):

        if rank > 1:

            invDiff=1/(diff+1e-10)
            interval1=np.abs(sim-invDiff)
            interval2=np.ans(1-sim)

            if interval1 > interval2:

                recommendations.add(st)

    return recommendations


def submit_query(q:str,
                 strings:List[str],
                 lengths:np.array=None,
                 precomputedEmbeddings:np.array=None,
                ) -> List[Dict]:

    if precomputedEmbeddings is None or lengths is None:

        lengths,embeddings=vectors_and_lengths(strings)

    queryLength,queryEmbedding=vectors_and_lengths([q])
    queryLength=queryLength[0]
    queryEmbedding=queryEmbedding[0]
    
    diffs=lengths-queryLength
    similarities=np.dot(embeddings,queryEmbedding)
    ranking=similarities*diffs
    recommendations=strong_recommendations(ranking,similarities,diffs,strings)

    def recommendation_f(st,rnk):

        if rnk > 1:

            return 'probable string inclusion'

        elif st in recommendations:

            return 'strong probability of string inclusion'

        else:

            return 'undetermined'


    distancesAndStrings=[{'string':st,
                          'ranking':float(rnk),
                          'recommendation':recommendation_f(st,rnk),
                         } for (st,rnk) in zip(strings,ranking)]
    distancesAndStrings=sorted(distancesAndStrings,
                               key=itemgetter('ranking'),
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
