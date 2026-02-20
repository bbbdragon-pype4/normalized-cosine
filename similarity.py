'''
python3 watch_file.py -p1 python3 similarity.py -d .
'''
from embeddings import get_embeddings, num_tokens
from typing import List,Union,Tuple,List,Dict,Set
import numpy as np
import tiktoken
from operator import itemgetter
import pprint as pp

#############
# CONSTANTS #
#############

ENCODER=tiktoken.get_encoding("cl100k_base")


#############
# FUNCTIONS #
#############

def vectors_and_lengths(strings:List[str],
                        encoder:tiktoken.core.Encoding=ENCODER,
                       ) -> Tuple[np.array,np.array]:
    '''
    Helper to extract embeddings and log token counts for a list of strings.
    '''
    vecs=get_embeddings(strings)
    lengths=np.array([np.log(num_tokens(st)) for st in strings])

    return lengths,vecs


def strong_recommendations(ranking:np.array,
                           similarities:np.array,
                           diffs:np.array,
                           strings:List[str]) -> Set[str]:
    '''
    Function to gather a set of strong recommendations for string inclusion.

    1) Test if the rank is above 1.
    2) If it is, take two intervals, d0=|sim - 1/delta| and d1=|1-sim|.
    3) If d0 > d1, include in recommendations.
    '''
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
    '''
    Main function to run queries.  

    1) If lengths and precomputedEmbeddings are not None, they will be used as precomputed 
    lengths and embeddings.  Otherwise compute the embeddings.
    2) Compute lengths and embeddings for query.
    3) Compute the diffs, similarities, and rankings.
    4) Compute recommendations.
    5) Create a closure to return a string for each recommendation - 'probable', 'strongly
    probable', or 'undetermined'.
    6) Create a list of dicts, each dict containing the string, ranking, and recommendation.
    Sort based on ranking.
    '''
    if precomputedEmbeddings is None or lengths is None: # (1)

        lengths,embeddings=vectors_and_lengths(strings)

    queryLength,queryEmbedding=vectors_and_lengths([q]) # (2) 
    queryLength=queryLength[0]
    queryEmbedding=queryEmbedding[0]
    
    diffs=lengths-queryLength # (3)
    similarities=np.dot(embeddings,queryEmbedding)
    ranking=similarities*diffs

    recommendations=strong_recommendations(ranking,similarities,diffs,strings) # (4)

    def recommendation_f(st,rnk): # (5)

        if rnk > 1:

            return 'probable'

        elif st in recommendations:

            return 'strongly probabile'

        else:

            return 'undetermined'


    distancesAndStrings=[{'string':st,  # (6)
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
