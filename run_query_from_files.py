'''
python3 watch_file.py -p1 python3 run_query_from_files.py example_data/query.txt example_data/strings.txt -d .
'''
from typing import List
from similarity import submit_query
import sys
import pprint as pp

def lines_from_file(fileNameOrString:str) -> List[str]:

    try: 

        with open(fileNameOrString,'r') as f:

            st=f.read()

    except Exception:

        return [fileNameOrString]

    return [s for s in st.split('\n') if s]
    

if __name__=='__main__':

    query=lines_from_file(sys.argv[1])[0]
    lines=lines_from_file(sys.argv[2])
    result=submit_query(query,lines)

    pp.pprint(result)

    
    
