# normalized-cosine

## Introduction

Welcome to the normalized cosine similarity project.  This is a code-light theory-heavy solution to the string inclusion problem.  Specifically, if you have a transformer embedding (OpenAI, BERT) of a longer string (v_l) and a shorter string (v_s), the cosine similarity of these embeddings will be highly sensitive to length.  Let's say we have three strings:

1) `I love my dog`
2) `I love my dog, she is so wonderful and cute, and let me tell you about ...`
3) `The dog park is a wonderful place to take your pet, you are bound to meet other dogs ...`

(where `...` represents a random continuation)

Let's say (2) and (3) are of similar length, and have a cosine similarity of around 0.6, perhaps because they share similar words although with completely different context and syntactic structure.  Then, let's say (1) and (2) have a similarity of 0.46.  This is a serious problem, because we can plainly see that (1) is a substring of (2).

Industry deals with this problem using several methods, such as:

1) Chopping the longer string into shorter strings, taking the embeddings of those shorter strings, and then comparing the query to the shorter strings.
2) Feeding the longer string and the shorter string into an LLM prompt, with instructions to determine string inclusion.
3) Concatenating a bag-of-words vector onto the transformer embedding, taking the similarity, and looking to see if there were direct matches in the bag-of-words vector.

(1) and (2) are expensive and time-consuming, and (3) depends on an exact string match between the two strings, which would fail in cases like `love the dog' and `adore the beagle`.  

In this project, we are modeling search using cosine similarity as a compression problem.  We assume a shorter query and an index of longer strings.  In our compression analogy, the shorter query is ``decompressed'' into a longer string, and our ranking is a search for the best decompression of the query.  We assume that parts of the longer string matching the query are ``signal'' and parts that do not match are compression artifacts, or ``noise''.  The analogy to compression allows us to borrow the compression ratio.  `a` is the cosine similarity of the embeddings, `N_l` is the token count for the longer string, and `N_s` is the token count for the shorter string.  The final ranking metric is:
```
v_l^T v_s [log N_l - log N_s]
```
See the pdf for an explanation of how this was derived.

## Installation

This requires Python 3.13 or above.  You will also need an access key for the OpenAI API.  First, install the requirements:
```
pip install -r requirements.txt
```
Then, you need to copy `dotenv_example.txt` to .env and fill out your OpenAI access key:
```
cp dotenv_example.txt .env
cat .env
OPENAI_KEY="key"
OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
```
Replace `key` with your OpenAI API key.  If you would like to use a different model, change `OPENAI_EMBEDDING_MODEL`.

## Running

To see the function running on a simple example, you can run:
```
python similarity.py
```
The usage is found in the `__main__` method in this text.
