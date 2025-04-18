import readline

from dotenv import load_dotenv
load_dotenv()

def llm(messages, temperature=1):
    '''
    >>> llm([
    ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
    ...     {'role': 'user', 'content': 'What is the capital of France?'},
    ...     ], temperature=0)
    'The capital of France is Paris!'
    '''
    import groq
    client = groq.Groq()

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=temperature,
    )
    return chat_completion.choices[0].message.content


def chunk_text_by_words(text, max_words=5, overlap=2):
    """
    Splits text into overlapping chunks by word count.

    Args:
        text (str): The input document as a string.
        max_words (int): Maximum words per chunk.
        overlap (int): Number of overlapping words between chunks.

    Returns:
        List[str]: A list of word-based text chunks.

    Examples:
        >>> text = "The quick brown fox jumps over the lazy dog. It was a sunny day and the birds were singing."
        >>> chunks = chunk_text_by_words(text, max_words=5, overlap=2)
        >>> len(chunks)
        7
        >>> chunks[0]
        'The quick brown fox jumps'
        >>> chunks[1]
        'fox jumps over the lazy'
        >>> chunks[4]
        'sunny day and the birds'
        >>> chunks[-1]
        'the birds were singing.'
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap

    return chunks


import string

def score_chunk(chunk: str, query: str) -> float:
    """
    Scores a chunk against a user query using Jaccard similarity of word sets.

    Args:
        chunk (str): The text chunk from the document.
        query (str): The user query string.

    Returns:
        float: A score between 0 and 1 indicating relevance (1 = most relevant).

    Examples:
        >>> score_chunk("Python is a programming language.", "What is Python?")
        0.3333333333333333
        >>> score_chunk("The sun is hot and bright.", "How hot is the sun?")
        0.5714285714285714
        >>> score_chunk("Bananas are yellow.", "How do airplanes fly?")
        0.0
    """
    def preprocess(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return set(text.split())

    chunk_words = preprocess(chunk)
    query_words = preprocess(query)

    if not chunk_words or not query_words:
        return 0.0

    intersection = chunk_words & query_words
    union = chunk_words | query_words

    return len(intersection) / len(union)


if __name__ == '__main__':
    messages = []
    messages.append({
        'role': 'system',
        'content': 'You are a helpful assistant.  You always speak like a pirate.  You always answer in 1 sentence.'
    })
    while True:
        # get input from the user
        text = input('docchat> ')
        # pass that input to llm
        messages.append({
            'role': 'user',
            'content': text,
        })
        result = llm(messages)
        # FIXME:
        # Add the "assistant" role to the messages list
        # so that the `llm` has access to the whole
        # conversation history and will know what it has previously
        # said and update its response with that info.

        # print the llm's response to the user
        print('result=', result)
        import pprint
        pprint.pprint(messages)