import os
import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import vertexai
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI

load_dotenv()

PROMPT_TEMPLATE = """
Answer the question  or check the correctness of the response based on the following context:

{context}

----
Use latex formatting for equations. Enclose equations in $ sign.
Use markdown formatting for everything else.
Here's the user input: {question}
----
If this is a question answer the question, if this is an attempt to answer a previous question, check if the answer is correct.
"""

SUMMARIZATION_PROMPT="""
Distill the above chat messages into a single summary message. Include as many specific details as you can.
"""

SOCRATIC_PROMPT_TEMPLATE="""
You are a tutor trying to help a student learn a concept. You are helping them with a problem and want to help them understand the concepts by figuring out the solution themselves with only small nudges in the right direction.

Here's the entire conversation with them so far: {summary}

Here's the student's most recent response: {response}

Based on the solution to the question, use the socratic method to guide the student towards the answer.

Do no answer the question, but provide hints or prompt the student to think of the next step.
"""



def get_vertex_embeddings():
    vertexai.init(
        project=os.environ.get("VERTEX_PROJECT_ID"),
        location=os.environ.get("VERTEX_PROJECT_LOCATION")
        )

    # Initialize the a specific Embeddings Model version
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    
    return embeddings

def get_vector_retriever():
    QDRANT_URL=os.environ.get("QDRANT_URL")
    QDRANT_API_KEY=os.environ.get("QDRANT_API_KEY")
    vectorstore = QdrantVectorStore.from_existing_collection(
        collection_name="dsa_notes",
        embedding=get_vertex_embeddings(),
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    return vectorstore.as_retriever(k=2)

async def stream_openai(query_text:str):
    
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_socratic = ChatPromptTemplate.from_template(SOCRATIC_PROMPT_TEMPLATE)
    llm = ChatOpenAI(temperature=0, streaming=True)
    model = ChatVertexAI(
        model_name="gemini-1.5-flash-001", 
        location=os.environ.get("VERTEX_PROJECT_LOCATION"), 
        project=os.environ.get("VERTEX_PROJECT_ID")
        )
    retriever = get_vector_retriever()
    history = ChatMessageHistory()
    
    retrieval_chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    history.add_user_message(query_text)
    # chat_with_message_history = RunnableWithMessageHistory(
    #     retrieval_chain,
    #     lambda session_id: history,
    #     input_messages_key="question",
    #     history_messages_key="chat_history",
    # )
    
    socratic_chain = (
        {'summary': retrieval_chain, "response": RunnablePassthrough()}
        | prompt_socratic
        | model
        | StrOutputParser()
    )
    
    
    
    config = {"configurable": {"thread_id": "1"}} 
    async for chunk in socratic_chain.astream(query_text):
        yield f"{chunk}"
        
        

# Test
async def text_stream(query_text):
    async for c in stream_openai(query_text):
        print(c, end="", flush=True)
        
if __name__ == "__main__":
    asyncio.run(text_stream("I'm very stuck! How do you integrate 5x^2?"))