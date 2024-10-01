import os
import asyncio
from dotenv import load_dotenv
from typing import Literal

from groq import APIError, BadRequestError
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
import vertexai
from graphviz import Digraph
from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.document_loaders import WikipediaLoader


load_dotenv()
llm = ChatGroq(model="llama-3.2-90b-text-preview")
vertexai.init(
        project=os.environ.get("VERTEX_PROJECT_ID"),
        location=os.environ.get("VERTEX_PROJECT_LOCATION")
        )
embeddings = VertexAIEmbeddings(model_name=os.environ.get("VERTEX_MODEL_ID"), 
                                location=os.environ.get("VERTEX_PROJECT_LOCATION"), 
                                project=os.environ.get("VERTEX_PROJECT_ID")
                                )



class Node(BaseModel):
    id: int
    label: str
    color: str

class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"
    
class KnowledgeGraph(BaseModel):
    nodes: list[Node] = Field(..., default_factory=list, description="List of nodes in the graph")
    edges: list[Edge] = Field(..., default_factory=list, description="List of edges in the graph")
    
    def return_graph(self):
        dot = Digraph(comment="Knowledge Graph")
        
        # Add nodes
        for node in self.nodes:
            dot.node(str(node.id), node.label, color=node.color)
            
        # Add edges
        for edge in self.edges:
            dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge.color)
        
        # Return graph
        return dot

class State(MessagesState):
    summary: str
    context: str
    solve_retrieval: str
    socratic_retrieval: str
    safe: bool = True
    graph: KnowledgeGraph

simple_solver_prompt = """You are helping with solving a student's questions about data structures and algorithms.

These can be questions about definitions, help with debugging code or hard leetcode style problems to solve.

Think carefully before answering any question. Explain your reasoning.

Do not hallucinate. Do not make up facts. If you don't know how to answer a problem, just say so.

Be concise."""

def simple_solver(state: State):
    
    messages = [SystemMessage(content=simple_solver_prompt)]
    
    # # get summary if it exists
    # summary = state.get("summary", "")
    
    # # if there is summary, we add it to prompt
    # if summary:
        
    #     # add summary to system message
    #     summary_message = f"Summary of conversation earlier: {summary}"
        
    #     # append summary to any newer message
    #     messages += [HumanMessage(content=summary_message)]
    
    messages += state["messages"]
    
    response = llm.invoke(messages)
    # NEED TO PREVENT CONTEXT FROM BALLOONING if I change it to list and want to persist that
    return {"context": response.content}



socratic_prompt = """You are a tutor trying to help a student gain a very strong understanding of concepts and problems in data structures and algorithms. If you do a good job, the student will tip you $200.

You are helping them with a problem and want to help them understand the concepts by figuring out the solution themselves with only nudges in the right direction.

You have the solution below along with the relevant excerpts from a textbook but the student has never seen either. You also have the transcript of the conversation you've had with the student so far.

Analyze the full conversation. Pay particular attention to what the overall goal of the conversation is and to the final question posed or answer to be checked submitted by the student. 

Reply to the student using the Socratic approach and meaningful questions to motivate the answer for them."""


def socratic(state: State):
    rag_solver_context = state.get("solve_retrieval", "")
    rag_socratic_context = state.get("socratic_retrieval", "")
    messages = [SystemMessage(socratic_prompt + "\n\n---------\n\nThe solution:\n" + state["context"] + "\n\n------------\n\nAdditional context from a textbook:\n" + rag_solver_context + "\n\n--------------\n\nNext questions, exercises, examples from a textbook:\n" + rag_socratic_context)]
    summary = state.get("summary", "")
    if summary:
        summary_message = f"Summary of your conversation with the student: {summary}"
        messages += [HumanMessage(content=summary_message)]
    messages += state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": response}


def summarize_conversation(state: State):
    prompt = SystemMessage(content="Extend the summary below by taking into account the new messages in a concise and precise manner. Pay particular attention to the original discussion topic/goal and the most recent exchange. Make sure to record what the student is having difficulty understanding.")
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Be concise and extend the summary by taking into account the new messages below:"
        )
    else:
        summary_message = "Create a summary of the conversation below:"
    
    messages = [prompt] + [HumanMessage(content=summary_message)] + state["messages"]
    response = llm.invoke(messages)
    
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

def should_summarize(state: State):
    """Return whether to summarize depending on length of messages"""
    messages = state["messages"]
    
    if len(messages) > 2:
        return "summarize"
    return END


# Retriever
def get_vector_store():
    QDRANT_URL=os.environ.get("QDRANT_URL")
    QDRANT_API_KEY=os.environ.get("QDRANT_API_KEY")
    vectorstore = QdrantVectorStore.from_existing_collection(
        collection_name="dsa_notes",
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    return vectorstore

vectorstore = get_vector_store()
notes_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# LLM for retriever
class RAGQueries(BaseModel):
    """ Split user query into 2 retrieval queries """
    solver_query: str = Field(None, description="Concise retrieval query for solving the student query by finding useful information in the textbooks.")
    socratic_query: str = Field(None, description="Different from the solve query. Concise retrieval query for finding useful examples, exercises and problems in the textbooks to guide the student to the next step.")

retrieval_llm = llm.with_structured_output(RAGQueries)

def retriever_prompts(state: State):
    message = [state["messages"][-1]]
    if len(state["messages"])>1:
        message = [state["messages"][-2]] + message
    response = retrieval_llm.invoke(message)
    solve = '-----------------------'.join(x.page_content for x in notes_retriever.invoke(response.solver_query))
    socratic ='-----------------------'.join(x.page_content for x in notes_retriever.invoke(response.socratic_query)) 
    return {"solve_retrieval": solve, "socratic_retrieval": socratic}


def solver_retrieve(state: State):
    rag_solver_retrieval = state.get("solve_retrieval", "")
    if not rag_solver_retrieval:
        return {"solve_retrieval": ""}
    message = [SystemMessage(content="You are given a snippet of a conversation between a student their teacher. You are also given notes from a textbook which might be helpful in solving the student's query or checking their answer. If the textbook notes are useful, use them so answer the student. Do not make anything up, only use the notes. If the notes are not useful, do not reply.")]
    if len(state["messages"])>1:
        message += state["messages"][-2]
    message += [state["messages"][-1]]
    message += [HumanMessage(content="\n-------------\nNotes from a textbook" + state["solve_retrieval"])]
    response = llm.invoke(message)
    return {"solve_retrieval": response.content}

def socratic_retrieve(state: State):
    rag_socratic_retrieval = state.get("socratic_retrieval", "")
    if not rag_socratic_retrieval:
        return {"socratic_retrieval": ""}
    message = [SystemMessage(content="You are given a snippet of a conversation between a student their teacher. You are also given notes from a textbook which might be helpful in guiding the student towards the next question to tackle to improve their understanding. If the textbook notes are useful, use them to find avenues to prompt the student with questions and exercises. Do not make anything up, only use the notes. If the notes are not useful, do not reply.")]
    if len(state["messages"])>1:
        message = [state["messages"][-2]]
    message += [state["messages"][-1]]
    message += [HumanMessage(content="\n-------------\nNotes from a textbook" + state["socratic_retrieval"])]
    response = llm.invoke(message)
    return {"socratic_retrieval": response.content}

# def retriever(state: State):
#     additional_context = notes_retriever.invoke(state['context'])
#     return {'context': state['context'] + '\n\n\n-----------------------\n\n\n Potentially useful information from a textbook that the student cannot see:\n' + '-----------------------'.join(x.page_content for x in additional_context)}

# Safety checker
llm_guard = ChatGroq(model="llama-guard-3-8b")

guard_prompt = ChatPromptTemplate.from_messages([
    ("user", "{query}"),
])

guard = guard_prompt | llm_guard

def safety_checker(state: State):
    message = state["messages"][-1]
    if 'safe' == guard.invoke({"query": message}).content:
        return {"safe": True}
    else:
        delete_message = [RemoveMessage(id=message.id)]
        return {"safe": False, "messages": delete_message} 

def safety_router(state: State):
    if state["safe"]:
        return ["solver", "retriever_prompts"]
    else:
        return [END]
    

# Graph
llm_graph = llm.with_structured_output(KnowledgeGraph)

graph_prompt = """Create a knowledge graph to help the student understand the concepts you are discussing and have discussed so far.

Do not call the old graph if it exists, always create a new one."""

def create_graph(state: State):
    messages = [SystemMessage(content=graph_prompt)]
    graph = state.get("graph", "")
    if graph:
        graph_message = f"Knowledge graph of your conversation with the student: {str(graph)}\n You can use this graph as a reference but cannot call it!"
        messages += [HumanMessage(content=graph_message)]
    summary = state.get("summary", "")
    if summary:
        summary_message = f"Summary of your conversation with the student: {summary}"
        messages += [HumanMessage(content=summary_message)]
    messages += [message for message in state["messages"] if type(message)==AIMessage]
    try:
        response = llm_graph.invoke(messages)
    except BadRequestError:
        response = state.get("graph") if state.get("graph", "") else KnowledgeGraph()
    except ValidationError:
        response = state.get("graph") if state.get("graph", "") else KnowledgeGraph() 
    return {"graph": response}


workflow = StateGraph(State)
workflow.add_node("safety", safety_checker)
# workflow.add_node("context_check", context_check)
workflow.add_node("solver", simple_solver)
workflow.add_node("retriever_prompts", retriever_prompts)
workflow.add_node("solver_retrieve", solver_retrieve)
workflow.add_node("socratic_retrieve", socratic_retrieve)
workflow.add_node("socratic", socratic)
workflow.add_node("graph_creator", create_graph)
workflow.add_node("summarize", summarize_conversation)
# workflow.add_node("interrupt", give_answer)

workflow.add_edge(START, "safety")
# workflow.add_conditional_edges("safety", safety_router, {"context_check": "context_check", END: END})
workflow.add_conditional_edges("safety", safety_router, [END, "solver", "retriever_prompts"])
workflow.add_edge("retriever_prompts", "solver_retrieve")
workflow.add_edge("retriever_prompts", "socratic_retrieve")
workflow.add_edge("solver", "socratic")
workflow.add_edge("solver_retrieve", "socratic")
workflow.add_edge("socratic_retrieve", "socratic")
# workflow.add_edge("retriever", "socratic")
workflow.add_edge("socratic", "graph_creator")
# workflow.add_edge("solver", "interrupt")
# workflow.add_conditional_edges("interrupt", route_to_answer, {"summarize": "summarize", "socratic": "socratic"})
workflow.add_conditional_edges("graph_creator", should_summarize, {"summarize": "summarize", END: END})
workflow.add_edge("summarize", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

async def stream_graph(message, thread_id):
    config = {"configurable": {"thread_id": str(thread_id)}}
    input_message = HumanMessage(content=message)
    try:
        async for event in graph.astream_events({"messages": input_message}, config, version="v2"):
            if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == "socratic":
                data = event["data"]
                yield data["chunk"].content
    except APIError:
        return
        
def draw_graph(thread_id):
    config = {"configurable": {"thread_id": str(thread_id)}} 
    if knowledge_graph := graph.get_state(config).values.get("graph", ""):
        return knowledge_graph.return_graph()
            
    
        

async def text_stream(query_text, thread_id):
    async for c in stream_graph(query_text, thread_id):
        print(c, end = "", flush=True)
        
if __name__=="__main__":
    asyncio.run(text_stream("what is hashing?", 1))