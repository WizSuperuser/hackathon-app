import os
import asyncio
from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import vertexai
from graphviz import Digraph

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

class State(MessagesState):
    summary: str
    context: str
    enough_context: bool = False
    safe: bool = True
    graph: Digraph

simple_solver_prompt = """You are helping with solving a student's questions about data structures and algorithms.

These can be questions about definitions, help with debugging code or hard leetcode style problems to solve.

Think carefully before answering any question. Explain your reasoning.

Do not hallucinate. Do not make up facts. If you don't know how to answer a problem, just say so.

Be concise."""

def simple_solver(state: State):
    
    messages = [SystemMessage(content=simple_solver_prompt)]
    
    # get summary if it exists
    summary = state.get("summary", "")
    
    # if there is summary, we add it to prompt
    if summary:
        
        # add summary to system message
        summary_message = f"Summary of conversation earlier: {summary}"
        
        # append summary to any newer message
        messages += [HumanMessage(content=summary_message)]
    
    messages += state["messages"]
    
    response = llm.invoke(messages)
    # NEED TO PREVENT CONTEXT FROM BALLOONING if I change it to list and want to persist that
    return {"context": response.content}


socratic_prompt = """You are a tutor trying to help a student gain a very strong understanding of a concept/problem. 

You are helping them with a problem and want to help them understand the concepts by figuring out the solution themselves with only nudges in the right direction.

You have the solution above but the student has never seen it.

If the student wants to learn about a new concept: use the solution to provide the necessary context. Then, based on that ask the student a question that requires them to apply the concept in code to help enhance their understanding.

If the question is a problem to solve: based on the solution to the question, use the socratic method to guide the student towards the answer.

Provide hints or prompt the student to think of the next step. If the student seems to be really stuggling with a concept, provide a larger hint. Always take a code-first approach when explaining, giving examples, or solving a problem."""

def socratic(state: State):
    messages = [SystemMessage(content=state["context"] + socratic_prompt)]
    summary = state.get("summary", "")
    if summary:
        summary_message = f"Summary of your conversation with the student: {summary}"
        messages += [HumanMessage(content=summary_message)]
    messages += state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": response}


def summarize_conversation(state: State):
    summary = state.get("summary", "")
    
    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Be concise and extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

def should_summarize(state: State):
    """Return whether to summarize depending on length of messages"""
    messages = state["messages"]
    
    if len(messages) > 6:
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
def retriever(state: State):
    additional_context = notes_retriever.invoke(state['context'])
    return {'context': state['context'] + '-----------------------'.join(x.page_content for x in additional_context)}

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
        return "context_check"
    else:
        return END
    
# Context Checker
class CheckContext(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Is the context enough to provide a response to the student's query? 'yes' or 'no'"
    )

llm_router = llm.with_structured_output(CheckContext)

system_router = """
You are a reasoning agent checking if the provided context is enough to answer a student's
query. The query can be a question: if so you must check if the context is enough to
answer the question. The query can also be a student's attempt at answering or taking the
next step in answering a question: if so, you must check if the context is enough to
check the student's response for correctness and be able to guide them towards the right
path. Give a binary score 'yes' or 'no' to indicate whether the context is enough 
for the task. If responding to either type of query requires more information or checking new code
not present in the context, score 'no'.
"""

prompt_router = ChatPromptTemplate.from_messages(
    [
        ('system', system_router),
        ('human', "Context: \n\n {context} \n\n Student query: {query}"),
    ]
)

router = prompt_router | llm_router

def context_check(state: State):
    if state.get("context", ""):
        return {"enough_context": 'yes'==router.invoke({'context': state["context"], 'query': state['messages'][-1]})}
    else:
        return {"enough_context": False}

def context_router(state: State):
    if state['enough_context']:
        return "socratic"
    else:
        return "solver"
    
    
# Graph 
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
    nodes: list[Node] = Field(..., default_factory=list)
    edges: list[Edge] = Field(..., default_factory=list)
    
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
    
    
llm_graph = llm.with_structured_output(KnowledgeGraph)

graph_prompt = """Create a knowledge to help the student understand the concepts you are discussing and have discussed so far.

Below you have the knowledge graph of the conversation so far. 
Extend it to include the most recent conversation topics.

Below that you have the conversation with the student. 
Use the recent messages to extend the graph based on the topics discussed and the student's understanding.
Do not extend the graph if there is no change: in this case just return the old graph.

Otherwise: output the complete graph with extensions."""

def create_graph(state: State):
    messages = [SystemMessage(content=graph_prompt)]
    graph = state.get("graph", "")
    if graph:
        graph_message = f"Knowledge graph of your conversation with the student: {str(graph)}"
        messages += [HumanMessage(content=graph_message)]
    summary = state.get("summary", "")
    if summary:
        summary_message = f"Summary of your conversation with the student: {summary}"
        messages += [HumanMessage(content=summary_message)]
    messages += state["messages"]
    
    response = llm_graph.invoke(messages)
    return {"graph": response}


workflow = StateGraph(State)
workflow.add_node("safety", safety_checker)
workflow.add_node("context_check", context_check)
workflow.add_node("solver", simple_solver)
workflow.add_node("retriever", retriever)
workflow.add_node("socratic", socratic)
workflow.add_node("graph_creator", create_graph)
workflow.add_node("summarize", summarize_conversation)
# workflow.add_node("interrupt", give_answer)

workflow.add_edge(START, "safety")
workflow.add_conditional_edges("safety", safety_router, {"context_check": "context_check", END: END})
workflow.add_conditional_edges("context_check", context_router, {"socratic": "socratic", "solver": "solver"})
workflow.add_edge("solver", "retriever")
workflow.add_edge("retriever", "socratic")
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
    async for event in graph.astream_events({"messages": input_message}, config, version="v2"):
        if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == "socratic":
            data = event["data"]
            yield data["chunk"].content
        
def draw_graph(thread_id):
    config = {"configurable": {"thread_id": str(thread_id)}} 
    if knowledge_graph := graph.get_state(config).values.get("graph", ""):
        return knowledge_graph.return_graph()

async def text_stream(query_text, thread_id):
    async for c in stream_graph(query_text, thread_id):
        print(c, end = "", flush=True)
        
if __name__=="__main__":
    asyncio.run(text_stream("what is hashing?", 1))