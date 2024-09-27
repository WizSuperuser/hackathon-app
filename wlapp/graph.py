import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
import vertexai

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


workflow = StateGraph(State)
workflow.add_node("solver", simple_solver)
workflow.add_node("socratic", socratic)
workflow.add_node("summarize", summarize_conversation)


workflow.add_edge(START, "solver")
workflow.add_edge("solver", "socratic")
workflow.add_conditional_edges("socratic", should_summarize, {"summarize": "summarize", END: END})
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
        


async def text_stream(query_text, thread_id):
    async for c in stream_graph(query_text, thread_id):
        print(c, end = "", flush=True)
        
if __name__=="__main__":
    asyncio.run(text_stream("what is hashing?", 1))