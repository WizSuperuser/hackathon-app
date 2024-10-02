import os
import asyncio
from dotenv import load_dotenv

from groq import BadRequestError
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import vertexai
from graphviz import Digraph

SUMMARIZE_AFTER_MESSAGES = 8
KEEP_FIRST_AND_LAST_MESSAGES = 2

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
    enough_context: bool = False
    safe: bool = True
    graph: KnowledgeGraph

simple_solver_prompt = """You are a helpful tutor in conversation with a student. You're helping with answering / solving a student's question / problem. Your core expertise is data structures and algorithms. The questions asked can be about definitions, help with debugging code or hard leetcode style problems to solve. 

Think carefully before answering any question. Explain your response / reasoning in a concise, lucid manner with all the relevant technical details like code, equations, specifications.

Do not hallucinate. Do not make up facts. If you don't know how to answer a problem, just say so.

Be concise."""

def simple_solver(state: State):
    
    messages = [SystemMessage(content=simple_solver_prompt)]
    # get summary if it exists
    summary = state.get("summary", "")
    # if there is summary, we add it to prompt
    if summary:
        summary_message = f"\n Above is the original question and response. \n Summary of the conversation that followed: {summary} \n Recent conversation: \n"
        messages += state["messages"][:KEEP_FIRST_AND_LAST_MESSAGES] + [HumanMessage(content=summary_message)] + state["messages"][KEEP_FIRST_AND_LAST_MESSAGES:]
    else:
        messages += state["messages"]
    response = llm.invoke(messages)

    return {"context": response.content}


socratic_prompt2 = """You are an empathetic Socratic tutor in conversation to help a student gain a very strong understanding of a concept or solve a problem.

You are helping them with a question/problem and want to help them understand the concepts or figure out solution on their own  with only nudges in the right direction. 

You will be provided with an answer / solution from another tutor but student has not seen it.

First you are going to check whether the question is a factual question and warrants direct answers, for example, like ‘Who built Taj Mahal?’ or if it is a complicated problem / concept that requires further nudging and probing. For simple factual questions, provide direct answers. 

Otherwise, based on the answer / solution, use the socratic method to guide the student towards it. Provide hints or prompt the student to think of the next step.  

You can use the answer / solution to guide the student to the answer in a Socratic manner without directly giving it away. You can also ask the student a question that requires them to apply the concept and enhance their understanding.

Your guidance and reasoning should be lucid and easy to understand. Do not overwhelm the student with a lot of questions at a time. 

Use simple analogies and examples to guide the student. Your core expertise is data structures and algorithms. Where it is suitable, take an algorithm based or a code-first approach to guide the student. Put any math syntax between $ signs.

If the student seems to be really struggling with a concept, provide a larger hint. 

Now you are given the answer / response from another tutor (the student has not seen this), and if available, a summary of your conversation with the student, followed by the most recent dialogue between you and the student.

You need to be a Socratic guide. Be concise and empathetic. Understand when you should give more inputs to the answer and when to probe the student based on your recent dialogue and conversation summary.

Response from another tutor: \n

"""

socratic_prompt = """You are a motivated tutor in conversation to help a student gain a strong understanding of a concept or solve a problem.

You are helping them with a question/problem and want to help them understand the concepts or figure out solution on their own  with only nudges in the right direction.

You will be provided with an answer / solution from another tutor but student has not seen it.

You also have the summarized transcript of the conversation you've had with the student so far.

First you are going to check whether the question is a factual question and warrants direct answers, for example, like ‘Who built Taj Mahal?’ or if it is a complicated problem / concept that requires further nudging and probing. For simple factual questions, provide direct answers.

Otherwise, based on the answer / solution, use the socratic method to guide the student towards it without directly giving it away. Provide hints or prompt the student to think of the next step.

You can also ask the student a question that requires them to apply the concept and enhance their understanding.

Your guidance and reasoning should be lucid and easy to understand. Do not overwhelm the student with a lot of questions at a time.

Use a simple analogy or example to initially frame the problem where appropriate. Your core expertise is data structures and algorithms. Remember to guide the student from the analogy to an algorithm based and a code-first approach. Be precise and explain any relevant equations too if they are central to a concept/problem. Remember to put any math syntax and equations between $ signs for proper formatting for the student.

If the student seems to be really struggling with a concept, provide a larger hint.

Analyze your conversation summary and pay particular attention to what the overall goal of the conversation is and to the final question posed or answer to be checked submitted by the student.

Reply to the student using the Socratic approach and meaningful questions to motivate the answer for them. Be concise and empathetic. Understand when you should give more inputs to the answer and when to probe the student based on your recent dialogue and conversation summary.

Now you are given the answer / response from another tutor (the student has not seen this), followed by the summary of your conversation / dialogue with the student.

Response from another tutor: \n
"""




def socratic(state: State):
    messages = [SystemMessage(socratic_prompt + state["context"])]
    summary = state.get("summary", "")
    if summary:
        summary_message = f"\n Above is the original question and response. \n Summary of the conversation that followed: {summary} \n Recent conversation: \n"
        
        messages += state["messages"][:KEEP_FIRST_AND_LAST_MESSAGES] + [HumanMessage(content=summary_message)] + state["messages"][KEEP_FIRST_AND_LAST_MESSAGES:]
        
    else:
        messages += state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": response} 


def summarize_conversation(state: State):
    summary = state.get("summary", "")
    
    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Be concise and extend the summary by taking into account the new messages above. Make a note of any observations on learner reactions and understanding as well:"
        )
        messages = state["messages"][KEEP_FIRST_AND_LAST_MESSAGES:] + [HumanMessage(content=summary_message)]
    else:
        summary_message = "\nCreate a summary of the conversation above. Make a note of any observations on learner reactions and understanding as well:"
        messages = state["messages"] + [summary_message]
    
    response = llm.invoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][KEEP_FIRST_AND_LAST_MESSAGES:-KEEP_FIRST_AND_LAST_MESSAGES]]

    return {"summary": response.content, "messages": delete_messages}


def should_summarize(state: State):
    """Return whether to summarize depending on length of messages"""
    messages = state["messages"]
    
    if len(messages) > SUMMARIZE_AFTER_MESSAGES:
        return "summarize"
    return END


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
        return "solver"
    else:
        return END    

# Graph     
llm_graph = llm.with_structured_output(KnowledgeGraph)

graph_prompt = """Create a knowledge graph to help the student understand the concepts you are discussing and have discussed so far. Illustrate examples and analogies used in the discussion if it helps with better understanding for the student."""
#Do not call the old graph if it exists, always create a new one.

def create_graph(state: State):
    messages = [SystemMessage(content=graph_prompt)]
    graph = state.get("graph", "")
    if graph:
        graph_message = f"Knowledge graph of your conversation with the student: {str(graph)}\n You can use this graph as a reference if needed but you don't have to stick to the exact template" #but cannot call it!"
        messages += [HumanMessage(content=graph_message)]
    summary = state.get("summary", "")
    if summary:
        summary_message = f"Summary of your conversation with the student: {summary}"
        messages += [HumanMessage(content=summary_message)]
    
    try:
        messages += [message for message in state["messages"][-3:] if type(message)==AIMessage]
    except:
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
workflow.add_node("solver", simple_solver)
workflow.add_node("socratic", socratic)
workflow.add_node("graph_creator", create_graph)
workflow.add_node("summarize", summarize_conversation)
# workflow.add_node("interrupt", give_answer)

workflow.add_edge(START, "safety")
workflow.add_conditional_edges("safety", safety_router, {"solver": "solver", END: END})
workflow.add_edge("solver", "socratic")
workflow.add_edge("socratic", "graph_creator")
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