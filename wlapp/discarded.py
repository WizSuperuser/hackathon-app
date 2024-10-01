# from langchain_community.document_loaders import WikipediaLoader


# # Search
# class SearchQuery(BaseModel):
#     search_query: str = Field(None, description="Search query for retrieval.")
    
# # Web Search
# search_instructions = SystemMessage(content=f"""You will be given a conversation between a student and their teacher.

# Your goal is to generate a query for use in wikipedia search related to the conversation.
        
# First, analyze the full conversation.

# Pay particular attention to the final question posed or answer to be checked submitted by the student.

# Convert this final message into a query that can be looked up on wikipedia.""")

# def search_web(state: State):
    
#     """ Retrieve docs from web search """

#     # Search query
#     search_llm = llm.with_structured_output(SearchQuery)
#     messages = []
#     summary = state.get("summary", "")
#     if summary:
#         summary_message = f"Summary of conversation earlier: {summary}"
#         messages = [HumanMessage(content=summary_message)]
#     messages += state["messages"] 
#     search_query = search_llm.invoke([search_instructions]+messages)
    
#     # Search
#     search_docs = tavily_search.invoke(search_query.search_query)

#      # Format
#     formatted_search_docs = "\n\n---\n\n".join(
#         [
#             f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
#             for doc in search_docs
#         ]
#     )

#     return {"web_search": formatted_search_docs} 

# def search_wikipedia(state: State):
    
#     """ Retrieve docs from wikipedia """

#     # Search query
#     search_llm = llm.with_structured_output(SearchQuery)
#     messages = []
#     summary = state.get("summary", "")
#     if summary:
#         summary_message = f"Summary of conversation earlier: {summary}"
#         messages = [HumanMessage(content=summary_message)]
#     messages += state["messages"] 
#     search_query = search_llm.invoke([search_instructions]+messages)
    
#     # Search
#     search_docs = WikipediaLoader(query=search_query.search_query, 
#                                   load_max_docs=1).load()

#      # Format
#     formatted_search_docs = "\n\n---\n\n".join(
#         [
#             f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
#             for doc in search_docs
#         ]
#     )

#     return {"wiki_search": formatted_search_docs} 


# class SocraticResponse(BaseModel):
#     solution: str = Field(None, description="An overview of the student's query and the most important insights of the solution.")
#     thoughts: str = Field(None, description="Based on the student's query and the solution, figure out the best way to respond to the student. Decide what context to reveal and not reveal. What is most important for the student to figure out for themselves for their understanding. How I can help them come to this understanding.")
#     reply: str = Field(None, description="The final response to display to the student.")





# Socratic Prompt
# First, analyze the full conversation and understand what the student is interested in and their progress so far.

# Pay particular attention to the final question posed or answer to be checked submitted by the student.

# Then, analyze the solution and figure out the best way to guide the student towards the answer. For definition questions, this might involve using analogies to help motivate the concept before moving to an implementation in code. For problems to solve or code to debug or proofs, this might involve explaining the problem and hinting at possible steps to try.

# If the student wants to learn about a new concept: use the solution to provide the necessary context. Then, based on that ask the student a question that requires them to apply the concept in code to help enhance their understanding.

# If the question is a problem to solve or code to debug: based on the solution to the question, use the socratic method to guide the student towards the answer.

# Provide hints or prompt the student to think of the next step. If the student seems to be really stuggling with a concept, provide a larger hint. Always take a code-first approach when explaining, giving examples, or solving a problem."""




# # Context Checker
# class CheckContext(BaseModel):
#     binary_score: Literal["yes", "no"] = Field(
#         description="Is the context enough to provide a response to the student's query? 'yes' or 'no'"
#     )

# llm_router = llm.with_structured_output(CheckContext)

# system_router = """
# You are a reasoning agent checking if the provided context is enough to answer a student's
# query. The query can be a question: if so you must check if the context is enough to
# answer the question. The query can also be a student's attempt at answering or taking the
# next step in answering a question: if so, you must check if the context is enough to
# check the student's response for correctness and be able to guide them towards the right
# path. Give a binary score 'yes' or 'no' to indicate whether the context is enough 
# for the task. If responding to either type of query requires more information or checking new code
# not present in the context, score 'no'.
# """

# prompt_router = ChatPromptTemplate.from_messages(
#     [
#         ('system', system_router),
#         ('human', "Context: \n\n {context} \n\n Student query: {query}"),
#     ]
# )

# router = prompt_router | llm_router

# def context_check(state: State):
#     if state.get("context", ""):
#         return {"enough_context": 'yes'==router.invoke({'context': state["context"] + state["web_search"] + state["wiki_search"], 'query': state['messages'][-1]})}
#     else:
#         return {"enough_context": False}

# def context_router(state: State):
#     if state['enough_context']:
#         return ["socratic"]
#     else:
#         return ["solver", "web", "wiki"]