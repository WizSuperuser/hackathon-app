import os
import asyncio
from typing import Literal, AsyncGenerator
import uuid

from pydantic import BaseModel, Field
import streamlit as st
import streamlit.components.v1 as components

from agent import stream_openai
from graph import stream_graph

IS_USING_IMAGE_RUNTIME = bool(os.environ.get("IS_USING_IMAGE_RUNTIME", False))
APP_TITLE = "WizlearnrAI"


class ChatMessage(BaseModel):
    """Message in a chat"""
    
    type: Literal["human", "ai"] = Field(description="Role of a message")
    content: str = Field(description="Content of a message")
    
async def main():
    st.set_page_config(page_title=APP_TITLE)
    with st.sidebar:
        st.header(f"{APP_TITLE}")
        "Learn Data Structures and Algorithms using the Socratic method."
        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved for product evaluation and improvement purposes only."
            )
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4()
    
    await asyncio.sleep(5)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    chat_history = st.session_state.chat_history
    
    if len(chat_history) == 0:
        WELCOME = "Hello! I'm an AI-powered learning assistant. I may take a few seconds to boot up when you send your first message. Let's learn some data structures and algorithms!"
        with st.chat_message("ai"):
            st.write(WELCOME)
            
    async def chat_history_iter():
        for message in chat_history:
            yield message
            
    await draw_history(chat_history_iter())
    
    if query_text := st.chat_input("Ask a question!"):
        chat_history.append(ChatMessage(type="human", content=query_text))
        st.chat_message("human").write(query_text)
        # draw ai response to screen
        await draw_message(stream_graph(query_text, st.session_state.thread_id))
        
    if st.session_state.chat_history:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reload"):
                st.session_state.chat_history.pop()
                alternate_response = f"Please respond using an alternate approach to the same question. "
                await draw_message(stream_graph(alternate_response+st.session_state.chat_history[-1].content, st.session_state.thread_id))
        with col2:
            if st.button("Copy Last Response"):
                last_ai_message = next((msg.content for msg in reversed(st.session_state.chat_history) if msg.type == "ai"), None)
                if last_ai_message:
                    st.write("Please use the copy button on the top right of the box below to copy the response!")
                    st.code(last_ai_message, language="text")
                else:
                    st.write("No AI response to copy.")
            
    
        # st.rerun()

async def draw_history(history: AsyncGenerator):
    async for message in history:
        match message.type:
            case "human":
                st.chat_message("human").write(message.content)
            case "ai":
                st.chat_message("ai").write(message.content)
    

async def draw_message(message: AsyncGenerator):
    streaming_content = ""
    with st.chat_message("ai"):
        streaming_placeholder = st.empty()
        async for msg in message:
            streaming_content += msg
            streaming_placeholder.markdown(streaming_content)
        
        if not streaming_content:
            safety_message = "Sorry, I can't answer that."
            streaming_content = safety_message
            streaming_placeholder.markdown(safety_message)
            
        st.session_state.chat_history.append(ChatMessage(type="ai", content=streaming_content))



if __name__ == "__main__":
    asyncio.run(main())