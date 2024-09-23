import os
import asyncio
from typing import Literal, AsyncGenerator
import uuid

from pydantic import BaseModel, Field
import streamlit as st

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
        
        st.rerun()

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
        st.session_state.chat_history.append(ChatMessage(type="ai", content=streaming_content))



if __name__ == "__main__":
    asyncio.run(main())