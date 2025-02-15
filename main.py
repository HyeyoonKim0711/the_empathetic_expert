import streamlit as st
from rag_bk.bk_logging import langsmith
from dotenv import load_dotenv
from rag_bk.modules.handler import stream_handler
from rag_bk.st_function import print_messages, add_message
from rag_bk.bk_messages import random_uuid
from langchain_core.prompts import load_prompt, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from rag_bk.modules.google import GoogleSearch
from rag_bk.modules.tools import WebSearchTool, retriever_tool
from rag_bk.modules.agent import create_agent_executor

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름
langsmith("챗봇상담")

st.title("The Empathetic Expert 💬")
clear_btn = st.button("대화 초기화")

# 고유 스레드 ID(랜덤으로 지어주기 -> 대화 기억용도 -> 대화내용 초기화하면 이것도 초기화)


if clear_btn:
    st.session_state["messages"] = []  # 대화 정보 지우기
    st.session_state["thread_id"] = random_uuid()  # 사용자정보 기억 지우기

# 대화기록을 저장하기 위한 용도로 생성
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ReAct Agent 초기화
if "react_agent" not in st.session_state:
    st.session_state["thread_id"] = random_uuid()

    loaded_prompt = load_prompt("prompts/empathetic_expert.yaml", encoding="utf-8")
    final_template = loaded_prompt.template

    prompt1 = PromptTemplate.from_template(template=final_template)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
    chain1 = prompt1 | llm | StrOutputParser()
    st.session_state["new_prompt"] = chain1.invoke("")

    tool1 = retriever_tool()  # pdf_search
    tool2 = GoogleSearch(max_results=3)

    st.session_state["react_agent"] = create_agent_executor(
        model_name="gpt-4o", tools=[tool1, tool2]
    )

# 이전 대화 기록 출력
print_messages()

# 사용자 입력 처리
user_input = st.chat_input("궁금한 내용을 물어보세요!")
warning_msg = st.empty()

if user_input:
    agent = st.session_state["react_agent"]
    if agent is not None:
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            container = st.empty()
            container_messages, tool_args, agent_answer = stream_handler(
                container,
                agent,
                {"messages": [("human", user_input)]},
                config,
            )
            add_message("user", user_input)
            for tool_arg in tool_args:
                add_message(
                    "assistant",
                    tool_arg["tool_result"],
                    "tool_result",
                    tool_arg["tool_name"],
                )
            add_message("assistant", agent_answer)
    else:
        warning_msg.warning("개인정보 입력을 완료해주세요.")
