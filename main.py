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
    st.session_state["new_prompt"] = loaded_prompt.template  # LLM 호출 없이 그대로 저장

    #loaded_prompt = load_prompt("prompts/empathetic_expert.yaml", encoding="utf-8")
    #final_template = loaded_prompt.template

    #prompt1 = PromptTemplate.from_template(template=final_template)
    #llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
    #chain1 = prompt1 | llm | StrOutputParser()
    #st.session_state["new_prompt"] = chain1.invoke("")

    tool1 = retriever_tool()  # pdf_search
    tool2 = GoogleSearch(max_results=3)

    st.session_state["react_agent"] = create_agent_executor(
        model_name="gpt-4o-mini", tools=[tool1, tool2]
    )

# 이전 대화 기록 출력
print_messages()


# 사용자 입력 처리
# user_input = st.chat_input("궁금한 내용을 물어보세요!")
# warning_msg = st.empty()

# if user_input:
#     agent = st.session_state["react_agent"]
#     if agent is not None:
#         config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

#         # 사용자 메시지 출력
#         st.chat_message("user").write(user_input)

#         # container는 만들되 사용은 안 함 (stream_handler 내부 출력 막기 위해 따로 출력)
#         container = st.empty()

#         # 챗봇 응답 처리 (container는 전달하되, container에 직접 쓰지 않음)
#         container_messages, tool_args, agent_answer = stream_handler(
#             container,
#             agent,
#             {"messages": [("human", user_input)]},
#             config,
#         )

#         # 도구 응답 저장
#         for tool_arg in tool_args:
#             add_message(
#                 "assistant",
#                 tool_arg["tool_result"],
#                 "tool_result",
#                 tool_arg["tool_name"],
#             )

#         # 최종 응답 저장
#         add_message("assistant", agent_answer)

#         # 직접 markdown으로 챗봇 응답 출력 (아바타 포함)
#         st.markdown(f"""
#         <div style='display: flex; align-items: flex-start; margin-top: 10px;'>
#             <img src='https://i.namu.wiki/i/nTpvyrZYPoJBnrydRk9_5WAUX6kz1B8Wu6IvFIrLnxwoaV9BD-fP23SGhHp3wjls59AftaAIAa1xWWGCaruCog.webp'
#                  width='50' style='margin-right: 10px; border-radius: 50%;'>
#             <div style='background-color: #f0f2f6; color: black; padding: 12px 18px; border-radius: 15px; max-width: 85%; font-size: 16px; line-height: 1.5;'>
#                 {agent_answer}
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

#     else:
#         warning_msg.warning("개인정보 입력을 완료해주세요.")



# 사용자 입력 처리
user_input = st.chat_input("궁금한 내용을 물어보세요!")
warning_msg = st.empty()

# if user_input:
#     agent = st.session_state["react_agent"]
#     if agent is not None:
#         config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
#         st.chat_message("user").write(user_input)
#         with st.chat_message("assistant"):
#             container = st.empty()
#             container_messages, tool_args, agent_answer = stream_handler(
#                 container,
#                 agent,
#                 {"messages": [("human", user_input)]},
#                 config,
#             )
#             add_message("user", user_input)
#             for tool_arg in tool_args:
#                 add_message(
#                     "assistant",
#                     tool_arg["tool_result"],
#                     "tool_result",
#                     tool_arg["tool_name"],
#                 )
#             add_message("assistant", agent_answer)
#     else:
#         warning_msg.warning("개인정보 입력을 완료해주세요.")

if user_input:
    agent = st.session_state["react_agent"]
    if agent is not None:
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

        # 사용자 메시지 출력
        st.chat_message("user").write(user_input)

        # 챗봇 응답 (스트리밍 + 이미지 함께 출력)
        with st.chat_message("assistant"):
            col1, col2 = st.columns([1, 9])

            with col1:
                st.image(
                    "https://github.com/HyeyoonKim0711/the_empathetic_expert/blob/main/fig/empathetic_expert.png?raw=true",
                    width=50,
                )

            with col2:
                container = st.empty()  # 여기로 스트리밍 응답이 실시간 출력됨

        # 실제 응답 처리 (stream_handler가 container에 streaming 출력)
        container_messages, tool_args, agent_answer = stream_handler(
            container,
            agent,
            {"messages": [("human", user_input)]},
            config,
        )

        # 메시지 기록 저장
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

