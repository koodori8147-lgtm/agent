import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

sys.stdout.reconfigure(encoding="utf-8")  # Windows(cp949) 콘솔에서 이모지 출력 시 UnicodeEncodeError 방지
load_dotenv()

tool = TavilySearch(max_results=3)
tools = [tool]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
graph_builder.add_node("chatbot", chatbot)

import json
from langchain.messages import ToolMessage

class BasicToolNode:
    """
    마지막 AIMessage에서 요청된 도구를 실행하는 노트
    """
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("ERROR: 입력에 메시지가 없습니다.")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
tool_node = BasicToolNode(tools = [tool])
graph_builder.add_node("tools", tool_node)

def route_tools(
        state: State,
):
    """
    마지막 메시지에 도구 호출이 있는 경우, ToolNode로 라우팅하고 그렇지 않으면 END로 라우팅
    """
    if isinstance(state,list):
        ai_message = state[-1]
    elif messages := state.get("messages",[]):
        ai_message = messages[-1]
    else:
        raise ValueError(f"ERROR: 입력에 메시지가 없습니다. 상태: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) >0:
        return "tools"
    return END

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)

graph_builder.add_edge("tools","chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

async def ainvoke():
    response = await graph.ainvoke(
        {
            "messages": ["도쿄의 날씨는 어떤가요?"]
        }
    )

    for msg in response["messages"]:
        msg.pretty_print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(ainvoke())
