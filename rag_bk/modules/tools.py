from typing import Any, List
from rag_bk.modules.google import GoogleSearch
from langchain_core.tools import Tool
from .base import BaseTool
from rag_bk.modules.retrieval import retriever
from langchain_core.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool


# class WebSearchTool(BaseTool[TavilySearch]):
#     """웹 검색을 수행하는 도구 클래스"""

#     def __init__(
#         self,
#         topic: str = "general",
#         max_results: int = 3,
#         include_answer: bool = False,
#         include_raw_content: bool = False,
#         include_images: bool = False,
#         format_output: bool = False,
#         include_domains: List[str] = [],
#         exclude_domains: List[str] = [],
#     ):
#         """WebSearchTool 초기화 메서드"""
#         super().__init__()
#         self.topic = topic
#         self.max_results = max_results
#         self.include_answer = include_answer
#         self.include_raw_content = include_raw_content
#         self.include_images = include_images
#         self.format_output = format_output
#         self.include_domains = include_domains
#         self.exclude_domains = exclude_domains

#     def _create_tool(self) -> TavilySearch:
#         """TavilySearch 객체를 생성하고 설정하는 내부 메서드"""
#         search = TavilySearch(
#             topic=self.topic,
#             max_results=self.max_results,
#             include_answer=self.include_answer,
#             include_raw_content=self.include_raw_content,
#             include_images=self.include_images,
#             format_output=self.format_output,
#             include_domains=self.include_domains,
#             exclude_domains=self.exclude_domains,
#         )
#         search.name = "web_search"
#         search.description = (
#             "if pdf_search tool don't use, Use this tool to search on the web"
#         )
#         return search

#     def __call__(self, *args: Any, **kwargs: Any) -> Any:
#         """도구를 실행하는 메서드"""
#         tool = self._create_tool()
#         return tool(*args, **kwargs)


class WebSearchTool:
    """웹 검색을 수행하는 도구 클래스"""

    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self.search_client = GoogleSearch(max_results=self.max_results)

    def search(self, query: str) -> Any:
        """Google 검색 수행"""
        return self.search_client.search(query)

    def create(self):
        """LangChain Tool 형식으로 변환"""
        return Tool(
            name="web_search",
            func=self.search,
            description="Use this tool to search on the web if the PDF search tool is not used."
        )




def retriever_tool():
    retriever_tool = create_retriever_tool(
        retriever(vectorstore_path="data/cache/embeddings/faiss_index.index"),
        name="pdf_search",  # 도구의 이름을 입력
        description="""use this tool to search information from the PDF document.
        if document include information of query. you must use this tool to make an answer.""",  # 도구에 대한 설명을 자세히 기입!
    )
    return retriever_tool




