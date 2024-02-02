"""Util that include web search."""
from mint.tools.base import Tool
from langchain.utilities import ArxivAPIWrapper


class ArxivSearchRun(Tool):
    """Tool that adds the capability to search using the Wikipedia API."""

    name = "arxiv_search"
    signature = f"{name}(query: str) -> str"
    description = """The Arxiv Search tool provides access to a vast collection of academic papers covering a wide range of topics."""
    api_wrapper = ArxivAPIWrapper()


    def __call__(
        self,
        query: str,
    ) -> str:
        output = self.api_wrapper.run(query)
        return output


if __name__ == "__main__":
    tool = ArxivSearchRun()
    print(tool("machine learning"))
