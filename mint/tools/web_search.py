"""Util that include web search."""
from mint.tools.base import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import ArxivAPIWrapper


class WebSearchRun(Tool):
    """Tool that adds the capability to search using the Google Search API."""

    name = "web_search"
    signature = f"{name}(query: str) -> str"
    description = f"""The Web Search tool uses the Google Search API to search the web for the query and returns the top 10 results."""
    api_wrapper = GoogleSearchAPIWrapper(k=10)


    def __call__(
        self,
        query: str,
    ) -> str:
        output = self.api_wrapper.run(query)
        return output

class ArxivSearchRun(Tool):
    """Tool that adds the capability to search using the Wikipedia API."""

    name = "web_search"
    signature = f"{name}(query: str) -> str"
    description = WEB_SEARCH_DESCRIPTION
    api_wrapper = GoogleSearchAPIWrapper()


    def __call__(
        self,
        query: str,
    ) -> str:
        output = self.api_wrapper.run(query)
        return output


if __name__ == "__main__":
    tool = WebSearchRun()
    print(tool("what is the capital of france"))
