"""Util that calls Wikipedia."""
from typing import Any, Dict, List, Optional
from mint.tools.base import Tool
from langchain.utilities import WikipediaAPIWrapper

WIKIPEDIA_MAX_QUERY_LENGTH = 300


WIKIPEDIA_DESCRIPTION = """The Wikipedia Search tool provides access to a vast collection of articles covering a wide range of topics.
Can query specific keywords or topics to retrieve accurate and comprehensive information.
"""


class WikipediaQueryRun(Tool):
    """Tool that adds the capability to search using the Wikipedia API."""

    name = "wikipedia_search"
    signature = f"{name}(query: str) -> str"
    description = WIKIPEDIA_DESCRIPTION
    api_wrapper = WikipediaAPIWrapper()

    def __call__(
        self,
        query: str,
    ) -> str:
        """Use the Wikipedia tool."""
        output = self.api_wrapper.run(query)
        return output
