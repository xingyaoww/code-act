import os
import json
from typing import List, Mapping
from enum import Enum
from collections import deque

from ...base import Task, register_task_iterator, ToolType

class WebPage:
    def __init__(self, name: str, content: str):
        self.name: str = name
        self.content: str = content


class WebBrowser:
    def __init__(self, pages: List[WebPage], default_page: str):
        self.pages: List[WebPage] = pages
        self.page_name_to_page: Mapping[str, WebPage] = {p.name: p for p in pages}
        self.current_page: WebPage = None
        self.page_history = deque()
        self.rendered_content = []
        self.scroll_position = 0
        self.lines_per_view = 5  # Number of lines per page view
        self._open_page(default_page)
        self.page_history.clear()

    def _open_page(self, page_name):
        if page_name in self.page_name_to_page:
            self.page_history.append(self.current_page)
            self.current_page = self.page_name_to_page[page_name]
            assert self.current_page is not None
            self.rendered_content = self.current_page.content.split("\n")
            self.scroll_position = 0
            return f"[Opened page: {page_name}]\n" + self.view()
        else:
            return "[Page not found.]"

    def click_url(self, url):
        url = url.replace("'", "")
        if self.current_page and url in self.page_name_to_page:
            return self._open_page(url)
        return "[URL not found.]"

    def go_to_previous_page(self):
        # Go back to the previous page
        if len(self.page_history) > 0:
            self.current_page = self.page_history.pop()
            self.rendered_content = self.current_page.content.split("\n")
            self.scroll_position = 0
            return f"[Opened page: {self.current_page.name}]\n" + self.view()
        return "[No previous page.]"

    def scroll_down(self):
        # Scroll down the view
        if self.scroll_position + self.lines_per_view < len(self.rendered_content):
            self.scroll_position += self.lines_per_view
            return self.view()
        return "[Reached the bottom of the page.]\n"

    def scroll_up(self):
        # Scroll up the view
        if self.scroll_position > 0:
            self.scroll_position -= self.lines_per_view
            return self.view()
        return "[Reached the top of the page.]\n"

    def view(self):
        # Show the current view of the rendered webpage
        start = self.scroll_position
        end = start + self.lines_per_view
        tot_lines = len(self.rendered_content)
        # Include page name, if available
        page_name = (
            f"[Web Page Name: {self.current_page.name}]\n"
            if self.current_page.name
            else ""
        )
        # return pagination info, if available
        _page_id = (
            start // self.lines_per_view + 1 if tot_lines > self.lines_per_view else 1
        )
        _tot_page = (
            tot_lines // self.lines_per_view + 1
            if tot_lines > self.lines_per_view
            else 1
        )
        pagination = f"[Viewing page {_page_id} of {_tot_page}]\n"
        return (
            "-" * 10
            + "\n"
            + page_name
            + pagination
            + "\n"
            + "\n".join(self.rendered_content[start:end])
            + "\n"
            + "-" * 10
        )

def calculator(expression: str) -> float:
    try:
        return eval(expression)
    except:
        return f"Failed to evaluate expression: {expression}"


# Load pages
with open(os.path.join(os.path.dirname(__file__), "web_pages.jsonl")) as f:
    PAGES = [
        WebPage(**json.loads(line))
        for line in f.read().splitlines()
    ]
with open(os.path.join(os.path.dirname(__file__), "metadata.json")) as f:
    METADATA = json.load(f)

CUR_TASK_NAME = "web_browsing"

def create_tools() -> Mapping[str, ToolType]:
    browser = WebBrowser(pages=PAGES, default_page="/")
    return {
        "click_url": ToolType(
            name="click_url",
            description=(
                "Clicks on a URL. A clickable URL looks like [Clickable '<url_argument>'] in the webpage.\n"
                "Arguments: url (str).\n"
                "Returns the rendered content of the webpage after clicking the URL showing on the current rendered page.\n"
            ),
            function=browser.click_url,
            fn_signature="click_url(url: str) -> str",
        ),
        "go_to_previous_page": ToolType(
            name="go_to_previous_page",
            description=(
                "Goes back to the previous page. It has no arguments.\n"
                "After going back to the previous page, return the rendered content of the webpage."
            ),
            function=browser.go_to_previous_page,
            fn_signature="go_to_previous_page() -> str",
        ),
        "scroll_down": ToolType(
            name="scroll_down",
            description=(
                "Scrolls down the view. It has no arguments.\n"
                "Returns the rendered content of the webpage after scrolling down."
            ),
            function=browser.scroll_down,
            fn_signature="scroll_down() -> str",
        ),
        "scroll_up": ToolType(
            name="scroll_up",
            description=(
                "Scrolls up the view. It has no arguments.\n"
                "Returns the rendered content of the webpage after scrolling up."
            ),
            function=browser.scroll_up,
            fn_signature="scroll_up() -> str",
        ),
        "view": ToolType(
            name="view",
            description=(
                "Return the current view in string format of the rendered webpage. It has no arguments.\n"
                "Returns the rendered content of the webpage.\n"
                "You should call this when you want to see the rendered content of the current webpage."
            ),
            function=browser.view,
            fn_signature="view() -> str",
        ),
        "calculator": ToolType(
            name="calculator",
            description='Evaluates the given expression and returns the result. Accepts a calculation expression as input. For example, "2 + (3 * 4)" will return 14.',
            function=calculator,
            fn_signature="calculator(expression: str) -> float",
        ),
    }

TASKS = []
browser_instances = []
# ==================
# Ask Price - Straight forward tasks
_price_of_items = [
    (item["name"], item["price"])
    for item in METADATA["item_info"]
]
_item_to_price = {item: price for item, price in _price_of_items}
for product, price in _price_of_items:
    # Select one instances per product
    if not "Legendary" in product:
        continue

    TASKS.append(
        Task(
            name=f"{CUR_TASK_NAME}/price_of_{product.lower().replace(' ', '_')}",
            tools=create_tools,
            instruction=(
                f"Find the current price of {product}.\n"
                "Answer in the format of 'xx.xx' (e.g., 12.34).\n"
            ),
            expected_output=price,
            is_single_tool_task=False,
        )
    )

# Ask Email/Phone Number/Department - Straight forward tasks
_person_email = [
    (person["name"], person["email"])
    for person in METADATA["person_info"]
]
_person_department = [
    (person["name"], person["department"])
    for person in METADATA["person_info"]
]
_person_phone_number = [
    (person["name"], person["phone_number"])
    for person in METADATA["person_info"]
]
_person_expertise = [
    (person["name"], person["expertise"])
    for person in METADATA["person_info"]
]
for person, email in _person_email[:4]:
    TASKS.append(
        Task(
            name=f"{CUR_TASK_NAME}/email_of_{person.lower().replace(' ', '_')}",
            tools=create_tools,
            instruction=(
                f"Find the email of {person}.\n"
                "Answer in the format of 'xxx@xxx.xxx'\n"
            ),
            expected_output=email,
            is_single_tool_task=False,
        )
    )
for person, department in _person_department[:4]:
    TASKS.append(
        Task(
            name=f"{CUR_TASK_NAME}/department_of_{person.lower().replace(' ', '_')}",
            tools=create_tools,
            instruction=(
                f"Find the department of {person}.\n"
                "Answer in the format of 'xxx'\n"
            ),
            expected_output=department,
            is_single_tool_task=False,
        )
    )
for person, phone_number in _person_phone_number[:4]:
    TASKS.append(
        Task(
            name=f"{CUR_TASK_NAME}/phone_number_of_{person.lower().replace(' ', '_')}",
            tools=create_tools,
            instruction=(
                f"Find the phone number of {person}.\n"
                "Answer in the same format as displayed in the webpage.\n"
            ),
            expected_output=phone_number,
            is_single_tool_task=False,
        )
    )
for person, expertise in _person_expertise[:4]:
    TASKS.append(
        Task(
            name=f"{CUR_TASK_NAME}/expertise_of_{person.lower().replace(' ', '_')}",
            tools=create_tools,
            instruction=(
                f"Find the expertise of {person}.\n"
                "Answer in the same format as displayed in the webpage (e.g., A/B).\n"
            ),
            expected_output=expertise,
            is_single_tool_task=False,
        )
    )

# Composite Task - Find the total price of selected items
_combinations = [
    ["Legendary Wand", "Enchanted Potion", "Magical Spellbook"],
    ["Mystical Crystal Ball", "Rare Crystal Ball", "Magical Spellbook"],
    ["Ancient Wand", "Mystical Wand", "Enchanted Potion"],
    ["Mystical Crystal Ball", "Rare Crystal Ball", "Magical Spellbook", "Mystical Potion"],
]
for i, cur_combination in enumerate(_combinations):
    TASKS.append(
        Task(
            name=f"{CUR_TASK_NAME}/total_price_of_selected_items_{i}",
            tools=create_tools,
            instruction=(
                f"Find the total price of [{', '.join(cur_combination)}].\n"
                "Answer in the format of 'xx.xx' (e.g., 12.34).\n"
            ),
            expected_output=sum(_item_to_price[item] for item in cur_combination),
            is_single_tool_task=False,
        )
    )
# Composite Task - Find the total price of selected items (after applying discount)
_combinations = [
    ["Legendary Wand", "Enchanted Potion", "Magical Spellbook"],
    ["Mystical Crystal Ball", "Magical Spellbook"],
    ["Ancient Wand", "Enchanted Potion"],
    ["Ancient Crystal Ball", "Rare Wand"]
]
for i, cur_combination in enumerate(_combinations):
    browser_instances.append(WebBrowser(pages=PAGES, default_page="/"))
    TASKS.append(
        Task(
            name=f"{CUR_TASK_NAME}/total_price_of_selected_items_after_discount_{i}",
            tools=create_tools,
            instruction=(
                f"Find the total price of [{', '.join(cur_combination)}] after applying a 10% off discount on the total price.\n"
                "Answer in the format of 'xx.xx' (e.g., 12.34).\n"
            ),
            expected_output=sum(_item_to_price[item] for item in cur_combination) * 0.9,
            is_single_tool_task=False,
        )
    )

# Count person Task
for dept in ["Sales", "Customer Support"]:
    TASKS.append(
        Task(
            name=f"{CUR_TASK_NAME}/people_in_{dept.lower().replace(' ', '_')}",
            tools=create_tools,
            instruction=(
                f"How many people are in the {dept} Department?\n"
                "Answer in the format of 'xx' (e.g., 9).\n"
            ),
            expected_output=sum(
                1 for person in METADATA["person_info"] if person["department"] == dept
            ),
            is_single_tool_task=False,
        )
    )

# Match phone number to person Task
for person, phone_number in _person_phone_number[-4:]:
    browser_instances.append(WebBrowser(pages=PAGES, default_page="/"))
    TASKS.append(
        Task(
            name=f"{CUR_TASK_NAME}/match_phone_number_to_person_{person.lower().replace(' ', '_')}",
            tools=create_tools,
            instruction=(
                f"Which person's phone number is {phone_number}?\n"
                "Answer the person's name (e.g., John Smith).\n"
            ),
            expected_output=person,
            is_single_tool_task=False,
        )
    )


register_task_iterator(TASKS, len(TASKS))
print(f"**** {len(TASKS)} tasks registered for {CUR_TASK_NAME} ****")

# br = WebBrowser(pages=PAGES, default_page="/")
# import pdb; pdb.set_trace()
