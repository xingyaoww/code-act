from ..base import Task, register_task_iterator, ToolType
from typing import List, Dict

# Fake data tables for flights and hotels
# LOC = set([flight["to_location"] for flight in flights] + [hotel["location"] for hotel in hotels])
LOC = [
    '"A"',
    '"B"',
    '"C"',
    '"D"',
    '"E"',
    '"F"',
]

flights = [
    {
        "from_location": LOC[4],
        "to_location": LOC[0],
        "date": "2023-12-25",
        "price": 450,
    },
    {
        "from_location": LOC[4],
        "to_location": LOC[1],
        "date": "2023-11-10",
        "price": 350,
    },
    {
        "from_location": LOC[4],
        "to_location": LOC[1],
        "date": "2023-11-10",
        "price": 165,
    },
    {
        "from_location": LOC[4],
        "to_location": LOC[1],
        "date": "2023-12-15",
        "price": 360,
    },
    {
        "from_location": LOC[1],
        "to_location": LOC[4],
        "date": "2023-12-08",
        "price": 470,
    },
    {
        "from_location": LOC[1],
        "to_location": LOC[4],
        "date": "2023-12-08",
        "price": 946,
    },
    {
        "from_location": LOC[4],
        "to_location": LOC[2],
        "date": "2023-10-05",
        "price": 600,
    },
    {
        "from_location": LOC[4],
        "to_location": LOC[2],
        "date": "2023-12-01",
        "price": 580,
    },
    {
        "from_location": LOC[4],
        "to_location": LOC[3],
        "date": "2023-08-15",
        "price": 400,
    },
    # Add more flights as needed
    {
        "from_location": LOC[1],
        "to_location": LOC[2],
        "date": "2023-11-13",
        "price": 300,
    },
    {
        "from_location": LOC[0],
        "to_location": LOC[2],
        "date": "2023-08-20",
        "price": 250,
    },
    {
        "from_location": LOC[0],
        "to_location": LOC[3],
        "date": "2023-12-28",
        "price": 250,
    },
    {
        "from_location": LOC[0],
        "to_location": LOC[3],
        "date": "2023-12-29",
        "price": 410,
    },
    {
        "from_location": LOC[0],
        "to_location": LOC[1],
        "date": "2023-12-29",
        "price": 460,
    },
    {
        "from_location": LOC[3],
        "to_location": LOC[2],
        "date": "2024-01-02",
        "price": 380,
    },
    {
        "from_location": LOC[3],
        "to_location": LOC[0],
        "date": "2023-08-18",
        "price": 490,
    },
    {
        "from_location": LOC[3],
        "to_location": LOC[0],
        "date": "2023-08-18",
        "price": 560,
    },
    {
        "from_location": LOC[2],
        "to_location": LOC[1],
        "date": "2024-01-05",
        "price": 300,
    },
    {
        "from_location": LOC[2],
        "to_location": LOC[1],
        "date": "2023-12-03",
        "price": 900,
    },
    {
        "from_location": LOC[2],
        "to_location": LOC[1],
        "date": "2023-12-03",
        "price": 450,
    },
    {
        "from_location": LOC[2],
        "to_location": LOC[3],
        "date": "2023-12-03",
        "price": 450,
    },
    {
        "from_location": LOC[1],
        "to_location": LOC[3],
        "date": "2023-12-06",
        "price": 250,
    },
    {
        "from_location": LOC[1],
        "to_location": LOC[3],
        "date": "2023-12-06",
        "price": 700,
    },
    {
        "from_location": LOC[3],
        "to_location": LOC[1],
        "date": "2023-12-06",
        "price": 490,
    },
    {
        "from_location": LOC[3],
        "to_location": LOC[1],
        "date": "2023-12-06",
        "price": 746,
    },
    {
        "from_location": LOC[1],
        "to_location": LOC[3],
        "date": "2024-01-01",
        "price": 490,
    },
    {
        "from_location": LOC[1],
        "to_location": LOC[3],
        "date": "2024-01-01",
        "price": 546,
    },
    {
        "from_location": LOC[1],
        "to_location": LOC[4],
        "date": "2024-01-07",
        "price": 320,
    },
    {
        "from_location": LOC[4],
        "to_location": LOC[0],
        "date": "2024-01-10",
        "price": 360,
    },
]

hotels = [
    {
        "location": LOC[0],
        "preferences": ["wifi", "pool"],
        "price_per_night": 120,
        "rating": 4,
    },
    {
        "location": LOC[0],
        "preferences": ["wifi", "pool"],
        "price_per_night": 50,
        "rating": 3,
    },
    {
        "location": LOC[1],
        "preferences": ["wifi", "gym"],
        "price_per_night": 150,
        "rating": 4,
    },
    {
        "location": LOC[1],
        "preferences": ["pool", "gym", "wifi"],
        "price_per_night": 160,
        "rating": 5,
    },
    {"location": LOC[2], "preferences": ["pool"], "price_per_night": 100, "rating": 3},
    {"location": LOC[2], "preferences": ["wifi"], "price_per_night": 95, "rating": 4},
    {"location": LOC[2], "preferences": ["wifi", "gym"], "price_per_night": 103, "rating": 4},
    {
        "location": LOC[2],
        "preferences": ["wifi", "pool"],
        "price_per_night": 110,
        "rating": 5,
    },
    {"location": LOC[3], "preferences": ["wifi"], "price_per_night": 130, "rating": 4},
    {
        "location": LOC[3],
        "preferences": ["wifi", "gym"],
        "price_per_night": 140,
        "rating": 4,
    },
    {
        "location": LOC[3],
        "preferences": ["wifi", "gym", "pool"],
        "price_per_night": 135,
        "rating": 5,
    },
    {
        "location": LOC[4],
        "preferences": ["wifi", "gym"],
        "price_per_night": 190,
        "rating": 4,
    },
    {
        "location": LOC[4],
        "preferences": ["wifi", "gym", "pool"],
        "price_per_night": 120,
        "rating": 5,
    },
    
]

SUPPORTED_LOC = LOC.copy()
SUPPORTED_LOC += [loc.strip('"') for loc in SUPPORTED_LOC]
SUPPORTED_LOC = set(SUPPORTED_LOC)


# Tool Definitions
def find_flights(from_location: str, to_location: str, date: str) -> List[Dict]:
    from_location = from_location.strip('"')
    to_location = to_location.strip('"')
    if from_location not in SUPPORTED_LOC:
        raise ValueError(
            f"Origin [{from_location}] is not supported. Supported from_locations: {LOC}"
        )

    if to_location not in SUPPORTED_LOC:
        raise ValueError(
            f"Destination [{to_location}] is not supported. Supported to_locations: {LOC}"
        )

    return [
        flight
        for flight in flights
        if flight["to_location"].strip('"') == to_location
        and flight["date"] == date
        and flight["from_location"].strip('"') == from_location
    ]


def book_hotel(location: str, *preferences: str) -> List[Dict]:
    location = location.strip('"')
    if location not in SUPPORTED_LOC:
        raise ValueError(
            f"Location [{location}] is not supported. Supported locations: {LOC}"
        )

    suitable_hotels = [
        hotel
        for hotel in hotels
        if hotel["location"].strip('"') == location
        and all(pref in hotel["preferences"] for pref in preferences)
    ]
    return suitable_hotels


def budget_calculator(
    flight_price: float, hotel_price_per_night: float, num_nights: int
) -> float:
    return flight_price + hotel_price_per_night * num_nights


# Registering the tools for Travel Itinerary Planning
CUR_TASK_NAME = "travel_itinerary_planning"
CUR_TOOLS = {
    "find_flights": ToolType(
        name="find_flights",
        description=(
            "Finds flights based on source, destination and date. Arguments: from_location (str), to_location (str), date (str) in YYYY-MM-DD format.\n"
            'Returns a list of flights, each represented as a dictionary with keys "from_location", "to_location" (destination), "date", and "price".\n'
            'Example: [{"from_location": "A", "to_location": "B", "date": "2023-12-25", "price": 450}]'
        ),
        function=find_flights,
        fn_signature="find_flights(destination: str, date: str) -> List[Dict]",
    ),
    "book_hotel": ToolType(
        name="book_hotel",
        description=(
            "Books a hotel based on location and preferences. Arguments: location (str), *preferences (variable number of str arguments).\n"
            'Returns a list of hotels, each represented as a dictionary with keys "location", "preferences", "price_per_night", and "rating".\n'
            'Example: [{"location": "A", "preferences": ["wifi", "pool"], "price_per_night": 120, "rating": 4}]'
        ),
        function=book_hotel,
        fn_signature="book_hotel(location: str, *preferences: str) -> List[Dict]",
    ),
    "budget_calculator": ToolType(
        name="budget_calculator",
        description=(
            "Calculates the total budget for a trip. Arguments: flight_price (float), hotel_price_per_night (float), num_nights (int).\n"
            "Returns the total budget (float)."
        ),
        function=budget_calculator,
        fn_signature="budget_calculator(flight_price: float, hotel_price_per_night: float, num_nights: int) -> float",
    ),
    "max": ToolType(
        name="max",
        description="Finds the maximum value among the given arguments. Accepts variable number of float arguments.",
        function=max,
        fn_signature="max(*args: float) -> float",
    ),
    "min": ToolType(
        name="min",
        description="Finds the minimum value among the given arguments. Accepts variable number of float arguments.",
        function=min,
        fn_signature="min(*args: float) -> float",
    ),
    "sum": ToolType(
        name="sum",
        description="Sums the given arguments. Accepts variable number of float arguments.",
        function=lambda *args: sum(args),
        fn_signature="sum(*args: float) -> float",
    ),
}

# Defining Tasks for Travel Itinerary Planning
TASKS = [
    Task(
        name=f"{CUR_TASK_NAME}/plan_trip_to_0",
        tools=CUR_TOOLS,
        instruction=f"You are at {LOC[4]}. Plan a trip to {LOC[0]} on 2023-12-25, staying in a hotel with wifi and a pool for 5 nights. Give me the total budget for the trip.",
        expected_output=budget_calculator(
            find_flights(LOC[4], LOC[0], "2023-12-25")[0]["price"],
            book_hotel(LOC[0], "wifi", "pool")[0]["price_per_night"],
            5,
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/cheapest_new_york_trip",
        tools=CUR_TOOLS,
        instruction=f"You are at {LOC[4]}. Find the cheapest flight and hotel combination for a trip to {LOC[1]} on 2023-11-10 for 3 nights. Give me the total budget for the trip.",
        expected_output=min(
            [
                budget_calculator(flight["price"], hotel["price_per_night"], 3)
                for flight in find_flights(LOC[4], LOC[1], "2023-11-10")
                for hotel in book_hotel(LOC[1])
            ],
            default=float("inf"),
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/luxury_tokyo_trip",
        tools=CUR_TOOLS,
        instruction=f"You are at {LOC[4]}. Plan a luxury trip to {LOC[2]} on 2023-10-05, staying in the highest-rated hotel for 7 nights. Always choose the cheaper flight. Give me the total budget for the trip.",
        expected_output=budget_calculator(
            min(find_flights(LOC[4], LOC[2], "2023-10-05"), key=lambda f: f["price"])[
                "price"
            ],
            max(book_hotel(LOC[2]), key=lambda h: h["rating"])["price_per_night"],
            7,
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/budget_trip_to_paris",
        tools=CUR_TOOLS,
        instruction=f"You are at {LOC[4]}. Plan a budget trip to {LOC[0]} on 2023-12-25, staying in the cheapest available hotel for 4 nights. Always choose the cheaper flight. Give me the total budget for the trip.",
        expected_output=budget_calculator(
            min(find_flights(LOC[4], LOC[0], "2023-12-25"), key=lambda f: f["price"])[
                "price"
            ],
            min(book_hotel(LOC[0]), key=lambda h: h["price_per_night"])[
                "price_per_night"
            ],
            4,
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/new_york_trip_with_specific_preferences",
        tools=CUR_TOOLS,
        instruction=f"You are at {LOC[4]}. Plan a trip to {LOC[1]} on 2023-11-10, staying in a hotel with a gym but without a pool for 3 nights. Always choose the cheaper flight and hotel. Give me the total budget for the trip.",
        expected_output=budget_calculator(
            min(find_flights(LOC[4], LOC[1], "2023-11-10"), key=lambda f: f["price"])[
                "price"
            ],
            min(
                [
                    hotel
                    for hotel in book_hotel(LOC[1], "gym")
                    if "pool" not in hotel["preferences"]
                ],
                key=lambda h: h["price_per_night"],
            )["price_per_night"],
            3,
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/london_business_trip",
        tools=CUR_TOOLS,
        instruction=f"You are at {LOC[4]}. Find the most economical flight and hotel for a business trip to {LOC[3]} on 2023-08-15 for 4 nights, with only a wifi requirement. Give me the total budget for the trip.",
        expected_output=min(
            [
                budget_calculator(flight["price"], hotel["price_per_night"], 4)
                for flight in find_flights(LOC[4], LOC[3], "2023-08-15")
                for hotel in book_hotel(LOC[3], "wifi")
            ],
            default=float("inf"),
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/last_minute_tokyo_trip",
        tools=CUR_TOOLS,
        instruction=f"You are at {LOC[4]}. Organize a last-minute trip to {LOC[2]} on 2023-12-01 for 3 nights in any available hotel. Always go with the cheaper hotel and flight. Give me the total budget for the trip.",
        expected_output=budget_calculator(
            min(find_flights(LOC[4], LOC[2], "2023-12-01"), key=lambda f: f["price"])[
                "price"
            ],
            min(book_hotel(LOC[2]), key=lambda h: h["price_per_night"])[
                "price_per_night"
            ],
            3,
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/new_york_holiday_season",
        tools=CUR_TOOLS,
        instruction=f"You are at {LOC[4]}. Plan a holiday trip to {LOC[1]} on 2023-12-15, preferring a hotel with a gym and pool, for 7 nights. Always go with the cheaper hotel and flight. Give me the total budget for the trip.",
        expected_output=budget_calculator(
            min(find_flights(LOC[4], LOC[1], "2023-12-15"), key=lambda f: f["price"])[
                "price"
            ],
            min(book_hotel(LOC[1], "gym", "pool"), key=lambda h: h["price_per_night"])[
                "price_per_night"
            ],
            7,
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/multi_city_business_trip",
        tools=CUR_TOOLS,
        instruction=(
            f"You are at {LOC[4]}. Plan a multi-city business trip: first to {LOC[1]} on 2023-11-10 for 2 nights, "
            f"then to {LOC[2]} on 2023-11-13 for 3 nights. Choose hotels with wifi. Always go with the cheaper hotel and flight. Give me the total budget for the trip."
        ),
        expected_output=(
            budget_calculator(
                min(
                    find_flights(LOC[4], LOC[1], "2023-11-10"), key=lambda f: f["price"]
                )["price"],
                min(book_hotel(LOC[1], "wifi"), key=lambda h: h["price_per_night"])[
                    "price_per_night"
                ],
                2,
            )
            + budget_calculator(
                min(
                    find_flights(LOC[1], LOC[2], "2023-11-13"), key=lambda f: f["price"]
                )["price"],
                min(book_hotel(LOC[2], "wifi"), key=lambda h: h["price_per_night"])[
                    "price_per_night"
                ],
                3,
            )
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/leisure_multi_city_trip",
        tools=CUR_TOOLS,
        instruction=(
            f"You are at {LOC[4]}. Plan a leisure multi-city trip: first to {LOC[0]} on 2023-12-25 for 3 nights, "
            f"then to {LOC[3]} on 2023-12-28 for 2 nights. No specific hotel preferences. Always go with the cheaper hotel and flight. Give me the total budget for the trip."
        ),
        expected_output=(
            budget_calculator(
                min(
                    find_flights(LOC[4], LOC[0], "2023-12-25"), key=lambda f: f["price"]
                )["price"],
                min(book_hotel(LOC[0]), key=lambda h: h["price_per_night"])[
                    "price_per_night"
                ],
                3,
            )
            + budget_calculator(
                min(
                    find_flights(LOC[0], LOC[3], "2023-12-28"), key=lambda f: f["price"]
                )["price"],
                min(book_hotel(LOC[3]), key=lambda h: h["price_per_night"])[
                    "price_per_night"
                ],
                2,
            )
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/three_city_european_tour",
        tools=CUR_TOOLS,
        instruction=(
            f"You are at {LOC[4]}. Plan a three-city European tour starting from {LOC[4]} to {LOC[0]} on 2023-12-25 (3 nights), "
            f"then going to {LOC[3]} on 2023-12-29 (4 nights), and finally to {LOC[2]} on 2024-01-02 (3 nights). "
            f"Choose hotels with wifi in each city. Always go with the cheaper flight and hotel. Give me the total budget for the trip."
        ),
        expected_output=(
            budget_calculator(
                min(
                    find_flights(LOC[4], LOC[0], "2023-12-25"), key=lambda f: f["price"]
                )["price"],
                min(book_hotel(LOC[0], "wifi"), key=lambda h: h["price_per_night"])[
                    "price_per_night"
                ],
                3,
            )
            + budget_calculator(
                min(
                    find_flights(LOC[0], LOC[3], "2023-12-29"), key=lambda f: f["price"]
                )["price"],
                min(book_hotel(LOC[3], "wifi"), key=lambda h: h["price_per_night"])[
                    "price_per_night"
                ],
                4,
            )
            + budget_calculator(
                min(
                    find_flights(LOC[3], LOC[2], "2024-01-02"), key=lambda f: f["price"]
                )["price"],
                min(book_hotel(LOC[2], "wifi"), key=lambda h: h["price_per_night"])[
                    "price_per_night"
                ],
                3,
            )
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/adventure_trip",
        tools=CUR_TOOLS,
        instruction=(
            f"Plan an adventure trip starting from {LOC[4]} to {LOC[2]} on 2023-12-01 (2 nights), "
            f"then going to {LOC[1]} on 2023-12-03 (3 nights), and finally to {LOC[3]} on 2023-12-06 (2 nights). "
            f"In {LOC[2]}, prefer a hotel with a pool; in {LOC[1]}, any hotel; in {LOC[3]}, a hotel with a gym. "
            f"Always go with the cheaper flight and hotel. Give me the total budget for the trip."
        ),
        expected_output=(
            budget_calculator(
                min(
                    find_flights(LOC[4], LOC[2], "2023-12-01"), key=lambda f: f["price"]
                )["price"],
                min(book_hotel(LOC[2], "pool"), key=lambda h: h["price_per_night"])[
                    "price_per_night"
                ],
                2,
            )
            + budget_calculator(
                min(
                    find_flights(LOC[2], LOC[1], "2023-12-03"), key=lambda f: f["price"]
                )["price"],
                min(book_hotel(LOC[1]), key=lambda h: h["price_per_night"])[
                    "price_per_night"
                ],
                3,
            )
            + budget_calculator(
                min(
                    find_flights(LOC[1], LOC[3], "2023-12-06"), key=lambda f: f["price"]
                )["price"],
                min(book_hotel(LOC[3], "gym"), key=lambda h: h["price_per_night"])[
                    "price_per_night"
                ],
                2,
            )
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/cultural_exploration_tour",
        tools=CUR_TOOLS,
        instruction=(
            f"Plan a cultural exploration tour starting from {LOC[4]} to {LOC[0]} on 2023-12-25 (4 nights), "
            f"then going to {LOC[1]} on 2023-12-29 (2 nights), and finally to {LOC[3]} on 2024-01-01 (3 nights). "
            f"Choose hotels with wifi in each city. In {LOC[3]}, prefer a hotel with a high rating. "
            f"Always go with the cheaper flight. Give me the total budget for the trip."
        ),
        expected_output=(
            budget_calculator(
                min(
                    find_flights(LOC[4], LOC[0], "2023-12-25"), key=lambda f: f["price"]
                )["price"],
                min(book_hotel(LOC[0], "wifi"), key=lambda h: h["price_per_night"])[
                    "price_per_night"
                ],
                4,
            )
            + budget_calculator(
                min(
                    find_flights(LOC[0], LOC[1], "2023-12-29"), key=lambda f: f["price"]
                )["price"],
                min(book_hotel(LOC[1], "wifi"), key=lambda h: h["price_per_night"])[
                    "price_per_night"
                ],
                2,
            )
            + budget_calculator(
                min(
                    find_flights(LOC[1], LOC[3], "2024-01-01"), key=lambda f: f["price"]
                )["price"],
                max(book_hotel(LOC[3], "wifi"), key=lambda h: h["rating"])[
                    "price_per_night"
                ],
                3,
            )
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/leisure_and_business_combo",
        tools=CUR_TOOLS,
        instruction=(
            f"Plan a combination of leisure and business trip starting from {LOC[4]} to {LOC[3]} on 2023-08-15 (3 nights) for business, "
            f"then going to {LOC[0]} on 2023-08-18 (2 nights) for leisure, and finally to {LOC[2]} on 2023-08-20 (2 nights) for business. "
            f"For business stays, prefer hotels with wifi and gym; for leisure, prefer hotels with a pool. "
            f"Always go with the cheaper flight and hotel. Give me the total budget for the trip."
        ),
        expected_output=(
            budget_calculator(
                min(
                    find_flights(LOC[4], LOC[3], "2023-08-15"), key=lambda f: f["price"]
                )["price"],
                min(
                    book_hotel(LOC[3], "wifi", "gym"),
                    key=lambda h: h["price_per_night"],
                )["price_per_night"],
                3,
            )
            + budget_calculator(
                min(
                    find_flights(LOC[3], LOC[0], "2023-08-18"), key=lambda f: f["price"]
                )["price"],
                min(book_hotel(LOC[0], "pool"), key=lambda h: h["price_per_night"])[
                    "price_per_night"
                ],
                2,
            )
            + budget_calculator(
                min(
                    find_flights(LOC[0], LOC[2], "2023-08-20"), key=lambda f: f["price"]
                )["price"],
                min(
                    book_hotel(LOC[2], "wifi", "gym"),
                    key=lambda h: h["price_per_night"],
                )["price_per_night"],
                2,
            )
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/grand_tour_vacation",
        tools=CUR_TOOLS,
        instruction=(
            f"Plan a grand tour vacation starting from {LOC[4]} to {LOC[2]} on 2023-12-01 (2 nights), "
            f"then to {LOC[3]} on 2023-12-03 (3 nights), followed by a visit to {LOC[1]} on 2023-12-06 (2 nights), "
            f"and finally to {LOC[4]} on 2023-12-08 (3 nights). "
            f"Choose hotels with a high rating in each city. Always go with the cheaper flight. Give me the total budget for the trip."
        ),
        expected_output=(
            budget_calculator(
                min(
                    find_flights(LOC[4], LOC[2], "2023-12-01"), key=lambda f: f["price"]
                )["price"],
                max(book_hotel(LOC[2]), key=lambda h: h["rating"])["price_per_night"],
                2,
            )
            + budget_calculator(
                min(
                    find_flights(LOC[2], LOC[3], "2023-12-03"), key=lambda f: f["price"]
                )["price"],
                max(book_hotel(LOC[3]), key=lambda h: h["rating"])["price_per_night"],
                3,
            )
            + budget_calculator(
                min(
                    find_flights(LOC[3], LOC[1], "2023-12-06"), key=lambda f: f["price"]
                )["price"],
                max(book_hotel(LOC[1]), key=lambda h: h["rating"])["price_per_night"],
                2,
            )
            + budget_calculator(
                min(
                    find_flights(LOC[1], LOC[4], "2023-12-08"), key=lambda f: f["price"]
                )["price"],
                max(book_hotel(LOC[4]), key=lambda h: h["rating"])["price_per_night"],
                3,
            )
        ),
        is_single_tool_task=False,
    ),
]
register_task_iterator(TASKS, len(TASKS))
print(f"**** {len(TASKS)} tasks registered for {CUR_TASK_NAME} ****")
