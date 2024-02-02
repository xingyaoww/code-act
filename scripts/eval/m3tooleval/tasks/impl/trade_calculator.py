from ..base import Task, register_task_iterator, ToolType

# Task 3: Intergalactic Trade Calculator
# Task Description: Calculate the trade value and tax for an intergalactic trader dealing in a variety of space commodities.


# Tool 1: Currency Converter
def convert_currency(base_price, conversion_rate):
    return base_price * conversion_rate


# Tool 2: Trade Tariff Calculator
def calculate_tariff(price, tariff_rate):
    return price * tariff_rate / 100


# Tool 3: Final Trade Value Estimator
def estimate_final_value(price, tariff):
    return price + tariff


def calculator(expression: str) -> float:
    try:
        return eval(expression)
    except:
        return f"Failed to evaluate expression: {expression}"


def find_minimum(*args):
    return min(args)


def find_maximum(*args):
    return max(args)


# Registering the tools and task for Task 3
CUR_TASK_NAME = "trade_calculator"
CUR_TOOLS = {
    "convert_currency": ToolType(
        name="convert_currency",
        description="Converts the commodity price to local currency. Arguments: base_price (float), conversion_rate (float)",
        function=convert_currency,
        fn_signature="convert_currency(base_price: float, conversion_rate: float) -> float",
    ),
    "calculate_tariff": ToolType(
        name="calculate_tariff",
        description="Calculates the trade tariff based on the converted price. Arguments: price (float), tariff_rate (float, in %)",
        function=calculate_tariff,
        fn_signature="calculate_tariff(price: float, tariff_rate: float) -> float",
    ),
    "estimate_final_value": ToolType(
        name="estimate_final_value",
        description="Estimates the final trade value including the tariff. Arguments: price (float), tariff (float)",
        function=estimate_final_value,
        fn_signature="estimate_final_value(price: float, tariff: float) -> float",
    ),
    "calculator": ToolType(
        name="calculator",
        description='Evaluates the given expression and returns the result. Accepts a calculation expression as input. For example, "2 + (3 * 4)" will return 14.',
        function=calculator,
        fn_signature="calculator(expression: str) -> float",
    ),
    "find_minimum": ToolType(
        name="find_minimum",
        description="Finds the minimum value among the given arguments. Accepts variable number of float arguments.",
        function=find_minimum,
        fn_signature="find_minimum(*args: float) -> float",
    ),
    "find_maximum": ToolType(
        name="find_maximum",
        description="Finds the maximum value among the given arguments. Accepts variable number of float arguments.",
        function=find_maximum,
        fn_signature="find_maximum(*args: float) -> float",
    ),
}


TASKS = [
    Task(
        name=f"{CUR_TASK_NAME}/estimate_final_value",
        tools=CUR_TOOLS,
        instruction="Estimate the final trade value for a price of 150 units with a tariff of 7.5 units.",
        expected_output=estimate_final_value(150, calculate_tariff(150, 7.5)),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/interstellar_trade_transaction",
        tools=CUR_TOOLS,
        instruction=(
            "Calculate the final trade value for an intergalactic trader who is exchanging 100 units of space ore. "
            "The base price per unit is 50 galactic credits, the conversion rate to local currency is 1.5, "
            "and the trade tariff rate is 8%. Provide the final trade value in local currency."
        ),
        expected_output=estimate_final_value(
            convert_currency(100 * 50, 1.5),
            calculate_tariff(convert_currency(100 * 50, 1.5), 8),
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/taxed_commodity_export",
        tools=CUR_TOOLS,
        instruction=(
            "An intergalactic trader is exporting 200 units of a rare mineral. The base price per unit is 120 galactic credits. "
            "The conversion rate today is 2.0 for converting galactic credits to the universal trade currency (UTC). "
            "The tariff rate for exporting this mineral is 5%. Calculate the total cost including tariff in UTC."
        ),
        expected_output=estimate_final_value(
            convert_currency(200 * 120, 2.0),
            calculate_tariff(convert_currency(200 * 120, 2.0), 5),
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/best_trade_route_selection",
        tools=CUR_TOOLS,
        instruction=(
            "An intergalactic trader is analyzing multiple trade routes to sell a cargo of 500 units of space spices. "
            "There are three routes available with different conversion rates and tariff rates as follows: "
            "Route 1: Conversion rate of 1.3 and tariff rate of 7%, "
            "Route 2: Conversion rate of 1.4 and tariff rate of 9%, "
            "Route 3: Conversion rate of 1.2 and tariff rate of 4%. "
            "Find the least final trade value after applying conversion and tariff and answer that value."
        ),
        expected_output=find_minimum(
            estimate_final_value(
                convert_currency(500, 1.3),
                calculate_tariff(convert_currency(500, 1.3), 7),
            ),
            estimate_final_value(
                convert_currency(500, 1.4),
                calculate_tariff(convert_currency(500, 1.4), 9),
            ),
            estimate_final_value(
                convert_currency(500, 1.2),
                calculate_tariff(convert_currency(500, 1.2), 4),
            ),
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/most_profitable_conversion_rate",
        tools=CUR_TOOLS,
        instruction=(
            "A trader has access to different currency markets to convert 1000 units of a high-value liquid commodity. "
            "The conversion rates available are 2.5, 2.3, and 2.7. The commodity has a fixed base price of 300 galactic credits per unit. "
            "However, a fixed tariff of 10% applies in all markets. "
            "Determine the final trade value with the most profitable conversion rate. Answer that value."
        ),
        expected_output=find_maximum(
            estimate_final_value(
                convert_currency(1000 * 300, 2.5),
                calculate_tariff(convert_currency(1000 * 300, 2.5), 10),
            ),
            estimate_final_value(
                convert_currency(1000 * 300, 2.3),
                calculate_tariff(convert_currency(1000 * 300, 2.3), 10),
            ),
            estimate_final_value(
                convert_currency(1000 * 300, 2.7),
                calculate_tariff(convert_currency(1000 * 300, 2.7), 10),
            ),
        ),
        is_single_tool_task=False,
    ),
    Task(
        name=f"{CUR_TASK_NAME}/bulk_trade_quantity_discount",
        tools=CUR_TOOLS,
        instruction=(
            "An intergalactic merchant is offering a bulk discount for a purchase of 2000 units or more of mechanized labor robots. "
            "The base price per unit is 1500 galactic credits. For purchases of 2000 units or more, a 5% discount is applied. "
            "The conversion rate is 1.8 and the tariff rate is 12%. Calculate the total cost after discount, conversion, and tariff "
            "for a trade of 2500 units."
        ),
        expected_output=(
            lambda units=2500, base_price=1500, conversion_rate=1.8, tariff_rate=12: estimate_final_value(
                convert_currency(
                    (units * base_price) * (0.95 if units >= 2000 else 1),
                    conversion_rate,
                ),
                calculate_tariff(
                    convert_currency(
                        (units * base_price) * (0.95 if units >= 2000 else 1),
                        conversion_rate,
                    ),
                    tariff_rate,
                ),
            )
        )(),
        is_single_tool_task=False,
    ),
]


# Task 6: Comparative Profit Analysis for Different Commodities
def task6_complex_operation(prices, conversion_rates, tariff_rates, quantities):
    final_values = [
        estimate_final_value(
            convert_currency(price * quantity, conversion_rate),
            calculate_tariff(
                convert_currency(price * quantity, conversion_rate), tariff_rate
            ),
        )
        for price, conversion_rate, tariff_rate, quantity in zip(
            prices, conversion_rates, tariff_rates, quantities
        )
    ]
    return max(final_values)


TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/comparative_profit_analysis",
        tools=CUR_TOOLS,
        instruction=(
            "Analyze the profit margins for selling two different commodities in an intergalactic market. "
            "The first commodity has a base price of 100 credits per unit, a conversion rate of 1.6, a tariff rate of 6%, and a quantity of 300 units. "
            "The second commodity has a base price of 80 credits per unit, a conversion rate of 1.8, a tariff rate of 5%, and a quantity of 500 units. "
            "Determine the maximum final trade values among these commodities."
        ),
        expected_output=task6_complex_operation(
            [100, 80], [1.6, 1.8], [6, 5], [300, 500]
        ),
        is_single_tool_task=False,
    )
)


# Task 7: Impact of Fluctuating Conversion Rates on Trade Value
def task7_complex_operation(base_price, quantities, conversion_rates, tariff_rate):
    fluctuating_values = [
        estimate_final_value(
            convert_currency(base_price * quantity, conversion_rate),
            calculate_tariff(
                convert_currency(base_price * quantity, conversion_rate), tariff_rate
            ),
        )
        for quantity, conversion_rate in zip(quantities, conversion_rates)
    ]
    return fluctuating_values


TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/fluctuating_conversion_rates_impact",
        tools=CUR_TOOLS,
        instruction=(
            "Evaluate how fluctuating conversion rates impact the trade value of a commodity. "
            "The commodity has a base price of 200 credits per unit and a tariff rate of 7%. "
            "Assess the final trade values for quantities of 100, 150, and 200 units at conversion rates of 1.7, 1.5, and 1.6 respectively. "
            "Provide your answer as a list of three values, separated by commas. "
            "Example: 'Answer: [1000, 2000, 3000]'"
        ),
        expected_output=task7_complex_operation(
            200, [100, 150, 200], [1.7, 1.5, 1.6], 7
        ),
        is_single_tool_task=False,
    )
)

# Task 8: Optimal Trade Route with Multiple Commodities
TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/optimal_trade_route_multiple_commodities",
        tools=CUR_TOOLS,
        instruction=(
            "A trader needs to select the most profitable trade route for three different commodities. "
            "Commodity 1: 150 units at 200 credits per unit, conversion rate 1.7, tariff rate 8%. "
            "Commodity 2: 250 units at 300 credits per unit, conversion rate 1.5, tariff rate 5%. "
            "Commodity 3: 100 units at 400 credits per unit, conversion rate 2.0, tariff rate 10%. "
            "Calculate the final trade value for each commodity and answer the highest value."
        ),
        expected_output=max(
            estimate_final_value(
                convert_currency(150 * 200, 1.7),
                calculate_tariff(convert_currency(150 * 200, 1.7), 8),
            ),
            estimate_final_value(
                convert_currency(250 * 300, 1.5),
                calculate_tariff(convert_currency(250 * 300, 1.5), 5),
            ),
            estimate_final_value(
                convert_currency(100 * 400, 2.0),
                calculate_tariff(convert_currency(100 * 400, 2.0), 10),
            ),
        ),
        is_single_tool_task=False,
    )
)

# Task 9: Seasonal Price Fluctuation Impact
TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/seasonal_price_fluctuation_impact",
        tools=CUR_TOOLS,
        instruction=(
            "Assess the impact of seasonal price fluctuations on trade value. "
            "A trader is dealing with a commodity whose base price varies seasonally: "
            "200 credits in summer, 250 credits in autumn, 180 credits in winter. "
            "The quantity is 200 units, the conversion rate is 1.8, and the tariff rate is 7%. "
            "Calculate the final trade values for each season and list them in the order of summer, autumn, winter."
            "Example: 'Answer: [1000, 2000, 3000]'"
        ),
        expected_output=[
            estimate_final_value(
                convert_currency(200 * 200, 1.8),
                calculate_tariff(convert_currency(200 * 200, 1.8), 7),
            ),
            estimate_final_value(
                convert_currency(250 * 200, 1.8),
                calculate_tariff(convert_currency(250 * 200, 1.8), 7),
            ),
            estimate_final_value(
                convert_currency(180 * 200, 1.8),
                calculate_tariff(convert_currency(180 * 200, 1.8), 7),
            ),
        ],
        is_single_tool_task=False,
    )
)

# Task 10: Balancing Multiple Trade Offers
TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/balancing_multiple_trade_offers",
        tools=CUR_TOOLS,
        instruction=(
            "A trader has to balance multiple trade offers to maximize profit. "
            "Offer 1: Sell 300 units at 250 credits per unit, conversion rate 1.6, tariff rate 6%. "
            "Offer 2: Buy 150 units at 220 credits per unit, conversion rate 1.5, tariff rate 4%. "
            "Determine the net profit or loss by subtracting the total cost of Offer 2 from the total revenue of Offer 1."
        ),
        expected_output=(
            estimate_final_value(
                convert_currency(300 * 250, 1.6),
                calculate_tariff(convert_currency(300 * 250, 1.6), 6),
            )
            - estimate_final_value(
                convert_currency(150 * 220, 1.5),
                calculate_tariff(convert_currency(150 * 220, 1.5), 4),
            )
        ),
        is_single_tool_task=False,
    )
)

# Task 11: Dynamic Pricing Strategy
TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/dynamic_pricing_strategy",
        tools=CUR_TOOLS,
        instruction=(
            "A trader is considering a dynamic pricing strategy for a commodity with a base price of 300 credits per unit. "
            "The strategy involves adjusting the price based on market demand: increase by 10% in high demand and decrease by 5% in low demand. "
            "Analyze three scenarios - high demand (120 units), normal demand (100 units), and low demand (80 units) with a constant conversion rate of 1.8 and a tariff rate of 6%. "
            "Calculate the final trade value for each scenario. "
            "Provide your answer as a list of three values, separated by commas."
            "Example: 'Answer: [1000, 2000, 3000]'"
        ),
        expected_output=[
            estimate_final_value(
                convert_currency(120 * 300 * 1.10, 1.8),
                calculate_tariff(convert_currency(120 * 300 * 1.10, 1.8), 6),
            ),
            estimate_final_value(
                convert_currency(100 * 300, 1.8),
                calculate_tariff(convert_currency(100 * 300, 1.8), 6),
            ),
            estimate_final_value(
                convert_currency(80 * 300 * 0.95, 1.8),
                calculate_tariff(convert_currency(80 * 300 * 0.95, 1.8), 6),
            ),
        ],
        is_single_tool_task=False,
    )
)

# Task 12: Trade-Off Between Quantity and Tariff Rate
TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/quantity_tariff_tradeoff",
        tools=CUR_TOOLS,
        instruction=(
            "A trader faces a trade-off between selling more units at a higher tariff rate or fewer units at a lower tariff rate. "
            "Option 1: Sell 200 units at a tariff rate of 10%, Option 2: Sell 150 units at a tariff rate of 5%. "
            "The base price per unit is 400 credits, and the conversion rate is 1.7. "
            "Determine the final trade value for each option and identify the more profitable option."
        ),
        expected_output=max(
            estimate_final_value(
                convert_currency(200 * 400, 1.7),
                calculate_tariff(convert_currency(200 * 400, 1.7), 10),
            ),
            estimate_final_value(
                convert_currency(150 * 400, 1.7),
                calculate_tariff(convert_currency(150 * 400, 1.7), 5),
            ),
        ),
        is_single_tool_task=False,
    )
)

# Task 13: Optimizing Bulk Purchase Discounts
TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/optimizing_bulk_purchase_discounts",
        tools=CUR_TOOLS,
        instruction=(
            "A trader can benefit from bulk purchase discounts: 5% off for orders of 1000 units or more. "
            "Evaluate two commodities: Commodity A (base price 150 credits per unit) and Commodity B (base price 180 credits per unit). "
            "Assess the final trade value for purchasing 950 units and 1050 units of each commodity, with a conversion rate of 1.6 and a tariff rate of 7%."
            "Determine the maximum final trade value among the four options."
        ),
        expected_output=max(
            estimate_final_value(
                convert_currency(950 * 150, 1.6),
                calculate_tariff(convert_currency(950 * 150, 1.6), 7),
            ),
            estimate_final_value(
                convert_currency(1050 * 150 * 0.95, 1.6),
                calculate_tariff(convert_currency(1050 * 150 * 0.95, 1.6), 7),
            ),
            estimate_final_value(
                convert_currency(950 * 180, 1.6),
                calculate_tariff(convert_currency(950 * 180, 1.6), 7),
            ),
            estimate_final_value(
                convert_currency(1050 * 180 * 0.95, 1.6),
                calculate_tariff(convert_currency(1050 * 180 * 0.95, 1.6), 7),
            ),
        ),
        is_single_tool_task=False,
    )
)

# Task 14: Hedging Against Currency Fluctuations
TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/currency_fluctuation_hedging",
        tools=CUR_TOOLS,
        instruction=(
            "A trader needs to hedge against currency fluctuations for an impending trade. "
            "The trade involves 500 units of a commodity at 250 credits per unit. "
            "Assess the final trade value under three different conversion rates - 1.5, 1.65, and 1.8, with a constant tariff rate of 8%. "
            "Identify the minimum final trade value among the three options."
        ),
        expected_output=min(
            estimate_final_value(
                convert_currency(500 * 250, 1.5),
                calculate_tariff(convert_currency(500 * 250, 1.5), 8),
            ),
            estimate_final_value(
                convert_currency(500 * 250, 1.65),
                calculate_tariff(convert_currency(500 * 250, 1.65), 8),
            ),
            estimate_final_value(
                convert_currency(500 * 250, 1.8),
                calculate_tariff(convert_currency(500 * 250, 1.8), 8),
            ),
        ),
        is_single_tool_task=False,
    )
)

# Task 15: Maximizing Profit with Variable Tariff Rates
TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/variable_tariff_profit_maximization",
        tools=CUR_TOOLS,
        instruction=(
            "A trader is exploring profit maximization strategies with variable tariff rates. "
            "The commodity price is 300 credits per unit. "
            "Compare three selling strategies: 400 units with a 6% tariff, 350 units with a 4% tariff, and 450 units with a 7% tariff. "
            "The conversion rate is fixed at 1.7. "
            "Determine the maximum final trade value among the three options."
        ),
        expected_output=max(
            estimate_final_value(
                convert_currency(400 * 300, 1.7),
                calculate_tariff(convert_currency(400 * 300, 1.7), 6),
            ),
            estimate_final_value(
                convert_currency(350 * 300, 1.7),
                calculate_tariff(convert_currency(350 * 300, 1.7), 4),
            ),
            estimate_final_value(
                convert_currency(450 * 300, 1.7),
                calculate_tariff(convert_currency(450 * 300, 1.7), 7),
            ),
        ),
        is_single_tool_task=False,
    )
)

# Task 16: Long-Term vs Short-Term Trade Decisions
TASKS.append(
    Task(
        name=f"{CUR_TASK_NAME}/long_term_vs_short_term_trade_decisions",
        tools=CUR_TOOLS,
        instruction=(
            "Evaluate long-term versus short-term trade decisions. "
            "A trader can sell a commodity either now or wait a month for potentially better prices. "
            "Current scenario: sell 600 units at 200 credits per unit, conversion rate 1.8, tariff 5%. "
            "Potential future scenario: sell the same quantity at a 10% higher price, but face a 2% increase in tariff and a 0.2 decrease in conversion rate. "
            "Calculate and compare the final trade values for both scenarios. "
            "Provide your answer as a list of two values in order of [current, future scenario], separated by commas. "
            "Example: 'Answer: [1000, 2000]'"
        ),
        expected_output=[
            estimate_final_value(
                convert_currency(600 * 200, 1.8),
                calculate_tariff(convert_currency(600 * 200, 1.8), 5),
            ),
            estimate_final_value(
                convert_currency(600 * 200 * 1.10, 1.6),
                calculate_tariff(convert_currency(600 * 200 * 1.10, 1.6), 7),
            ),
        ],
        is_single_tool_task=False,
    )
)

register_task_iterator(TASKS, len(TASKS))
print(f"**** {len(TASKS)} tasks registered for {CUR_TASK_NAME} ****")
