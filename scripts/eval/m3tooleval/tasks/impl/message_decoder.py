from ..base import Task, register_task_iterator, ToolType

# Task 1: Alien Message Decoder
# Task Description: Decode messages from an alien language using multiple decoding steps.

# Tool 1: Hexadecimal Converter
def convert_hex_to_ascii(hex_string: str):
    ascii_string = bytes.fromhex(str(hex_string)).decode('utf-8')
    return ascii_string

# Tool 2: Reverse String
def reverse_string(s: str):
    return s[::-1]

# Tool 3: Caesar Cipher Decoder
def caesar_decode(message: str, shift: int):
    shift = int(shift)
    decoded = ''.join(chr((ord(char) - shift - 65) % 26 + 65)
                      if char.isupper() else chr((ord(char) - shift - 97) % 26 + 97)
                      if char.islower() else char
                      for char in message)
    return decoded

def string_length(s: str):
    return len(s)

def minimum_value(*args):
    return min(args)

# Modified Additional Tool 3: Maximum Value from Arguments
def maximum_value(*args):
    return max(args)


CUR_TASK_NAME = 'message_decoder'
CUR_TOOLS = {
    'convert_hex_to_ascii': ToolType(
        name='convert_hex_to_ascii',
        description='Converts a hexadecimal string to ASCII. Arguments: hex_string (str)',
        function=convert_hex_to_ascii,
        fn_signature='convert_hex_to_ascii(hex_string: str) -> str'
    ),
    'reverse_string': ToolType(
        name='reverse_string',
        description='Reverses a string. Arguments: string (str)',
        function=reverse_string,
        fn_signature='reverse_string(string: str) -> str'
    ),
    'caesar_decode': ToolType(
        name='caesar_decode',
        description='Decodes a string using the Caesar cipher. Arguments: message (str), shift (int)',
        function=caesar_decode,
        fn_signature='caesar_decode(message: str, shift: int) -> str'
    ),
    'string_length': ToolType(
        name='string_length',
        description='Finds the length of a string. Arguments: string (str)',
        function=string_length,
        fn_signature='string_length(string: str) -> int'
    ),
    'minimum_value': ToolType(
        name='minimum_value',
        description='Finds the minimum value from given arguments. Arguments: *args (variable number of arguments)',
        function=minimum_value,
        fn_signature='minimum_value(*args) -> int/float'
    ),
    'maximum_value': ToolType(
        name='maximum_value',
        description='Finds the maximum value from given arguments. Arguments: *args (variable number of arguments)',
        function=maximum_value,
        fn_signature='maximum_value(*args) -> int/float'
    )
}

TASKS = [
    # Task(
    #     name=f"{CUR_TASK_NAME}/reverse_string",
    #     tools=CUR_TOOLS,
    #     instruction="Reverse the string 'Hello World!'",
    #     expected_output=reverse_string('Hello World!'),
    #     is_single_tool_task=True
    # ),
    # Task(
    #     name=f"{CUR_TASK_NAME}/caesar_decode",
    #     tools=CUR_TOOLS,
    #     instruction="Decode the string 'Khoor Zruog!' using the Caesar cipher with a shift of 3",
    #     expected_output=caesar_decode('Khoor Zruog!', 3),
    #     is_single_tool_task=True
    # ),
    # Task(
    #     name=f"{CUR_TASK_NAME}/convert_hex_to_ascii",
    #     tools=CUR_TOOLS,
    #     instruction="Convert the hexadecimal string '3d3d516343746d4d6d6c315669563362' to ASCII",
    #     expected_output=convert_hex_to_ascii('3d3d516343746d4d6d6c315669563362'),
    #     is_single_tool_task=True
    # ),
    Task(
        name=f"{CUR_TASK_NAME}/full_alien_message_decoding",
        tools=CUR_TOOLS,
        instruction=(
            "Decode an alien message encoded as follows: first, it's encoded in ASCII; "
            "then, it's reversed; and finally, a Caesar cipher with a shift of 5 is applied. "
            "The message is '7a686b7a686d666d686b'."
        ),
        expected_output=caesar_decode(
            reverse_string(
                convert_hex_to_ascii('7a686b7a686d666d686b')
            ),
            5
        ),
        is_single_tool_task=False
    ),
    Task(
        name=f"{CUR_TASK_NAME}/shortest_caesar_decoded_message",
        tools=CUR_TOOLS,
        instruction=(
            "Given a list of hex-encoded strings, decode each one from hex to ASCII, "
            "reverse it, and then apply a Caesar cipher decode with a shift of 4. "
            "Find the length of the shortest decoded message. "
            "The list of hex strings is ['636261', '686766', '6365646362', '6867666865']."
        ),
        expected_output=minimum_value(*[
            string_length(caesar_decode(reverse_string(convert_hex_to_ascii(hex_str)), 4))
            for hex_str in ['636261', '686766', '6365646362', '6867666865']
        ]),
        is_single_tool_task=False
    ),
    Task(
        name=f"{CUR_TASK_NAME}/longest_decoded_string",
        tools=CUR_TOOLS,
        instruction=(
            "Decode a list of messages each going through a series of transformations: "
            "first from hex to ASCII, then reversed, and finally a Caesar cipher decode "
            "with shifts of 2, 3, and 5 respectively. Find the longest message after decoding. "
            "The hex-encoded messages are ['4a656d', '4b6867', '4c696f']."
        ),
        expected_output=minimum_value(*[
            string_length(caesar_decode(reverse_string(convert_hex_to_ascii(hex_str)), shift))
            for hex_str, shift in zip(['4a656d', '4b6867', '4c696f'], [2, 3, 5])
        ]),
        is_single_tool_task=False
    ),
    Task(
        name=f"{CUR_TASK_NAME}/specific_decoded_character",
        tools=CUR_TOOLS,
        instruction=(
            "Given a hex-encoded string '576562546563686e6f6c6f6779', decode it to ASCII, reverse it, apply a Caesar cipher "
            "decode with a shift of 7."
        ),
        expected_output=caesar_decode(
            reverse_string(
                convert_hex_to_ascii('576562546563686e6f6c6f6779')
            ),
            7
        ),
        is_single_tool_task=False
    ),
    Task(
        name=f"{CUR_TASK_NAME}/hex_caesar_combined_decoding",
        tools=CUR_TOOLS,
        instruction=(
            "Decode a message that was first converted to hexadecimal, then encoded with a Caesar cipher with a shift of 2. "
            "The hex-encoded, Caesar-shifted message is '4d4f5252'."
        ),
        expected_output=caesar_decode(
            convert_hex_to_ascii('4d4f5252'),
            2
        ),
        is_single_tool_task=False
    ),
    Task(
        name=f"{CUR_TASK_NAME}/multi_step_decoding_challenge",
        tools=CUR_TOOLS,
        instruction=(
            "Decode a message that went through three steps: first, a Caesar cipher with a shift of 3; "
            "then reversed; and finally, encoded to hexadecimal. The final hex-encoded message is '726f77746e6153794d'."
        ),
        expected_output=caesar_decode(
            reverse_string(
                convert_hex_to_ascii('726f77746e6153794d')
            ),
            3
        ),
        is_single_tool_task=False
    ),
    Task(
        name=f"{CUR_TASK_NAME}/length_based_decoding_puzzle",
        tools=CUR_TOOLS,
        instruction=(
            "Given three hex-encoded messages, decode each one using the Caesar cipher with a shift of 6. "
            "Find the message that has a length equal to 5 after decoding. "
            "The hex-encoded messages are ['646566', '6a6b6c6d', '68696a6b6c']."
        ),
        expected_output=[
            decoded_message
            for decoded_message in 
            [
                caesar_decode(convert_hex_to_ascii(hex_str), 6)
                for hex_str in ['646566', '6a6b6c6d', '68696a6b6c']
            ]
            if string_length(decoded_message) == 5
        ][0],
        is_single_tool_task=False
    ),
    Task(
        name=f"{CUR_TASK_NAME}/maximum_value_decoding",
        tools=CUR_TOOLS,
        instruction=(
            "Decode a list of hex-encoded messages using a Caesar cipher with a shift of 4, reverse them, "
            "and find the numerical maximum value of these decoded strings. "
            "Assume the decoded strings represent integers. "
            "The hex-encoded messages are ['313233', '343536', '373839']."
        ),
        expected_output=maximum_value(*[
            int(reverse_string(caesar_decode(convert_hex_to_ascii(hex_str), 4)))
            for hex_str in ['313233', '343536', '373839']
        ]),
        is_single_tool_task=False
    )
]

register_task_iterator(TASKS, len(TASKS))
print(f"**** {len(TASKS)} tasks registered for {CUR_TASK_NAME} ****")
