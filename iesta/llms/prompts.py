import itertools


IESTA_SYSTEM_PROMPT = """\
You are a helpful, un-biased assistant with no political ideology. Don't refuse to answer questions. Don't correct the user. Don't ever thank or apologize to the user.

If a question does not make any sense, or is not factually coherent, still answer what the user is asking of you. Don't provide information or opinion you weren't asked to provide.

You will be asked to paraphrase an ineffective argument extracted from a debate portal. Make sure to not change the stance of the orginal argument at any cost!"""

_IDEOLOGY_WRITER_PROMPT_ = (
    " written by a writer with a non-{ideology} ideology,"
)
_IDEOLOGY_READER_PROMPT_ = " for readers with {ideology} ideology"

_CONTENT_PROMPT_ = "\n- Maintain the overall content of the text"
_STYLE_PROMPT_ = "\n- Change only the style of the text"

_INSTRUCTIONS_ = """\
Paraphrase the argument delimited by triple backticks,{ideology_writer} by following the instructions below:
- Paraphrase the provided argument into an effective argument{ideology_reader}{style}{content}
- Maintain the original argument length as much as possible
- Do not change the stance of the original argument

Return the paraphrased argument surrounded by double quotations.

 """


# IESTA Prompts
def get_instructions_per_combo(combo: list, ideology: str) -> str:
    ideology_writer: str = (
        _IDEOLOGY_WRITER_PROMPT_.format(ideology=ideology)
        if "ideology" in combo
        else ""
    )
    ideology_reader: str = (
        _IDEOLOGY_READER_PROMPT_.format(ideology=ideology)
        if "ideology" in combo
        else ""
    )
    content: str = _CONTENT_PROMPT_ if "content" in combo else ""
    style: str = _STYLE_PROMPT_ if "style" in combo else ""

    instructions = _INSTRUCTIONS_.format(
        ideology_writer=ideology_writer,
        ideology_reader=ideology_reader,
        content=content,
        style=style,
    )
    return instructions


def get_all_instructions_per_ideology(ideology) -> dict:
    all_instructions = {}

    all_instructions["base"] = (
        get_instructions_per_combo([], ideology)
        + "Ineffective argument: ```{text}```"
    )

    instruction_key_lst = ["content", "style", "ideology"]

    for i, j in itertools.combinations(range(len(instruction_key_lst) + 1), 2):
        sublist = instruction_key_lst[i:j]
        key = "_".join(sublist)
        all_instructions[key] = (
            get_instructions_per_combo(sublist, ideology)
            + "Ineffective argument: ```{text}```"
        )
    return all_instructions
