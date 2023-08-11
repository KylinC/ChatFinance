from langchain import PromptTemplate

PROMPT = """
你需要扮演一个优秀的关键信息提取助手，从人类的对话中提取关键性内容（最多5个关键词），以协助其他助手更精准地回答问题。

注意：你不需要做任何解释说明，只需严格按照示例的格式输出关键词。

示例：
人类：我有一个服装厂，是否可以应用你们的装箱算法改善装载率呢？
AI: 服装厂, 装箱算法, 装载率

现在开始：
人类：{query}
AI:
"""


def information_extraction_raw_prompt():
    return PromptTemplate(template=PROMPT, input_variables=["query"])


def information_extraction_prompt(query: str):
    P = PromptTemplate(template=PROMPT, input_variables=["query"])
    return P.format(query=query)


if __name__ == "__main__":
    print(information_extraction_prompt("你们的装箱算法能不能用在家居业呀？"))
