from langchain import PromptTemplate

PROMPT = """
你需要扮演一个优秀的金融助手。现在已经为你提供了相关的数据给你参考，你需要根据提供的额外信息回答人类的问题。

回答要简练，清晰，准确。多参考额外信息。

现在开始：

人类：{query}
额外信息：该公司的数据如下
{extra_information}
AI:
"""


def answer_generation_raw_prompt():
    return PromptTemplate(template=PROMPT, input_variables=["extra_information", "query"])


def answer_generation_prompt(extra_information: str, query: str):
    P = PromptTemplate(template=PROMPT, input_variables=[
                       "extra_information", "query"])
    return P.format(extra_information=extra_information, query=query)


if __name__ == "__main__":
    print(answer_generation_prompt("你们公司的装箱算法可以用在服装业吗"))
