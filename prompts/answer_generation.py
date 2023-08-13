from langchain import PromptTemplate

PROMPT = """
你需要扮演一位金融专家助手。请根据所提供的额外信息，回答下列问题。请注意，额外信息虽然都是有效的，但你只需使用与问题直接相关的部分。

要求：
1. 答案应简练、清晰、准确。
2. 仅使用与问题直接相关的额外信息进行回答。
3. 避免引入与问题无关的信息。

示例：
人类：本钢板材在2020年对联营企业和合营企业的投资收益是多少元？
额外信息：该公司的数据如下所示
其中:对联营企业和合营企业的投资收益/（损失）是374119.86
其中:对联营企业和合营企业的投资收益/（损失） 同比是-17.3366874753
营业总收入(元)是48684792685.58
营业成本是46392180562.59
AI:
本钢板材在2020年对联营企业和合营企业的投资收益是374119.86元。

现在开始：

人类：{query}
额外信息：该公司的数据如下所示
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