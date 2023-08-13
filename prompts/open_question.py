from langchain import PromptTemplate

PROMPT = """
你需要扮演一位金融专家助手。请根据你的专业知识，回答下列问题。请注意，你只需使用与问题直接相关的部分。

要求：
1. 答案应简练、清晰、准确。
2. 仅使用与问题直接相关的额外信息进行回答。
3. 避免引入与问题无关的信息。

示例：
人类：什么是营业税金及附加？
AI: 营业税金及附加是指对企业或个人因经营活动所产生的收入或销售额征收的税费，以及可能与之相关的其他费用或附加费。

现在开始：

人类：{query}
AI:
"""

def open_question_prompt(query: str):
    P = PromptTemplate(template=PROMPT, input_variables=["query"])
    return P.format(query=query)


if __name__ == "__main__":
    print(open_question_prompt("你们公司的装箱算法可以用在服装业吗"))