from langchain import PromptTemplate

PROMPT = """
你需要扮演一个优秀的文本相关性评估助手。你需要评估额外信息是否有助于提供更优质和简练的回答。

你不需要做任何解释说明，并且严格按照示例的格式进行输出，仅回答["是", "否"]

以下是一个示例：
人类：我有一个服装厂，是否可以应用你们的装箱算法改善装载率呢？
额外信息：问：能否介绍一下蓝胖子机器智能的主力产品？ 答：蓝胖子机器智能的主力产品是“蓝胖智汇Doraopt”系列AI软件产品及解决方案。这是由我们的AIoT产品事业部打造的，用于提供智能供应链的整体解决方案。
AI:否

现在开始：
人类：{query}
额外信息：{extra_information}
AI:
"""


def relevance_scoring_raw_prompt():
    return PromptTemplate(template=PROMPT, input_variables=["query", "extra_information"])


def relevance_scoring_prompt(query: str, extra_information: str):
    P = PromptTemplate(template=PROMPT, input_variables=[
                       "query", "extra_information"])
    return P.format(query=query, extra_information=extra_information)


if __name__ == "__main__":
    print(relevance_scoring_prompt(
        query="你们的装箱算法能不能用在家居业呀？主要用于是沙发的装箱。",
        extra_information="问：DoraCLP「装满满」适用于哪些行业？ 答：DoraCLP「装满满」可以广泛应用于多个行业，例如家居业和鞋服业等。"),
    )
