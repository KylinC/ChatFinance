import pandas as pd
import numpy as np
import datasets


def csv2parquet(csv_file_path,parquet_train_file_path,parquet_test_file_path):
    df = pd.read_csv(csv_file_path)

    ################################
    #     label       review
    #       1           绝了
    #       0           不行
    #       ....
    #
    #################################

    df['tag'] = df['label'].map({0:'差评',1:'好评'}) 
    df = df.rename({'review':'text'},axis = 1)
    ds_dic = datasets.Dataset.from_pandas(df).train_test_split(
        test_size = 0.2,shuffle=True, seed = 43)
    dftrain = ds_dic['train'].to_pandas() 
    dftest = ds_dic['test'].to_pandas()
    dftrain.to_parquet(parquet_train_file_path)
    dftest.to_parquet(parquet_test_file_path)


if __name__ == '__main__':
    csv_file_path = "/home/kylin/workspace/ChatFinance/data/chatglm_llm_fintech_raw_dataset/intent_10k.csv"
    parquet_train_file_path = "/home/kylin/workspace/ChatFinance/data/sft/intent_sft_10k.parquet"
    parquet_test_file_path = "/home/kylin/workspace/ChatFinance/data/sft/intent_sft_10k_val.parquet"
    csv2parquet(csv_file_path,parquet_train_file_path,parquet_test_file_path)
    
    