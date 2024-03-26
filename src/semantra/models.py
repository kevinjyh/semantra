import os
from abc import ABC, abstractmethod

import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import tiktoken
import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
from typing import List, Iterable, Union

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

minilm_model_name = "sentence-transformers/all-MiniLM-L6-v2"
mpnet_model_name = "sentence-transformers/all-mpnet-base-v2"
sgpt_model_name = "Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit"
sgpt_1_3B_model_name = "Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit"


def mean_pooling(model_output, attention_mask):
    """
    這段程式碼定義了一個名為 mean_pooling 的函數，該函數用於對模型輸出進行平均池化操作。

    該函數接受兩個參數：model_output 和 attention_mask。model_output 是模型的輸出，attention_mask 是一個用於指示模型應該關注哪些輸入元素的遮罩。

    在函數內部，首先從 model_output 中獲取所有的 token embeddings，然後將 attention_mask 擴展到與 token_embeddings 相同的大小。

    接著，將 token_embeddings 和 input_mask_expanded 進行元素級別的乘法操作，然後沿著第一維度（即每個序列的長度）進行求和操作，得到 sum_embeddings。

    然後，將 input_mask_expanded 沿著第一維度進行求和操作，得到 sum_mask。在這裡，使用了 torch.clamp 函數來確保 sum_mask 中的每個元素都至少為 1e-9，以防止在後續的除法操作中出現除以零的情況。

    最後，將 sum_embeddings 除以 sum_mask，得到平均池化後的結果。

    這個函數的主要目的是將模型的輸出進行平均池化操作，以獲得每個輸入序列的固定長度表示。這種操作在處理變長輸入序列時非常有用，例如在自然語言處理任務中。
    """
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def filter_none(x):
    return [i for i in x if i is not None]


def as_numpy(x):
    # If x is a tensor, convert it to a numpy array
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x


class BaseModel(ABC):
    """
    這類別定義了一個名為 BaseModel 的抽象基礎類別。這個類別定義了一些抽象方法，這些方法在具體的子類別中需要被實現。這些方法包括：

    get_num_dimensions：返回模型的維度數量。
    get_tokens：將給定的文本轉換為 tokens。
    get_token_length：返回給定 tokens 的長度。
    get_text_chunks：將給定的文本和 tokens 轉換為文本片段。
    get_config：返回模型的配置信息。
    embed：將給定的 tokens 和 offsets 轉換為嵌入向量。
    此外，這個類別還定義了一些具體方法，這些方法在子類別中可以直接使用，也可以根據需要進行覆寫。這些方法包括：

    embed_document：將給定的文檔轉換為嵌入向量。
    embed_query：將給定的查詢轉換為嵌入向量。
    embed_queries：將給定的多個查詢轉換為嵌入向量，並將這些嵌入向量進行加權求和。
    embed_queries_and_preferences：將給定的查詢和偏好轉換為嵌入向量，並將這些嵌入向量進行加權求和。
    is_asymmetric：返回模型是否為非對稱的。
    這個類別的主要目的是提供一個模型的基礎結構，以便在具體的子類別中實現特定的功能。
    """
    @abstractmethod
    def get_num_dimensions(self) -> int:
        ...

    @abstractmethod
    def get_tokens(self, text: str):
        ...

    @abstractmethod
    def get_token_length(self, tokens) -> int:
        ...

    @abstractmethod
    def get_text_chunks(self, text: str, tokens) -> "list[str]":
        ...

    @abstractmethod
    def get_config(self):
        ...

    @abstractmethod
    def embed(self, tokens, offsets, is_query: bool = False) -> "list[list[float]]":
        ...

    def embed_document(self, document) -> "list[float]":
        """
        這段程式碼是在一個類別中定義的 embed_document 方法。這個方法的目的是將一個文檔轉換成一個嵌入向量。以下是該方法的步驟：

        使用 get_tokens 方法將文檔分割成 tokens。這些 tokens 通常是單詞或者詞組。

        使用 embed 方法將這些 tokens 轉換成嵌入向量。這個方法需要 tokens 和一個包含每個 token 長度的元組列表作為輸入。在這裡，我們只有一個 token，所以我們傳入一個元組列表，其中只有一個元組，該元組的第一個元素是 0（起始位置），第二個元素是 token 的長度。

        embed 方法返回一個嵌入向量的列表，我們只需要列表中的第一個元素，所以我們使用 [0] 來獲取它。

        這個方法的返回值是一個浮點數列表，代表文檔的嵌入向量。
        """
        tokens = self.get_tokens(document)
        return self.embed(tokens, [(0, self.get_token_length(tokens))], False)[0]

    def embed_query(self, query: str) -> "list[float]":
        """
        這段程式碼是在一個類別中定義的 embed_query 方法。這個方法的目的是將一個查詢轉換成一個嵌入向量。以下是該方法的步驟：

        使用 get_tokens 方法將查詢分割成 tokens。這些 tokens 通常是單詞或者詞組。

        使用 embed 方法將這些 tokens 轉換成嵌入向量。這個方法需要 tokens 和一個包含每個 token 長度的元組列表作為輸入。在這裡，我們只有一個 token，所以我們傳入一個元組列表，其中只有一個元組，該元組的第一個元素是 0（起始位置），第二個元素是 token 的長度。

        embed 方法返回一個嵌入向量的列表，我們只需要列表中的第一個元素，所以我們使用 [0] 來獲取它。

        這個方法的返回值是一個浮點數列表，代表查詢的嵌入向量。
        """
        tokens = self.get_tokens(query)
        return self.embed(tokens, [(0, self.get_token_length(tokens))], True)[0]

    def embed_queries(self, queries) -> "list[float]":
        """
        這段程式碼是在一個類別中定義的 embed_queries 方法。這個方法的目的是將一個查詢列表轉換成一個嵌入向量的列表，並將這些嵌入向量加總。以下是該方法的步驟：

        對於查詢列表中的每一個查詢，我們使用 embed_query 方法將查詢轉換成嵌入向量，然後將這個嵌入向量乘以查詢的權重。這個步驟的結果是一個嵌入向量的列表，其中每個嵌入向量都已經被相應的查詢權重調整過。

        使用 numpy 的 sum 函數將所有的嵌入向量加總。這個函數會將所有的嵌入向量沿著第一個軸（即，列表中的每個元素）加總。結果是一個嵌入向量，其元素是所有嵌入向量相應元素的總和。

        這個方法的返回值是一個浮點數列表，代表所有查詢的加權嵌入向量的總和。
        """
        all_embeddings = [
            as_numpy(self.embed_query(query["query"])) * query["weight"]
            for query in queries
        ]
        # Return sum of embeddings
        return np.sum(all_embeddings, axis=0)

    def embed_queries_and_preferences(self, queries, preferences, documents):
        """
        這段程式碼是在一個類別中定義的 embed_queries_and_preferences 方法。這個方法的目的是將查詢和偏好轉換成嵌入向量，並將這些嵌入向量加總。以下是該方法的步驟：

        如果查詢列表不為空，我們使用 embed_queries 方法將查詢列表轉換成嵌入向量，否則設置查詢嵌入向量為 None。

        對於偏好列表中的每一個偏好，我們從文件中獲取相應的嵌入向量，並將這個嵌入向量乘以偏好的權重。

        使用 numpy 的 sum 函數將查詢嵌入向量和所有偏好的嵌入向量加總。這個函數會將所有的嵌入向量沿著第一個軸（即，列表中的每個元素）加總。結果是一個嵌入向量，其元素是所有嵌入向量相應元素的總和。

        這個方法的返回值是一個浮點數列表，代表所有查詢和偏好的加權嵌入向量的總和。
        """
        query_embedding = self.embed_queries(queries) if len(queries) > 0 else None
        # Add preferences to embeddings
        return np.sum(
            [
                *([query_embedding] if query_embedding is not None else []),
                *[
                    documents[pref["file"]["filename"]].embeddings[
                        pref["searchResult"]["index"]
                    ]
                    * pref["weight"]
                    for pref in preferences
                ],
            ],
            axis=0,
        )

    def is_asymmetric(self):
        return False


class OpenAIModel(BaseModel):
    def __init__(
        self,
        model_name="text-embedding-3-small",
        num_dimensions=1536,
        tokenizer_name="cl100k_base",
    ):
        # Check if OpenAI API key is set
        if "OPENAI_API_KEY" not in os.environ:
            raise Exception(
                "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable or create a `.env` file with the key in the current working directory or the Semantra directory, which is revealed by running `semantra --show-semantra-dir`."
            )
        

        self.model_name = model_name
        self.num_dimensions = num_dimensions
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def get_config(self):
        return {
            "model_type": "openai",
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer.name,
        }

    def get_num_dimensions(self) -> int:
        return self.num_dimensions

    def get_tokens(self, text: str):
        return self.tokenizer.encode(text)

    def get_token_length(self, tokens) -> int:
        return len(tokens)

    def get_text_chunks(self, _: str, tokens) -> "list[str]":
        return [self.tokenizer.decode([token]) for token in tokens]

    def embed(self, tokens, offsets, _is_query=False) -> "list[list[float]]":
        texts = [tokens[i:j] for i, j in offsets] # TODO [:155] 限制長度
        response = client.embeddings.create(model=self.model_name, input=texts)
        return np.array(response.data[0].embedding)

def zero_if_none(x):
    return 0 if x is None else x


class TransformerModel(BaseModel):
    """
    這類別定義了一個名為 TransformerModel 的類別，該類別繼承自 BaseModel。這個類別主要用於處理與 Transformer 模型相關的任務，例如模型的初始化、配置獲取、輸入的分塊處理、輸入的正規化以及嵌入的生成等。

    在 __init__ 方法中，首先檢查是否有可用的 CUDA，然後使用給定的模型名稱從預訓練模型中加載 tokenizer 和模型。接著，對文檔和查詢的前後 token 進行編碼，並將其存儲為類別的屬性。最後，如果 CUDA 可用，則將模型轉移到 GPU 上。

    get_config 方法返回一個包含模型配置的字典，包括模型類型、模型名稱、文檔和查詢的前後 token 以及是否為非對稱模型。

    get_tokens 方法接受一個文本字符串，並使用 tokenizer 將其轉換為 tokens，並返回包含 tokens 和其他相關信息的字典。

    get_text_chunks 方法接受一個文本字符串和 tokens，並根據 tokens 的 offset_mapping 將文本分割為多個 chunks。

    normalize_input_ids 和 normalize_attention_mask 方法用於對輸入的 input_ids 和 attention_mask 進行正規化。如果設定了前後 token，則會在輸入的前後添加對應的 token。

    embed 方法接受 tokens 和 offsets，並生成對應的嵌入。首先，對 input_ids 和 attention_mask 進行正規化和填充，然後將其轉移到 GPU（如果可用）。接著，使用模型生成模型輸出，並對其進行平均池化，得到最終的嵌入。
    """
    def __init__(
        self,
        model_name,
        doc_token_pre=None,
        doc_token_post=None,
        query_token_pre=None,
        query_token_post=None,
        asymmetric=False,
        cuda=None,
    ):
        if cuda is None:
            cuda = torch.cuda.is_available()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Get tokens
        self.pre_post_tokens = [
            doc_token_pre,
            doc_token_post,
            query_token_pre,
            query_token_post,
        ]
        self.doc_token_pre = (
            self.tokenizer.encode(doc_token_pre, add_special_tokens=False)
            if doc_token_pre
            else None
        )
        self.doc_token_post = (
            self.tokenizer.encode(doc_token_post, add_special_tokens=False)
            if doc_token_post
            else None
        )
        self.query_token_pre = (
            self.tokenizer.encode(query_token_pre, add_special_tokens=False)
            if query_token_pre
            else None
        )
        self.query_token_post = (
            self.tokenizer.encode(query_token_post, add_special_tokens=False)
            if query_token_post
            else None
        )

        self.asymmetric = asymmetric

        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()

    def get_config(self):
        return {
            "model_type": "transformers",
            "model_name": self.model_name,
            "doc_token_pre": self.pre_post_tokens[0],
            "doc_token_post": self.pre_post_tokens[1],
            "query_token_pre": self.pre_post_tokens[2],
            "query_token_post": self.pre_post_tokens[3],
            "asymmetric": self.asymmetric,
        }

    def is_asymmetric(self):
        return self.asymmetric

    def get_num_dimensions(self) -> int:
        return int(self.model.config.hidden_size)

    def get_tokens(self, text: str):
        return self.tokenizer(
            text, return_offsets_mapping=True, verbose=False, return_tensors="pt"
        )

    def get_token_length(self, tokens) -> int:
        return len(tokens["input_ids"][0])

    def get_text_chunks(self, text: str, tokens) -> "list[str]":
        """
        這段程式碼定義了一個名為 get_text_chunks 的方法，該方法接受一個文本字符串和 tokens，並根據 tokens 的 offset_mapping 將文本分割為多個 chunks。

        首先，從 tokens 中獲取 offset_mapping，並將其存儲在變量 offsets 中。然後，初始化一個空的列表 chunks 用於存儲文本的 chunks。

        接著，對 offsets 中的每一對偏移量 i 和 j 進行迭代。如果 i 和 j 相等，則將 new_i 設為 prev_j，否則將其設為 i。如果 prev_i 不為 None，則將文本從 prev_i 到 new_i 的部分添加到 chunks 中。如果 prev_i 為 None，則將其設為 0，否則如果 new_i 大於 prev_i，則將 prev_i 設為 new_i。同樣，如果 prev_j 為 None，則將其設為 j，否則如果 j 大於 prev_j，則將 prev_j 設為 j。

        最後，將文本從 prev_i 到結尾的部分添加到 chunks 中，並返回 chunks。如果 prev_i 為 None，則從文本的開頭開始。
        """
        offsets = tokens["offset_mapping"][0]
        chunks = []
        prev_i = None
        prev_j = None
        for i, j in offsets:
            new_i = prev_j if i == j else i
            if prev_i is not None:
                chunks.append(text[prev_i:new_i])
            if prev_i is None:
                prev_i = 0
            elif new_i > prev_i:
                prev_i = new_i
            if prev_j is None:
                prev_j = j
            elif j > prev_j:
                prev_j = j
        chunks.append(text[0 if prev_i is None else prev_i :])
        return chunks

    def normalize_input_ids(self, input_ids, is_query):
        """
        這段程式碼是一個名為 normalize_input_ids 的方法，它的目的是對輸入的識別碼（input_ids）進行正規化。這個方法會根據 is_query 的值來決定使用哪種方式來正規化識別碼。

        如果 self.query_token_pre 和 self.query_token_post 都為 None，則直接返回 input_ids。

        否則，會根據 is_query 的值來選擇 token_pre 和 token_post。如果 is_query 為 True，則選擇 self.query_token_pre 和 self.query_token_post；否則選擇 self.doc_token_pre 和 self.doc_token_post。

        接著，使用 filter_none 函數來過濾掉 None 的元素，並將 token_pre、input_ids 和 token_post 進行連接。這裡使用了 torch.cat 函數來連接張量，並使用 torch.tensor 函數來將 token_pre 和 token_post 轉換為張量。

        這段程式碼的主要目的是將前後標記（如果存在的話）添加到輸入識別碼的前後，並將結果作為一個新的張量返回。
        """
        if self.query_token_pre is None and self.query_token_post is None:
            return input_ids
        else:
            token_pre = self.query_token_pre if is_query else self.doc_token_pre
            token_post = self.query_token_post if is_query else self.doc_token_post
            return torch.cat(
                filter_none(
                    [
                        torch.tensor(token_pre) if token_pre is not None else None,
                        input_ids,
                        torch.tensor(token_post) if token_post is not None else None,
                    ]
                )
            )

    def normalize_attention_mask(self, attention_mask, is_query):
        """
        這段程式碼是一個名為 normalize_attention_mask 的方法，它的目的是對輸入的注意力遮罩（attention_mask）進行正規化。這個方法會根據 is_query 的值來決定使用哪種方式來正規化注意力遮罩。

        如果 self.query_token_pre 和 self.query_token_post 都為 None，則直接返回 attention_mask。

        否則，會根據 is_query 的值來選擇 token_pre 和 token_post。如果 is_query 為 True，則選擇 self.query_token_pre 和 self.query_token_post；否則選擇 self.doc_token_pre 和 self.doc_token_post。

        接著，使用 filter_none 函數來過濾掉 None 的元素，並將 token_pre、attention_mask 和 token_post 進行連接。這裡使用了 torch.cat 函數來連接張量，並使用 torch.ones 函數來創建一個所有元素都為 1 的張量。

        這段程式碼的主要目的是將前後標記（如果存在的話）添加到注意力遮罩的前後，並將結果作為一個新的張量返回。
        """
        if self.query_token_pre is None and self.query_token_post is None:
            return attention_mask
        else:
            token_pre = self.query_token_pre if is_query else self.doc_token_pre
            token_post = self.query_token_post if is_query else self.doc_token_post
            return torch.cat(
                filter_none(
                    [
                        torch.ones(len(token_pre)) if token_pre is not None else None,
                        attention_mask,
                        torch.ones(len(token_post)) if token_post is not None else None,
                    ]
                )
            )

    def embed(self, tokens, offsets, is_query=False) -> "list[list[float]]":
        """
        這段程式碼是一個名為 embed 的方法，它的目的是將輸入的 tokens 和 offsets 進行嵌入，並返回一個浮點數列表的列表。

        首先，它使用 torch.nn.utils.rnn.pad_sequence 函數對 input_ids 進行填充。這裡的 input_ids 是由 normalize_input_ids 方法返回的，該方法將 tokens 的 input_ids 和 is_query 作為輸入。填充的值是 self.tokenizer.pad_token_id，如果該值為 None，則使用 zero_if_none 函數將其轉換為 0。

        接著，它對 attention_mask 進行了相同的操作，但填充的值固定為 0。

        如果 self.cuda 為 True，則將 input_ids 和 attention_mask 移到 GPU 上。

        然後，它使用 self.model 對 input_ids 和 attention_mask 進行處理，並將結果存儲在 model_output 中。

        最後，它使用 mean_pooling 函數對 model_output 和 attention_mask 進行平均池化，並返回結果。

        這段程式碼的主要目的是將輸入的 tokens 和 offsets 進行嵌入，並返回嵌入的結果。
        """
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [
                self.normalize_input_ids(
                    tokens["input_ids"][0].index_select(0, torch.tensor(range(i, j))),
                    is_query,
                )
                for i, j in offsets
            ],
            batch_first=True,
            padding_value=zero_if_none(self.tokenizer.pad_token_id),
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [
                self.normalize_attention_mask(
                    tokens["attention_mask"][0].index_select(
                        0, torch.tensor(range(i, j))
                    ),
                    is_query,
                )
                for i, j in offsets
            ],
            batch_first=True,
            padding_value=0,
        )
        if self.cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        with torch.no_grad():
            model_output = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        return mean_pooling(model_output, attention_mask)


models = {
    "openai": {
        "cost_per_token": 0.0004 / 1000,
        "pool_size": 50000,
        "pool_count": 2000,
        "get_model": lambda: OpenAIModel(
            model_name="text-embedding-3-small",
            num_dimensions=1536,
            tokenizer_name="cl100k_base",
        ),
    },
    "minilm": {
        "cost_per_token": None,
        "pool_size": 50000,
        "get_model": lambda: TransformerModel(model_name=minilm_model_name),
    },
    "mpnet": {
        "cost_per_token": None,
        "pool_size": 15000,
        "get_model": lambda: TransformerModel(model_name=mpnet_model_name),
    },
    "sgpt": {
        "cost_per_token": None,
        "pool_size": 10000,
        "get_model": lambda: TransformerModel(
            model_name=sgpt_model_name,
            query_token_pre="[",
            query_token_post="]",
            doc_token_pre="{",
            doc_token_post="}",
            asymmetric=True,
        ),
    },
    "sgpt-1.3B": {
        "cost_per_token": None,
        "pool_size": 1000,
        "get_model": lambda: TransformerModel(
            model_name=sgpt_1_3B_model_name,
            query_token_pre="[",
            query_token_post="]",
            doc_token_pre="{",
            doc_token_post="}",
            asymmetric=True,
        ),
    },
}
