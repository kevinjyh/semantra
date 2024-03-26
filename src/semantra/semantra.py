import hashlib
import io
import json
import math
import os

import click
import numpy as np
import pkg_resources
from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, request, send_file, send_from_directory
from tqdm import tqdm

from .models import BaseModel, TransformerModel, as_numpy, models
from .pdf import get_pdf_content
from .util import (
    HASH_LENGTH,
    file_md5,
    get_annoy_filename,
    get_config_filename,
    get_embeddings_filename,
    get_num_annoy_embeddings,
    get_num_embeddings,
    get_offsets,
    get_tokens_filename,
    join_text_chunks,
    load_annoy_db,
    read_embeddings_file,
    sort_results,
    write_annoy_db,
    write_embedding,
)

VERSION = "0.1.8" # pkg_resources.require("semantra")[0].version
DEFAULT_ENCODING = "utf-8"
DEFAULT_PORT = 8080

package_directory = os.path.dirname(os.path.abspath(__file__))


class Content:
    def __init__(self, rawtext, filename):
        self.rawtext = rawtext
        self.filename = filename
        self.filetype = "text"


def get_text_content(md5, filename, semantra_dir, force, silent, encoding):
    if filename.endswith(".pdf"):
        return get_pdf_content(md5, filename, semantra_dir, force, silent)

    with open(filename, "r", encoding=encoding, errors="ignore") as f:
        rawtext = f.read()
        return Content(rawtext, filename)


TRANSFORMER_POOL_DEFAULT = 15000


class Document:
    """
    這是一個名為Document的類別，它用於表示一個文檔。這個類別有以下的屬性和方法：

    __init__：這是初始化方法，用於設定文檔的各種屬性，如檔案名稱、MD5值、Semantra目錄、基礎檔案名稱、配置、嵌入檔案名稱、是否使用Annoy、Annoy檔案名稱、窗口、偏移量、tokens檔案名稱、嵌入維度數和編碼。

    content：這是一個屬性，用於獲取文檔的內容。

    text_chunks：這是一個屬性，用於讀取tokens檔案並將其解析為JSON格式。

    num_embeddings：這是一個屬性，用於獲取嵌入的數量。

    embedding_db：這是一個屬性，如果設定了使用Annoy，則會從Annoy數據庫中讀取嵌入；否則，會引發一個ValueError。

    embeddings：這是一個屬性，用於讀取嵌入檔案並返回嵌入的結果。如果讀取的嵌入數量與預期的嵌入數量不符，則會引發一個斷言錯誤。
    """
    def __init__(
        self,
        filename,
        md5,
        semantra_dir,
        base_filename,
        config,
        embeddings_filenames,
        use_annoy,
        annoy_filenames,
        windows,
        offsets,
        tokens_filename,
        num_dimensions,
        encoding,
    ):
        self.filename = filename
        self.md5 = md5
        self.semantra_dir = semantra_dir
        self.base_filename = base_filename
        self.config = config
        self.embeddings_filenames = embeddings_filenames
        self.use_annoy = use_annoy
        self.annoy_filenames = annoy_filenames
        self.windows = windows
        self.offsets = offsets
        self.tokens_filename = tokens_filename
        self.num_dimensions = num_dimensions
        self.encoding = encoding

    @property
    def content(self):
        return get_text_content(
            self.md5, self.filename, self.semantra_dir, False, True, self.encoding
        )

    @property
    def text_chunks(self):
        with open(self.tokens_filename, "r") as f:
            return json.loads(f.read())

    @property
    def num_embeddings(self):
        return len(self.offsets[0])

    @property
    def embedding_db(self):
        if not self.use_annoy:
            raise ValueError("Embeddings are not stored in Annoy database")
        return load_annoy_db(self.annoy_filenames[0], self.num_dimensions)

    @property
    def embeddings(self):
        results, embedding_count = read_embeddings_file(
            self.embeddings_filenames[0],
            self.num_dimensions,
            self.num_embeddings,
        )
        assert embedding_count == self.num_embeddings
        return results


def process(
    filename,
    semantra_dir,
    model,
    num_dimensions,
    use_annoy,
    num_annoy_trees,
    windows,
    cost_per_token,
    pool_count,
    pool_size,
    force,
    silent,
    no_confirm,
    encoding,
):
    """
    處理文本內容並生成相應的詞嵌入（embeddings）的。首先，它會檢查指定的目錄是否存在，如果不存在，則會創建該目錄。然後，它會獲取文件的MD5值和配置信息，並根據這些信息生成一些文件名。

    如果強制重新計算或者詞標記（tokens）文件不存在，則會從原始文本中提取詞標記。這些詞標記會被用來生成文本塊（text chunks），並將其寫入到文件中。如果詞標記文件已經存在，則會直接從文件中讀取文本塊。

    接著，程式碼會根據配置參數來獲取嵌入偏移量（embedding offsets）。然後，它會生成一個完整的配置，包含了原始配置以及一些額外的詳細信息。

    如果強制重新計算或者配置文件不存在，並且每個詞標記的成本不為None，則會提示用戶詞嵌入的成本。無論如何，都會將完整的配置寫入到配置文件中。

    然後，程式碼會計算詞嵌入。如果詞嵌入文件已經存在，並且不需要使用Annoy或者Annoy文件也存在，則會跳過該步驟。否則，會從文本塊中提取詞標記，並生成詞嵌入。這些詞嵌入會被寫入到文件中，並且如果需要使用Annoy，則還會寫入到Annoy數據庫中。

    最後，程式碼會返回一個Document對象，該對象包含了所有的配置信息、文件名、詞嵌入等信息。

    在這段程式碼中，還定義了一個Document類，該類包含了一些屬性，如文件名、MD5值、配置信息、詞嵌入文件名等。此外，該類還定義了一些屬性方法，用於獲取文本內容、文本塊、詞嵌入數量、詞嵌入數據庫和詞嵌入等。
    """

    # Check if semantra dir exists
    if not os.path.exists(semantra_dir):
        os.makedirs(semantra_dir)

    # Get the md5 and config
    md5 = file_md5(filename)
    base_filename = os.path.basename(filename)
    config = model.get_config()
    if encoding != DEFAULT_ENCODING:
        config["encoding"] = encoding
    config_hash = hashlib.shake_256(json.dumps(config).encode()).hexdigest(HASH_LENGTH)

    # File names
    tokens_filename = os.path.join(semantra_dir, get_tokens_filename(md5, config_hash))
    config_filename = os.path.join(semantra_dir, get_config_filename(md5, config_hash))

    should_calculate_tokens = True
    if force or not os.path.exists(tokens_filename):
        # Calculate tokens to get text chunks
        content = get_text_content(md5, filename, semantra_dir, force, silent, encoding)
        text = content.rawtext
        tokens = model.get_tokens(text)
        should_calculate_tokens = False
        text_chunks = model.get_text_chunks(text, tokens)
        with open(tokens_filename, "w") as f:
            f.write(json.dumps(text_chunks))
    else:
        with open(tokens_filename, "r") as f:
            text_chunks = json.loads(f.read())
    num_tokens = len(text_chunks)

    # Get embedding offsets based on config parameters
    (
        offsets,
        num_embedding_tokens,
    ) = get_offsets(num_tokens, windows)

    # Full config contains additional details
    full_config = {
        **config,
        "filename": filename,
        "md5": md5,
        "base_filename": base_filename,
        "num_dimensions": num_dimensions,
        "cost_per_token": cost_per_token,
        "windows": windows,
        "num_tokens": num_tokens,
        "num_embeddings": len(offsets),
        "num_embedding_tokens": num_embedding_tokens,
        "use_annoy": use_annoy,
        "num_annoy_trees": num_annoy_trees,
        "semantra_version": VERSION,
    }

    if force or not os.path.exists(config_filename):
        if cost_per_token is not None and not no_confirm:
            click.confirm(
                f"Tokens will cost ${num_embedding_tokens * cost_per_token:.2f}. Proceed?",
                abort=True,
            )

    # Write out the config every time
    with open(config_filename, "w") as f:
        f.write(json.dumps(full_config))

    embeddings_filenames = []
    annoy_filenames = []
    with tqdm(
        total=num_embedding_tokens,
        desc="Calculating embeddings",
        leave=False,
        disable=silent,
    ) as pbar:
        for (size, offset, rewind), sub_offsets in zip(windows, offsets):
            embeddings_filename = os.path.join(
                semantra_dir,
                get_embeddings_filename(md5, config_hash, size, offset, rewind),
            )
            annoy_filename = os.path.join(
                semantra_dir,
                get_annoy_filename(
                    md5, config_hash, size, offset, rewind, num_annoy_trees
                ),
            )
            embeddings_filenames.append(embeddings_filename)
            annoy_filenames.append(annoy_filename)

            if os.path.exists(embeddings_filename) and (
                not use_annoy or os.path.exists(annoy_filename)
            ):
                num_embeddings = get_num_embeddings(embeddings_filename, num_dimensions)
                if use_annoy:
                    num_annoy_embeddings = get_num_annoy_embeddings(
                        annoy_filename, num_dimensions
                    )

                if (
                    not force
                    and num_embeddings == len(sub_offsets)
                    and (not use_annoy or num_annoy_embeddings == len(sub_offsets))
                ):
                    # Embedding is fully calculated
                    continue

            if should_calculate_tokens:
                tokens = model.get_tokens(join_text_chunks(text_chunks))
                should_calculate_tokens = False

            # Read embeddings if they exist
            embedding_index = 0
            if not force and os.path.exists(embeddings_filename):
                embeddings, embedding_index = read_embeddings_file(
                    embeddings_filename, num_dimensions, len(sub_offsets)
                )
            else:
                embeddings = np.empty(
                    (len(sub_offsets), num_dimensions), dtype=np.float32
                )
                embedding_index = 0

            num_skip = embedding_index
            iteration = 0

            # Write embeddings
            pool = []
            pool_token_count = 0

            with open(embeddings_filename, "ab") as f:

                def flush_pool():
                    """
                    這段程式碼的主要功能是將一組待處理的詞彙（pool）通過模型進行嵌入，並將結果寫入文件。以下是詳細的步驟：

                        1.檢查詞彙池（pool）是否有內容。如果有，則進行下一步；如果沒有，則不進行任何操作。

                        2.使用模型對詞彙池中的詞彙進行嵌入，得到嵌入結果（embedding_results）。

                        3.檢查嵌入結果是否有"cpu"方法。如果有，則調用該方法將嵌入結果轉移到CPU。

                        4.將嵌入結果存儲到嵌入列表（embeddings）中的適當位置。這裡的位置由嵌入索引（embedding_index）和詞彙池的大小（len(pool)）決定。

                        5.遍歷嵌入結果，對每個嵌入調用write_embedding函數，將其寫入文件。

                        6.更新嵌入索引，將其增加詞彙池的大小。

                        7.清空詞彙池，並將詞彙池的詞彙數量（pool_token_count）重置為0。

                    這段程式碼的主要目的是將一組詞彙進行嵌入，並將嵌入結果寫入文件，以便後續的處理和分析。
                    """
                    nonlocal pool, pool_token_count, embeddings, embedding_index, f

                    if len(pool) > 0:
                        embedding_results = model.embed(tokens, pool)
                        # Call .cpu if embedding_results contains it
                        if hasattr(embedding_results, "cpu"):
                            embedding_results = embedding_results.cpu()
                        embeddings[
                            embedding_index : embedding_index + len(pool)
                        ] = embedding_results
                        for embedding in embedding_results:
                            write_embedding(f, embedding, num_dimensions)
                        embedding_index += len(pool)
                        pool = []
                        pool_token_count = 0

                for offset in sub_offsets:
                    size = offset[1] - offset[0]

                    # Skip if already calculated
                    if iteration < num_skip:
                        iteration += 1
                        pbar.update(size)
                        continue

                    window_text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                    if len(window_text) == 0:
                        pbar.update(size)
                        continue

                    pool.append(offset)
                    pool_token_count += size
                    if (
                        pool_count is not None and len(pool) >= pool_count
                    ) or pool_token_count >= pool_size:
                        flush_pool()
                    pbar.update(size)

                flush_pool()

            # Write embeddings db
            if use_annoy:
                write_annoy_db(
                    filename=annoy_filename,
                    num_dimensions=num_dimensions,
                    embeddings=embeddings,
                    num_trees=num_annoy_trees,
                )

    return Document(
        filename=filename,
        md5=md5,
        semantra_dir=semantra_dir,
        base_filename=base_filename,
        config=full_config,
        embeddings_filenames=embeddings_filenames,
        use_annoy=use_annoy,
        annoy_filenames=annoy_filenames,
        windows=windows,
        offsets=offsets,
        tokens_filename=tokens_filename,
        num_dimensions=num_dimensions,
        encoding=encoding,
    )


def process_windows(windows: str) -> "list[tuple[int, int, int]]":
    """
    這是一個名為 process_windows 的函數，它接受一個名為 windows 的字符串參數，並返回一個元組列表。每個元組包含三個整數。

    函數的工作流程如下：

    首先，它會將 windows 字符串按照逗號 (",") 分割，得到一個窗口列表。

    然後，對於列表中的每個窗口，它會檢查窗口是否包含下劃線 ("_")。

    如果窗口包含一個下劃線，那麼它會將窗口按照下劃線分割，並將分割後的兩個部分分別賦值給 size 和 offset。此時，rewind 被設定為 0。

    如果窗口包含兩個下劃線，那麼它會將窗口按照下劃線分割，並將分割後的三個部分分別賦值給 size、offset 和 rewind。

    如果窗口不包含下劃線，那麼它會將窗口的值賦值給 size，並將 offset 和 rewind 都設定為 0。

    最後，函數會生成一個元組，包含 size、offset 和 rewind 這三個整數，並將這個元組添加到返回的列表中。

    這個函數的主要用途是解析 windows 字符串，並將其轉換為一個元組列表，以便於後續的處理。
    """
    for window in windows.split(","):
        if "_" in window:
            # One or two occurrences?
            if window.count("_") == 1:
                size, offset = window.split("_")
                rewind = 0
            else:
                size, offset, rewind = window.split("_")
            yield int(size), int(offset), int(rewind)
        else:
            yield int(window), 0, 0


@click.command()
@click.argument("filename", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--model",
    type=click.Choice(models.keys(), case_sensitive=True),
    default="mpnet",
    show_default=True,
    help="Preset model to use for embedding",
)
@click.option(
    "--encoding",
    type=str,
    default=DEFAULT_ENCODING,
    show_default=True,
    help="Encoding to use for reading text files",
)
@click.option(
    "--transformer-model",
    type=str,
    help="Custom Huggingface transformers model name to use for embedding",
)
@click.option(
    "--windows",
    type=str,
    default="128_0_16",
    show_default=True,
    help='Embedding windows to extract. A comma-separated list of the format "size[_offset=0][_rewind=0]. A window with size 128, offset 0, and rewind of 16 (128_0_16) will embed the document in chunks of 128 tokens which partially overlap by 16. Only the first window is used for search.',
)
@click.option(
    "--no-server",
    is_flag=True,
    default=False,
    show_default=True,
    help="Do not start the UI server (only process)",
)
@click.option(
    "--port",
    type=int,
    default=DEFAULT_PORT,
    show_default=True,
    help="Port to use for embedding server",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="Host to use for embedding server. Set to 0.0.0.0 to make the server available externally.",
)
@click.option(
    "--pool-size",
    type=int,
    default=None,
    help="Max number of embedding tokens to pool together in requests",
)
@click.option(
    "--pool-count",
    type=int,
    default=None,
    help="Max number of embeddings to pool together in requests",
)
@click.option(
    "--doc-token-pre",
    type=str,
    default=None,
    help="Token to prepend to each document in transformer models (default: None)",
)
@click.option(
    "--doc-token-post",
    type=str,
    default=None,
    help="Token to append to each document in transformer models (default: None)",
)
@click.option(
    "--query-token-pre",
    type=str,
    default=None,
    help="Token to prepend to each query in transformer models (default: None)",
)
@click.option(
    "--query-token-post",
    type=str,
    default=None,
    help="Token to append to each query in transformer models (default: None)",
)
@click.option(
    "--num-results",
    type=int,
    default=10,
    show_default=True,
    help="Number of results (neighbors) to retrieve per file for queries",
)
@click.option(
    "--annoy",
    is_flag=True,
    default=True,
    show_default=True,
    help="Use approximate kNN via Annoy for queries (faster querying at a slight cost of accuracy); if false, use exact exhaustive kNN",
)
@click.option(
    "--num-annoy-trees",
    type=int,
    default=100,
    show_default=True,
    help="Number of trees to use for approximate kNN via Annoy",
)
@click.option(
    "--svm",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use SVM instead of any kind of kNN for queries (slower and only works on symmetric models)",
)
@click.option(
    "--svm-c",
    type=float,
    default=1.0,
    show_default=True,
    help="SVM regularization parameter; higher values penalize mispredictions more",
)
@click.option(
    "--explain-split-count",
    type=int,
    default=9,
    show_default=True,
    help="Number of splits on a given window to use for explaining a query",
)
@click.option(
    "--explain-split-divide",
    type=int,
    default=6,
    show_default=True,
    help="Factor to divide the window size by to get each split length for explaining a query",
)
@click.option(
    "--num-explain-highlights",
    type=int,
    default=2,
    show_default=True,
    help="Number of split results to highlight for explaining a query",
)
@click.option(
    "--force", is_flag=True, default=False, help="Force process even if cached"
)
@click.option(
    "--silent",
    is_flag=True,
    default=False,
    help="Do not print progress information",
)
@click.option(
    "--no-confirm",
    is_flag=True,
    default=False,
    help="Do not show cost and ask for confirmation before processing with OpenAI",
)
@click.option(
    "--version",
    is_flag=True,
    default=False,
    help="Print version and exit",
)
@click.option(
    "--list-models",
    is_flag=True,
    default=False,
    help="List preset models and exit",
)
@click.option(
    "--show-semantra-dir",
    is_flag=True,
    default=False,
    help="Print the directory semantra will use to store processed files and exit",
)
@click.option(
    "--semantra-dir",
    type=click.Path(exists=False),
    default=None,
    help="Directory to store semantra files in",
)
def main(
    filename,
    windows="128_0_16",
    no_server=False,
    port=8080,
    host="127.0.0.1",
    pool_size=None,
    pool_count=None,
    doc_token_pre=None,
    doc_token_post=None,
    query_token_pre=None,
    query_token_post=None,
    model="mpnet",
    transformer_model=None,
    encoding=DEFAULT_ENCODING,
    num_annoy_trees=100,
    num_results=10,
    annoy=True,
    svm=False,
    svm_c=1.0,
    explain_split_count=9,
    explain_split_divide=6,
    num_explain_highlights=2,
    force=False,
    silent=False,
    no_confirm=False,
    version=False,
    list_models=False,
    show_semantra_dir=False,
    semantra_dir=None,  # auto
):
    """
    這段程式碼是 Python 語言中的一個 main 函數，它是一個命令行工具的主要入口點。該函數接受許多參數，包括文件名、窗口大小、服務器設置、模型選擇等等。

    在函數的開始，它會檢查 version 和 list_models 參數，如果設置了這些參數，它將打印版本信息或模型列表，然後返回。

    接著，它會檢查 semantra_dir 參數，如果未設置，則會使用 click.get_app_dir("Semantra") 獲取應用程序目錄。如果設置了 show_semantra_dir 參數，它將打印 semantra_dir 的值並返回。

    然後，它會從 .env 文件中加載環境變量，並檢查是否提供了 filename 參數。如果沒有提供，則會引發一個錯誤。

    接下來，它會處理窗口大小，並根據 transformer_model 參數的設置來選擇模型。如果設置了 transformer_model，則會使用自定義的轉換器模型，否則，會使用預設的模型。

    然後，它會檢查模型是否與 SVM 兼容，並處理文件名列表，對每個文件進行處理。

    最後，它會啟動一個 Flask 服務器，並定義了幾個路由，包括基本路由、靜態文件路由和 API 路由。

    這個函數的主要目的是處理文件，並在處理後啟動一個服務器，以便用戶可以通過 API 獲取處理結果。
    """
    if version:
        print(VERSION)
        return

    if list_models:
        for model_name in models:
            print(f"- {model_name}")
        return

    if semantra_dir is None:
        semantra_dir = click.get_app_dir("Semantra")

    if show_semantra_dir:
        print(semantra_dir)
        return
    
    # Load environment from Semantra dir
    env_path = os.path.join(semantra_dir, ".env")
    load_dotenv(env_path)

    if filename is None or len(filename) == 0:
        raise click.UsageError("Must provide a filename to process/query")

    processed_windows = list(process_windows(windows))

    if transformer_model is not None:
        # Handle custom transformers model
        if pool_size is None:
            pool_size = TRANSFORMER_POOL_DEFAULT

        cost_per_token = None
        model = TransformerModel(
            transformer_model,
            doc_token_pre=doc_token_pre,
            doc_token_post=doc_token_post,
            query_token_pre=query_token_pre,
            query_token_post=query_token_post,
        )
    else:
        # Pull preset model
        model_config = models[model]
        cost_per_token = model_config["cost_per_token"]
        if pool_size is None:
            pool_size = model_config["pool_size"]
        if pool_count is None:
            pool_count = model_config.get("pool_count", None)
        model: BaseModel = model_config["get_model"]()

    # Check if model is compatible
    if svm and model.is_asymmetric():
        raise ValueError(
            "SVM is not compatible with asymmetric models. "
            "Please use a symmetric model or kNN."
        )

    documents = {}
    pbar = tqdm(filename, disable=silent)
    for fn in pbar:
        pbar.set_description(f"{os.path.basename(fn)}")
        documents[fn] = process(
            filename=fn,
            semantra_dir=semantra_dir,
            model=model,
            num_dimensions=model.get_num_dimensions(),
            use_annoy=annoy,
            num_annoy_trees=num_annoy_trees,
            windows=processed_windows,
            cost_per_token=cost_per_token,
            pool_count=pool_count,
            pool_size=pool_size,
            force=force,
            silent=silent,
            no_confirm=no_confirm,
            encoding=encoding,
        )

    cached_content = None
    cached_content_filename = None

    def get_content(filename):
        """
        這段Python程式碼定義了一個名為get_content的函數，該函數用於從指定的文件中獲取內容。這個函數使用了一種稱為"快取"的技術來提高效率。

        以下是這段程式碼的詳細解釋：

            1. nonlocal cached_content, cached_content_filename：這行程式碼聲明了cached_content和cached_content_filename兩個變數為非局部變數。這意味著這兩個變數在這個函數外部也有定義，並且在這個函數中對它們的修改會影響到外部的值。

            2. if filename == cached_content_filename：這行程式碼檢查輸入的文件名是否與快取中的文件名相同。如果相同，則直接返回快取的內容。

            3. content = documents[filename].content：這行程式碼從documents字典中獲取指定文件的內容。

            4. cached_content_filename = filename和cached_content = content：這兩行程式碼將新獲取的文件名和內容保存到快取中。

            5. return content：這行程式碼返回獲取的文件內容。

        這種使用快取的方式可以避免重複讀取同一文件，從而提高程式的運行效率。
        """
        nonlocal cached_content, cached_content_filename
        # Check if we can pull from cache
        if filename == cached_content_filename:
            return cached_content
        # If not, grab content
        content = documents[filename].content
        # Cache the content
        cached_content_filename = filename
        cached_content = content
        # Return the now-cached content
        return content

    # Start a Flask server
    app = Flask(__name__)

    @app.route("/")
    def base():
        return send_from_directory(
            pkg_resources.resource_filename("semantra.semantra", "client_public"),
            "index.html",
        )

    # Path for all the static files (compiled JS/CSS, etc.)
    @app.route("/<path:path>")
    def home(path):
        return send_from_directory(
            pkg_resources.resource_filename("semantra.semantra", "client_public"),
            path,
        )

    @app.route("/api/files", methods=["GET"])
    def files():
        return jsonify(
            [
                {
                    "basename": doc.base_filename,
                    "filename": doc.filename,
                    "filetype": doc.content.filetype,
                }
                for doc in documents.values()
            ]
        )

    @app.route("/api/query", methods=["POST"])
    def query():
        """
        這段程式碼定義了一個 Flask 路由 /api/query，該路由接受 POST 請求。當此路由被訪問時，它會執行 query 函數。

        query 函數的主要工作流程如下：

            1. 從請求的 JSON 數據中提取 "queries" 和 "preferences"。

            2. 檢查 svm 和 annoy 變數。如果 svm 為真，則調用 querysvm 函數並返回結果。如果 annoy 為真，則調用 queryann 函數並返回結果。

            3. 如果 svm 和 annoy 都不為真，則使用 model.embed_queries_and_preferences 函數來獲取查詢和偏好的嵌入向量。

            4. 對每個文檔，它會獲取文檔的嵌入向量，並使用餘弦相似度來獲取最近鄰。

            5. 對於每個文檔，它會提取最相似的文本塊，並將其與其他相關信息一起存儲在結果中。

            6. 最後，它會將結果排序並返回 JSON 格式的結果。

        這段程式碼的主要目的是將查詢和偏好嵌入到文檔中，並找出與查詢和偏好最相似的文本塊。
        """
        queries = request.json["queries"]
        preferences = request.json["preferences"]
        if svm:
            return querysvm()
        if annoy:
            return queryann()

        # Get combined query and preference embedding
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)

        results = []
        for doc in documents.values():
            embeddings = doc.embeddings

            # Get kNN with cosine similarity
            distances = np.dot(embeddings, embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embedding)
            )
            sorted_ix = np.argsort(-distances)

            text_chunks = doc.text_chunks
            offsets = doc.offsets[0]
            sub_results = []
            for index in sorted_ix[:num_results]:
                distance = float(distances[index])
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        "distance": distance,
                        "offset": offset,
                        "index": int(index),
                        "filename": doc.filename,
                        "queries": queries,
                        "preferences": preferences,
                    }
                )
            results.append([doc.filename, sub_results])
        return jsonify(sort_results(results, True))

    @app.route("/api/querysvm", methods=["POST"])
    def querysvm():
        """
        這段程式碼是在 Flask 應用中定義一個路由 /api/querysvm，該路由接受 POST 請求。當此路由被訪問時，它會執行 querysvm 函數。

        querysvm 函數的主要工作流程如下：

            1. 從請求的 JSON 數據中提取 "queries" 和 "preferences"。

            2. 使用 model.embed_queries_and_preferences 函數來獲取查詢和偏好的嵌入向量。

            3. 對每個文檔，它會將文檔的嵌入向量和查詢嵌入向量結合，並使用這些數據來訓練一個支持向量機 (SVM) 分類器。

            4. 使用訓練好的 SVM 分類器來推斷相似度，並將結果排序。

            5. 對於每個文檔，它會提取最相似的文本塊，並將其與其他相關信息一起存儲在結果中。

            6. 最後，它會將結果排序並返回 JSON 格式的結果。

        這段程式碼的主要目的是將查詢和偏好嵌入到文檔中，並使用 SVM 來找出與查詢和偏好最相似的文本塊。
        """
        from sklearn import svm

        queries = request.json["queries"]
        preferences = request.json["preferences"]

        # Get combined query and preference embedding
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)
        results = []
        for doc in documents.values():
            embeddings = doc.embeddings

            x = np.concatenate([embeddings, embedding[None, ...]])
            y = np.zeros(len(embeddings) + 1)
            y[-1] = 1

            # Train the svm
            clf = svm.LinearSVC(
                class_weight="balanced",
                verbose=False,
                max_iter=10000,
                tol=1e-6,
                C=svm_c,
            )
            clf.fit(x, y)

            # Infer similarities
            similarities = clf.decision_function(x)[: len(embeddings)]
            sorted_ix = np.argsort(-similarities)

            text_chunks = doc.text_chunks
            offsets = doc.offsets
            sub_results = []
            for index in sorted_ix[:num_results]:
                distance = similarities[index]
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        "distance": distance,
                        "offset": offset,
                        "index": int(index),
                        "filename": doc.filename,
                        "queries": queries,
                        "preferences": preferences,
                    }
                )
            results.append([doc.filename, sub_results])

        return jsonify(sort_results(results, True))

    @app.route("/api/queryann", methods=["POST"])
    def queryann():
        """
        這段程式碼是一個 Flask 應用程式的一部分，它定義了一個路由 /api/queryann，該路由接受 POST 請求。當此路由被訪問時，它會執行 queryann 函數。

        在 queryann 函數中，首先從請求的 JSON 資料中取出 "queries" 和 "preferences"。然後，使用 model.embed_queries_and_preferences 方法來獲取查詢和偏好的嵌入向量。

        接著，對於每個文檔，它會從文檔中取出嵌入數據庫、文本塊和偏移量。然後，使用 embedding_db.get_nns_by_vector 方法來獲取與嵌入向量最接近的結果。

        對於每個結果，它會創建一個包含文本、距離、偏移量、索引、文件名、查詢和偏好的字典，並將其添加到子結果列表中。

        最後，它會將所有的結果排序，並將其轉換為 JSON 格式返回。

        這段程式碼的主要目的是處理查詢和偏好，並返回與之最相關的文檔。
        """
        queries = request.json["queries"]
        preferences = request.json["preferences"]

        # Get combined query and preference embedding
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)

        results = []
        for doc in documents.values():
            embedding_db = doc.embedding_db
            text_chunks = doc.text_chunks
            offsets = doc.offsets[0]
            sub_results = []
            for [index, distance] in zip(
                *embedding_db.get_nns_by_vector(embedding, num_results, -1, True)
            ):
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        # Convert distance from Euclidean distance of normalized vectors to cosine
                        "distance": 1 - distance**2.0 / 2.0,
                        "offset": offset,
                        "index": int(index),
                        "filename": doc.filename,
                        "queries": queries,
                        "preferences": preferences,
                    }
                )
            results.append([doc.filename, sub_results])
        return jsonify(sort_results(results, True))

    @app.route("/api/explain", methods=["POST"])
    def explain():
        """
        這段程式碼定義了一個 Flask 應用的路由，該路由對應於一個名為 /api/explain 的 HTTP 端點。當這個端點接收到 POST 請求時，它會執行 explain 函數的內容。

        explain 函數首先從請求的 JSON 數據中提取出文件名、偏移量、查詢和偏好。然後，它使用這些數據來生成一個嵌入向量，該向量是由模型根據查詢和偏好生成的。

        接下來，函數定義了一些輔助函數，用於對文本進行分割和評分。get_splits 函數將文本分割成多個窗口，exclude_window 函數則將特定窗口的文本從整體文本中排除。get_highest_ranked_split 函數對每個窗口進行評分，並將它們按照評分排序。as_tokens 函數則將這些窗口轉換成一個包含文本和類型的字典列表。

        最後，函數使用這些輔助函數來找到評分最高的窗口，並將其轉換成字典列表。這個列表然後被轉換成 JSON 格式，並作為 HTTP 響應返回。

        這段程式碼的主要目的是對給定的文本進行分析，並找出其中與查詢和偏好最相關的部分。
        """
        filename = request.json["filename"]
        offset = request.json["offset"]
        tokens = documents[filename].text_chunks[offset[0] : offset[1]]
        queries = request.json["queries"]
        preferences = request.json["preferences"]
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)

        # Find hot-spots within the result tokens
        def get_splits(divide_factor=2, num_splits=3, start=0, end=len(tokens)):
            """
            這段Python程式碼定義了一個名為get_splits的函數，該函數用於將一個範圍（由start和end參數定義）分割成多個子範圍。這個函數有四個參數：

                - divide_factor：用於計算窗口長度的除數。窗口長度是每個子範圍的最大長度。
                - num_splits：要創建的子範圍的數量。
                - start：範圍的起始點。
                - end：範圍的結束點。

            函數首先計算窗口長度和分割長度，然後創建一個空列表splits。接著，它進行一個迴圈，每次迴圈都計算一個子範圍的起始點和結束點，並將這個子範圍添加到splits列表中。子範圍的結束點是起始點加上窗口長度，但不能超過範圍的結束點end。

            最後，函數返回splits列表，該列表包含了所有的子範圍。
            """
            window_length = math.ceil((end - start) / divide_factor)
            split_length = math.ceil((end - start) / num_splits)
            splits = []
            for i in range(num_splits):
                splits.append(
                    (
                        start + i * split_length,
                        min(end, start + i * split_length + window_length),
                    )
                )
            return splits

        def exclude_window(start, end):
            nonlocal tokens
            return join_text_chunks(tokens[:start] + tokens[end:])

        def get_highest_ranked_split(splits):
            """
            這段程式碼的主要功能是找出一系列分割點（splits）中，與目前的詞嵌入（embedding）最相似的分割點。以下是詳細的步驟：

                1. split_queries：這一行程式碼將每個分割點轉換為一個查詢，查詢的內容是除了分割點之外的所有詞彙。

                2. split_windows：這一行程式碼將每個查詢轉換為一個詞嵌入向量。這是通過模型的embed_document方法來實現的。

                3. distances：這一行程式碼計算每個詞嵌入向量與目前的詞嵌入的餘弦距離。餘弦距離是一種衡量兩個向量相似度的方法，其值範圍為-1到1，值越大表示相似度越高。

                4. 最後一行程式碼將分割點和對應的距離打包成元組，然後按照距離的大小進行排序，並返回排序後的結果。這樣，最相似的分割點將被放在最前面。

            這段程式碼的主要用途可能是在處理自然語言數據時，找出與某個詞彙或短語最相似的其他詞彙或短語。
            """
            nonlocal tokens, embedding
            split_queries = [exclude_window(start, end) for start, end in splits]
            split_windows = np.array(
                [
                    as_numpy(model.embed_document(split_query))
                    for split_query in split_queries
                ]
            )
            distances = split_windows.dot(embedding) / (
                np.linalg.norm(split_windows, axis=1) * np.linalg.norm(embedding)
            )
            # Return the splits in order of highest to lowest ranked
            return sorted(zip(splits, distances), key=lambda x: x[1], reverse=False)

        def as_tokens(splits):
            """
            這段程式碼的主要功能是將一段文本分割成多個部分，並標記每個部分的類型。以下是詳細的步驟：

                1. indices：這一行程式碼將每個分割點的起始位置提取出來，並按照起始位置的大小進行排序。

                2. append：這是一個內部函數，用於將指定範圍的文本添加到chunks列表中。它會檢查起始位置是否小於結束位置，如果是，則將這段文本和對應的類型打包成一個字典，並添加到chunks列表中。

                3. 在for循环中，程式碼會遍歷每個分割點，並使用append函數將分割點之間的文本添加到chunks列表中。每個分割點都會被標記為"highlight"，而其他部分則被標記為"normal"。

                4. 最後，程式碼會將最後一個分割點到文本結尾的部分添加到chunks列表中。

            這段程式碼的主要用途可能是在處理自然語言數據時，將文本分割成多個部分，並標記每個部分的類型，以便於後續的處理。
            """
            nonlocal tokens
            indices = sorted([split[0] for split in splits], key=lambda x: x[0])
            last_index = 0
            chunks = []

            def append(start, end, type):
                """
                這段Python程式碼定義了一個名為append的函數，該函數接受三個參數：start、end和type。這個函數的主要目的是將一個新的字典添加到chunks列表中。這個字典包含兩個鍵值對：text和type。

                如果start大於或等於end，則函數直接返回，不進行任何操作。這可能是為了防止無效的範圍（即起始位置在結束位置之後）。

                如果start小於end，則函數會執行以下操作：

                    1. 從tokens列表中取出從start到end（不包括end）的元素，並將這些元素連接成一個字符串。這是通過調用join_text_chunks函數實現的。

                    2. 創建一個新的字典，其中text鍵的值是上一步得到的字符串，type鍵的值是函數的第三個參數。

                    3. 將這個新的字典添加到chunks列表的末尾。

                這裡的nonlocal關鍵字表示chunks和tokens是在包含這個函數的外部範疇中定義的變量。
                """
                if start >= end:
                    return
                nonlocal chunks, tokens
                chunks.append(
                    {
                        "text": join_text_chunks(tokens[start:end]),
                        "type": type,
                    }
                )

            for index in indices:
                append(last_index, index[0], "normal")
                append(max(index[0], last_index), index[1], "highlight")
                last_index = index[1]

            append(last_index, len(tokens), "normal")
            return chunks

        splits = get_splits(
            divide_factor=explain_split_divide,
            num_splits=explain_split_count,
            start=0,
            end=len(tokens),
        )
        top_splits = get_highest_ranked_split(splits)[:num_explain_highlights]
        return jsonify(as_tokens(top_splits))

    @app.route("/api/getfile", methods=["GET"])
    def getfile():
        filename = request.args.get("filename")
        content = get_content(filename)
        filename = content.filename
        return send_file(filename)

    @app.route("/api/pdfpositions", methods=["GET"])
    def pdfpositions():
        """
        這段程式碼定義了一個名為 pdfpositions 的路由，該路由對應於一個 HTTP GET 請求。當用戶訪問 /api/pdfpositions 並提供適當的查詢參數（filename）時，此路由將返回一個 PDF 文件的位置信息。

        首先，程式碼從請求的查詢參數中獲取 filename，並使用 get_content 函數獲取相應的文件內容。get_content 函數會檢查文件是否已經在快取中，如果是，則直接從快取中返回文件內容；否則，它會從 documents 字典中獲取文件內容，並將其存入快取。

        然後，程式碼檢查文件的類型是否為 "pdf"。如果是，則返回文件的位置信息；否則，返回一個空列表。位置信息和空列表都被封裝在一個 JSON 響應中，以便於用戶在客戶端處理。

        這段程式碼的主要用途是提供一種方式，讓用戶能夠獲取 PDF 文件的位置信息，這些信息可能包括每個頁面的大小、每個元素的位置等。
        """
        filename = request.args.get("filename")
        content = get_content(filename)
        if content.filetype == "pdf":
            return jsonify(content.positions)
        else:
            return jsonify([])

    @app.route("/api/pdfpage", methods=["GET"])
    def pdfpage():
        """
        這段程式碼定義了一個名為 pdfpage 的路由，該路由對應於一個 HTTP GET 請求。當用戶訪問 /api/pdfpage並提供適當的查詢參數（filename、page 和 scale）時，此路由將返回一個 PDF 文件的特定頁面的圖像。

        首先，程式碼從請求的查詢參數中獲取 filename，並使用 get_content 函數獲取相應的文件內容。get_content 函數會檢查文件是否已經在快取中，如果是，則直接從快取中返回文件內容；否則，它會從 documents 字典中獲取文件內容，並將其存入快取。

        然後，程式碼從請求的查詢參數中獲取 page 和 scale，並使用這兩個參數以及剛剛獲取的文件內容來生成 PDF 文件的特定頁面的圖像。

        最後，程式碼將圖像保存為 PNG 格式，並將其作為 HTTP 響應的內容返回給用戶。響應的 Content-Type 頭部被設置為 image/png，以告訴用戶響應的內容是一個 PNG 圖像。
        """
        filename = request.args.get("filename")
        content = get_content(filename)
        page = request.args.get("page")
        scale = request.args.get("scale")
        if content.filetype == "pdf":
            pil_image = content.get_page_image_pil(int(page), float(scale))
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            response = make_response(img_byte_arr.getvalue())
            response.headers.set("Content-Type", "image/png")
            return response

    @app.route("/api/pdfchars", methods=["GET"])
    def pdfchars():
        """
        這段程式碼定義了一個 Flask 應用的路由，該路由對應於 /api/pdfchars URL，並且只接受 GET 請求。

        在這個路由的處理函數 pdfchars() 中，首先從請求的參數中獲取 filename。然後，使用 get_content(filename) 函數獲取該文件的內容。

        接著，檢查文件的類型是否為 "pdf"。如果不是，則返回一個空的 JSON 陣列。

        如果文件類型是 "pdf"，則從請求的參數中獲取 page，並使用 content.get_page_chars(int(page)) 獲取該頁面的字符。最後，將這些字符轉換為 JSON 格式並返回。

        這個路由的主要目的是提供一個 API，用戶可以通過這個 API 獲取 PDF 文件的特定頁面的字符。
        """
        filename = request.args.get("filename")
        content = get_content(filename)
        if content.filetype != "pdf":
            return jsonify([])
        page = request.args.get("page")
        return jsonify(content.get_page_chars(int(page)))

    @app.route("/api/text", methods=["GET"])
    def text():
        """
        這段程式碼是在一個條件下運行 Flask 應用程式的服務器。如果 no_server 變數為 False，則會嘗試運行服務器。app.run(host=host, port=port) 這行程式碼會啟動一個服務器，並且將其綁定到指定的主機地址和端口。

        如果在啟動服務器時發生 SystemExit 異常，則會捕獲該異常並進行處理。如果端口是預設端口，則會引發一個新的異常，並建議用戶嘗試再次運行並使用 --port <port> 命令來指定一個不同的端口。如果端口不是預設端口，則會引發一個新的異常，並建議用戶嘗試指定一個不同的端口。
        """
        filename = request.args.get("filename")
        return jsonify(documents[filename].text_chunks)
    
    # 這段程式碼是在啟動 Flask 應用時使用的。如果 no_server 變數為 False，則會嘗試在指定的 host 和 port 上運行應用。
    # 這裡使用了一個 try/except 區塊來處理可能出現的 SystemExit 異常。如果在運行應用時出現 SystemExit 異常，則會進入 except 區塊。
    # 在 except 區塊中，首先將 sys.tracebacklimit 設置為 0，這樣在出現異常時，Python 就不會打印出錯誤的回溯信息。
    # 然後，檢查 port 是否等於 DEFAULT_PORT。如果等於，則引發一個新的異常，並提示用戶嘗試再次運行並使用 --port <port> 命令來指定一個不同的端口。如果 port 不等於 DEFAULT_PORT，則引發一個新的異常，並提示用戶嘗試指定一個不同的端口。
    # 這段程式碼的主要目的是處理在啟動 Flask 應用時可能出現的 SystemExit 異常，並給出相應的錯誤提示。
    if not no_server:
        try:
            app.run(host=host, port=port)
        except SystemExit as e:
            import sys
            sys.tracebacklimit=0
            if port == DEFAULT_PORT:
                raise Exception(
                    f'Try running again and adding `--port <port>` to the command to specify a different port.'
                ) from None
            else:
                raise Exception(f"Try specifying a different port with `--port <port>`.") from None


if __name__ == "__main__":
    main()
