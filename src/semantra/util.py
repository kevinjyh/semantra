import struct
import hashlib
import os
import numpy as np

HASH_LENGTH = 24


def file_md5(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:HASH_LENGTH]


def join_text_chunks(chunks):
    return "".join(chunks)


# Filenames for generated files
def get_converted_pdf_txt_filename(md5):
    return f"{md5}.pdf.txt"


def get_pdf_positions_filename(md5):
    return f"{md5}.pdf.positions.json"


def get_tokens_filename(md5, config_hash):
    return f"{md5}.{config_hash}.tokens.json"


def get_embeddings_filename(md5, config_hash, size, offset, rewind):
    return f"{md5}.{config_hash}.{size}_{offset}_{rewind}.embeddings"


def get_annoy_filename(md5, config_hash, size, offset, rewind, num_trees):
    return f"{md5}.{config_hash}.{size}_{offset}_{rewind}.{num_trees}t.annoy"


def get_config_filename(md5, config_hash):
    return f"{md5}.{config_hash}.config.json"


def write_embedding(file, embedding, num_dimensions):
    """
    函數接受三個參數：file、embedding 和 num_dimensions。file 是要寫入的文件對象，embedding 是要寫入的嵌入向量，num_dimensions 是嵌入向量的維度數。

    在函數內部，首先使用 range(num_dimensions) 來生成一個範圍對象，該對象包含從 0 到 num_dimensions 的整數。

    然後，對於範圍對象中的每個整數 i，使用 file.write(struct.pack("f", embedding[i])) 來將嵌入向量的第 i 個元素轉換為浮點數編碼，並寫入文件。

    最後，使用 file.flush() 來清空文件緩衝區，並將所有未寫入的資訊寫入文件。這是為了確保所有的嵌入向量都已經被寫入文件。

    這個函數的主要用途是在 Python 中處理嵌入向量的寫入，尤其是將嵌入向量轉換為浮點數編碼並寫入文件，以便進一步處理或分析。
    """
    # Write float-encoded embeddings
    for i in range(num_dimensions):
        file.write(struct.pack("f", embedding[i]))
    file.flush()


def read_embedding(chunk, num_dimensions):
    """
    這段程式碼的功能是讀取浮點數編碼的嵌入。嵌入是一種將對象（在這種情況下可能是詞彙）表示為數字向量的方式。這種表示可以捕獲對象之間的相似性，並且可以用於許多機器學習任務。

    讓我們逐行解釋這段程式碼：

        1. def read_embedding(chunk, num_dimensions):：這行程式碼定義了一個名為read_embedding的函數，該函數接受兩個參數：chunk和num_dimensions。chunk是包含嵌入的二進制數據，num_dimensions是嵌入的維度數。

        2. embedding = []：這行程式碼創建了一個空的列表，用於存儲讀取的嵌入。

        3. for i in range(num_dimensions):：這行程式碼開始了一個循環，對每個維度進行迭代。

        4. embedding.append(struct.unpack("f", chunk[i * 4 : (i + 1) * 4])[0])：這行程式碼讀取了chunk中的下一個浮點數，並將其添加到嵌入列表中。struct.unpack函數用於將二進制數據轉換為Python數據類型。在這裡，它被用來將四個字節的二進制數據轉換為一個浮點數。

        5. return embedding：這行程式碼返回讀取的嵌入。

    這個函數可以用於讀取存儲在二進制文件中的嵌入，例如由word2vec或GloVe這類詞嵌入模型生成的嵌入。
    """
    # Read float-encoded embeddings
    embedding = []
    for i in range(num_dimensions):
        embedding.append(struct.unpack("f", chunk[i * 4 : (i + 1) * 4])[0])
    return embedding

# Annoy（Approximate Nearest Neighbors Oh Yeah）是一種用於搜索近似最近鄰的C++庫，由Spotify開發。
# 它可以快速查詢大量數據點之間的相似性，並且在內存使用和查詢速度之間提供了一種平衡。
# 在這段程式碼中，Annoy被用來存儲和查詢詞嵌入，這些詞嵌入是由模型生成的，用於表示文本中的詞彙。
def write_annoy_db(filename, num_dimensions, embeddings, num_trees):
    """
    這段程式碼的功能是將嵌入寫入Annoy數據庫。Annoy是一種用於搜索近似最近鄰的C++庫，由Spotify開發。在這段程式碼中，Annoy被用來存儲和查詢詞嵌入，這些詞嵌入是由模型生成的，用於表示文本中的詞彙。

    讓我們逐行解釋這段程式碼：

        1. from annoy import AnnoyIndex：這行程式碼導入了Annoy庫的AnnoyIndex類。

        2. dbs = []：這行程式碼創建了一個空的列表，用於存儲Annoy數據庫的實例。

        3. db = AnnoyIndex(num_dimensions, "angular")：這行程式碼創建了一個Annoy數據庫的實例。這個數據庫將用於存儲具有指定維度的嵌入，並使用角度距離來計算嵌入之間的相似性。

        4. for i, embedding in enumerate(embeddings):：這行程式碼開始了一個循環，對每個嵌入進行迭代。

        5. db.add_item(i, embedding)：這行程式碼將每個嵌入添加到Annoy數據庫中，並使用其在列表中的索引作為該嵌入的ID。

        6. db.build(num_trees)：這行程式碼構建了Annoy數據庫的索引。這個過程需要一些時間，但是一旦完成，就可以快速查詢嵌入的最近鄰。

        7. db.save(filename)：這行程式碼將Annoy數據庫保存到指定的文件中。

        8. dbs.append(db)：這行程式碼將剛剛建立的Annoy數據庫添加到列表中。

        9. return dbs：這行程式碼返回包含所有Annoy數據庫的列表。
    """
    # Import annoy here so that it's not required for the CLI
    from annoy import AnnoyIndex

    dbs = []
    db = AnnoyIndex(num_dimensions, "angular")
    for i, embedding in enumerate(embeddings):
        db.add_item(i, embedding)
    db.build(num_trees)
    db.save(filename)
    dbs.append(db)

    return dbs


def load_annoy_db(filename, num_dimensions):
    """
    這段程式碼定義了一個名為 load_annoy_db 的函數，該函數用於加載 Annoy 數據庫。

    該函數接受兩個參數：filename 和 num_dimensions。filename 是要加載的 Annoy 數據庫的文件名，num_dimensions 是數據庫中向量的維度。

    在函數內部，首先將 Annoy 庫導入到函數的局部作用域中。這樣做的好處是，如果不需要使用這個函數，則不需要安裝或導入 Annoy 庫，這可以節省記憶體和加載時間。

    然後，使用 AnnoyIndex 類創建一個新的 Annoy 索引。這個索引的維度是 num_dimensions，並使用 "angular" 為距離度量。

    接著，使用 load 方法從 filename 指定的文件中加載 Annoy 索引。

    最後，返回加載的 Annoy 索引。

    這個函數的主要目的是提供一種方便的方式來加載 Annoy 數據庫，並將其作為一個索引返回。
    """
    # Import annoy here so that it's not required for the CLI
    from annoy import AnnoyIndex

    db = AnnoyIndex(num_dimensions, "angular")
    db.load(filename)
    return db


def get_num_annoy_embeddings(annoy_filename, num_dimensions):
    return load_annoy_db(annoy_filename, num_dimensions).get_n_items()


def safe_remove(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def get_num_embeddings(embeddings_filename, num_dimensions):
    """
    這段程式碼的功能是計算一個給定的嵌入文件中的嵌入數量。這裡的嵌入通常指的是用於表示詞彙的向量。

    讓我們逐行解釋這段程式碼：

        1. with open(embeddings_filename, "rb") as f:：這行程式碼打開了一個名為 embeddings_filename 的文件，並將其內容讀取為二進制格式。with 語句確保文件在操作完成後會被正確關閉。

        2. f.seek(0, 2): seek 函數用於改變文件讀取的位置。這裡的 0, 2 表示將讀取位置移動到文件的結尾。

        3. file_size = f.tell(): tell 函數返回當前的讀取位置。由於我們剛剛將讀取位置移動到了文件的結尾，所以這裡返回的就是文件的大小。

        4. return file_size // (num_dimensions * 4): 這行程式碼計算了嵌入的數量。每個嵌入都是一個浮點數的向量，每個浮點數佔用4個字節，所以嵌入的數量就是文件大小除以（嵌入的維度數量乘以4）。
    """
    # Get the file size
    with open(embeddings_filename, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()

    # Calculate the number of embeddings
    return file_size // (num_dimensions * 4)


def read_embeddings_file(embeddings_filename, num_dimensions, capacity):
    """
    這段程式碼是從一個檔案中讀取詞嵌入（embeddings）的函數。詞嵌入是一種將詞彙轉換為數字向量的技術，這些向量可以捕捉詞彙之間的語義關係。

    以下是該函數的步驟：

    計算詞嵌入的數量。這裡使用了min函數來確保詞嵌入的數量不會超過預先設定的容量。

    使用open函數以二進制模式打開檔案，並使用truncate方法將檔案大小調整為預期的大小。這裡的預期大小是詞嵌入的數量乘以每個詞嵌入的維度數乘以4（因為每個浮點數需要4個位元組）。

    如果詞嵌入的數量為0，則返回一個形狀為（容量，維度數）的全零數組。

    使用np.memmap函數將檔案映射到記憶體中。這樣可以在不將整個檔案加載到記憶體的情況下讀取檔案。

    創建一個形狀為（容量，維度數）的全零數組。

    將原始詞嵌入複製到新的數組中。

    返回詞嵌入數組和詞嵌入的數量。

    這個函數可以用於讀取大型詞嵌入檔案，並將其轉換為可以在Python中使用的數組。
    """
    # Calculate the number of embeddings
    num_embeddings = min(
        get_num_embeddings(embeddings_filename, num_dimensions), capacity
    )

    # Change the file size to the expected size
    with open(embeddings_filename, "ab") as f:
        f.truncate(num_embeddings * num_dimensions * 4)

    if num_embeddings == 0:
        return np.zeros((capacity, num_dimensions), dtype="float32"), 0

    # Memory map the file
    read_embeddings = np.memmap(
        embeddings_filename,
        dtype="float32",
        mode="r",
        shape=(num_embeddings, num_dimensions),
    )

    # Create an array with shape (capacity, num_dimensions) filled with 0s
    embeddings = np.zeros((capacity, num_dimensions), dtype="float32")

    # Copy the original embeddings into the new array
    embeddings[:num_embeddings] = read_embeddings[:num_embeddings]

    return embeddings, num_embeddings


def get_offsets(doc_size, windows):
    """
    這段程式碼定義了一個名為 get_offsets 的函數，該函數用於計算文檔中的偏移量。這些偏移量可以用於將文檔分割成多個窗口，每個窗口都有一個特定的大小和偏移量。

    以下是該函數的步驟：

        1.初始化 num_tokens 和 offsets。num_tokens 用於計算文檔中的總標記數，offsets 用於存儲每個窗口的偏移量。

        2.遍歷每個窗口。每個窗口都由三個元素組成：窗口大小 size，窗口偏移量 offset 和回滾量 rewind。

        3.對於每個窗口，計算其偏移量。如果偏移量大於0，則將其添加到 sub_offsets 中，並將其加到 num_tokens 上。否則，將 x 設置為 rewind。

        4.當 x 小於 doc_size 時，將 x 減去 rewind，然後將 [x, min(x + size, doc_size)] 添加到 sub_offsets 中，並將 min(x + size, doc_size) - x 加到 num_tokens 上。然後將 x 增加 size。

        5.將 sub_offsets 添加到 offsets 中。

        6.返回 offsets 和 num_tokens。

    這個函數可以用於將文檔分割成多個窗口，並計算每個窗口的偏移量和總標記數。這對於處理大型文檔或需要將文檔分割成多個部分的情況非常有用。
    """
    num_tokens = 0

    offsets = []

    for size, offset, rewind in windows:
        sub_offsets = []
        x = 0
        if offset > 0:
            sub_offsets.append([0, offset])
            num_tokens += offset
            x = offset
        else:
            x = rewind

        while x < doc_size:
            x -= rewind
            sub_offsets.append([x, min(x + size, doc_size)])
            num_tokens += min(x + size, doc_size) - x
            x += size

        offsets.append(sub_offsets)

    return offsets, num_tokens


def sort_results(results, reverse):
    """
    這段程式碼定義了一個名為 sort_results 的函數，該函數用於對結果進行排序。這些結果是一個列表，每個元素都是一個包含距離的字典。

    以下是該函數的步驟：

        1.初始化 avg_distances 為空列表。這個列表將用於存儲每個結果的平均距離。

        2.遍歷每個結果。對於每個結果，計算其所有距離的平均值，並將其添加到 avg_distances 中。

        3.使用 zip 函數將 avg_distances 和 results 組合成一個元組列表，然後使用 sorted 函數對這個列表進行排序。排序的依據是每個元組的第一個元素，即平均距離。如果 reverse 為 True，則進行降序排序；否則進行升序排序。

        4.使用列表推導式從排序後的元組列表中提取出結果，並將其存儲在字典的 "results" 鍵下。

        5.將排序方式存儲在字典的 "sort" 鍵下。如果 reverse 為 True，則排序方式為 "desc"；否則為 "asc"。

        6.返回這個字典。

    這個函數可以用於將結果按照平均距離進行排序，並返回一個包含排序後的結果和排序方式的字典。這對於需要將結果按照某種標準進行排序的情況非常有用。
    """
    # Get average distance per result
    avg_distances = []
    for result in results:
        avg_distances.append(np.mean([item["distance"] for item in result[1]]))

    # Sort results by average distance
    return {
        "results": [x for _, x in sorted(zip(avg_distances, results), reverse=reverse)],
        "sort": "desc" if reverse else "asc",
    }
