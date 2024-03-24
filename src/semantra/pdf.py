import pypdfium2 as pdfium
from threading import Lock
import json
import os
from tqdm import tqdm
from .util import get_converted_pdf_txt_filename, get_pdf_positions_filename

mutexes = {}


def get_mutex(filename):
    # Ensure that only one thread is accessing a PDF at a time
    if filename not in mutexes:
        mutexes[filename] = Lock()
    return mutexes[filename]


class PDFContent:
    def __init__(self, rawtext, filename, positions):
        self.rawtext = rawtext
        self.filename = filename
        self.positions = positions
        self.pdfium = pdfium.PdfDocument(filename)
        self.mutex = get_mutex(filename)
        self.filetype = "pdf"

    def get_page_image_pil(self, page_number, scale):
        """
        這段程式碼是在 Python 中定義的一個方法，名為 get_page_image_pil。該方法的目的是將 PDF 文件的特定頁面轉換為 PIL（Python Imaging Library）圖像。

        該方法接受兩個參數：page_number 和 scale。page_number 是要轉換的 PDF 頁面的編號，而 scale 是轉換的縮放比例。

        在方法內部，首先使用互斥鎖 mutex 來確保在同一時間只有一個線程可以訪問該段程式碼。這是為了防止在多線程環境中出現資源競爭的問題。

        然後，使用 self.pdfium[page_number] 來獲取指定的 PDF 頁面。self.pdfium 是一個 PDF 文件對象，可以用來訪問文件中的各個頁面。

        接著，使用 page.render(scale=scale) 來渲染該頁面。這將會生成一個位圖（bitmap）對象，該對象代表了渲染後的頁面圖像。

        最後，使用 bitmap.to_pil() 將位圖對象轉換為 PIL 圖像對象，並將其返回。

        這個方法的主要用途是在 Python 中處理 PDF 文件，尤其是將 PDF 頁面轉換為圖像，以便進一步處理或顯示。
        """
        with self.mutex:
            page = self.pdfium[page_number]
            bitmap = page.render(scale=scale)
            return bitmap.to_pil()

    def get_page_chars(self, page_number):
        """
        這段程式碼定義了一個名為 get_page_chars 的方法，該方法的目的是從 PDF 文件的特定頁面中提取字符及其對應的邊界框。

        該方法接受一個參數 page_number，該參數指定要提取字符的 PDF 頁面的編號。

        在方法內部，首先使用互斥鎖 mutex 來確保在同一時間只有一個線程可以訪問該段程式碼。這是為了防止在多線程環境中出現資源競爭的問題。

        然後，使用 self.pdfium[page_number] 來獲取指定的 PDF 頁面。self.pdfium 是一個 PDF 文件對象，可以用來訪問文件中的各個頁面。

        接著，使用 page.get_textpage() 來獲取該頁面的文本頁面對象。這個對象可以用來訪問頁面上的文本內容。

        然後，使用 textpage.count_chars() 來獲取該頁面上的字符數量，並使用 textpage.get_charbox(i) 來獲取每個字符的邊界框。

        最後，使用 textpage.get_text_range(index=i, count=1) 來獲取每個字符的文本，並將字符和其對應的邊界框一起返回。

        這個方法的主要用途是在 Python 中處理 PDF 文件，尤其是提取 PDF 頁面上的字符及其邊界框，以便進一步處理或分析。
        """
        with self.mutex:
            page = self.pdfium[page_number]
            textpage = page.get_textpage()
            num_chars = textpage.count_chars()
            char_boxes = [textpage.get_charbox(i) for i in range(num_chars)]
            chars = [
                textpage.get_text_range(index=i, count=1) for i in range(num_chars)
            ]
            return [(c, b) for c, b in list(zip(chars, char_boxes))]


# Page separator character
LINE_FEED = "\f"


def get_pdf_content(md5, filename, semantra_dir, force, silent):
    """
    這段程式碼定義了一個名為 get_pdf_content 的函數，該函數的目的是從指定的 PDF 文件中提取文本內容和位置信息。

    該函數接受五個參數：md5、filename、semantra_dir、force 和 silent。md5 是 PDF 文件的 MD5 哈希值，用於生成唯一的文件名。filename 是 PDF 文件的路徑。semantra_dir 是存儲轉換後的文本文件和位置索引文件的目錄。force 是一個布爾值，如果為 True，則無論文件是否已存在，都會強制提取 PDF 內容。silent 是一個布爾值，如果為 True，則在提取 PDF 內容時不會顯示進度條。

    在函數內部，首先創建一個 pdfium.PdfDocument 對象來讀取 PDF 文件，並獲取 PDF 文件的頁面數量。

    然後，檢查是否需要提取 PDF 內容。如果 force 為 True，或者轉換後的文本文件或位置索引文件不存在，則進行提取。

    在提取過程中，對於 PDF 文件的每一頁，都會獲取頁面的大小和文本內容，並將文本內容寫入轉換後的文本文件，同時記錄每一頁的字符索引、寬度和高度。

    提取完成後，將位置信息寫入位置索引文件，並讀取轉換後的文本文件的內容。然後，創建一個 PDFContent 對象，並將文本內容、文件名和位置信息作為參數傳入，最後返回該對象。

    如果不需要提取 PDF 內容，則直接讀取轉換後的文本文件和位置索引文件的內容，並創建並返回一個 PDFContent 對象。

    這個函數的主要用途是在 Python 中處理 PDF 文件，尤其是提取 PDF 文件的文本內容和位置信息，以便進一步處理或分析。
    """
    converted_txt = os.path.join(semantra_dir, get_converted_pdf_txt_filename(md5))
    position_index = os.path.join(semantra_dir, get_pdf_positions_filename(md5))

    pdf = pdfium.PdfDocument(filename)
    n_pages = len(pdf)

    if force or not os.path.exists(converted_txt) or not os.path.exists(position_index):
        positions = []
        position = 0
        # newline="" ensures pdfium's \r is preserved
        with open(
            converted_txt, "w", newline="", encoding="utf-8", errors="ignore"
        ) as f:
            for page_index in tqdm(
                range(n_pages),
                desc="Extracting PDF contents",
                leave=False,
                disable=silent,
            ):
                page = pdf[page_index]
                page_width, page_height = page.get_size()
                textpage = page.get_textpage()
                pagetext = textpage.get_text_range()

                positions.append(
                    {
                        "char_index": position,
                        "page_width": page_width,
                        "page_height": page_height,
                    }
                )
                position += f.write(pagetext)
                position += f.write(LINE_FEED)
        with open(position_index, "w", encoding="utf-8") as f:
            json.dump(positions, f)
        with open(
            converted_txt, "r", newline="", encoding="utf-8", errors="ignore"
        ) as f:
            rawtext = f.read()
        return PDFContent(rawtext, filename, positions)
    else:
        with open(
            converted_txt, "r", newline="", encoding="utf-8", errors="ignore"
        ) as f:
            rawtext = f.read()
        with open(position_index, "r", encoding="utf-8") as f:
            positions = json.load(f)

        return PDFContent(rawtext, filename, positions)
