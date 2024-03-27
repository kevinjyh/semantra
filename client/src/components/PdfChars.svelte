<script lang="ts">
  import { onMount, tick } from "svelte";
  import type { File, Offset, PdfChar, PdfPosition } from "../types";
  import { copyChars, layout } from "../layoutEngine";

  export let file: File;
  export let pageNumber: number;
  export let position: PdfPosition;
  export let selectedOffset: Offset | null;
  export let zoom: number;
  export let scrollHighlights: boolean;

  function x(x: number): number {
    return (x / position.page_width) * 100;
  }

  function y(y: number): number {
    return (y / position.page_height) * 100;
  }

  async function scrollHighlightsIntoView(...args: any) {
    await tick();
    const highlights = document.querySelectorAll(".page-highlight");
    if (highlights.length == 0) return;
    highlights[0].scrollIntoView({
      block: "center",
    });
    scrollHighlights = false;
  }

  function processChars(chars: PdfChar[]): [PdfChar[], [number, number][]] {
    // Join words together
    const processedChars: PdfChar[] = [];

    const isSpace = (char: PdfChar): boolean => {
      return /\s/.test(char[0]);
    };

    const wordIndexMap: [number, number][] = [];
    let currentWord: PdfChar[] = [];
    let wordStart: number | null = null;
    let wordEnd: number | null = null;

    const pushChar = (char: PdfChar, start: number, end: number) => {
      processedChars.push(char);
      wordIndexMap.push([start, end]);
    };

    const buildWord = (char: PdfChar, i: number) => {
      if (wordStart == null) wordStart = i;
      wordEnd = i + 1;
      currentWord.push(char);
    };

    const getMin = (l: number[]): number => {
      let min: number;
      for (const x of l) {
        if (min == null || x < min) {
          min = x;
        }
      }
      return min;
    };

    const getMax = (l: number[]): number => {
      let max: number;
      for (const x of l) {
        if (max == null || x > max) {
          max = x;
        }
      }
      return max;
    };

    const pushWord = () => {
      if (currentWord.length == 0 || wordStart == null || wordEnd == null)
        return;
      const word = currentWord.map((x) => x[0]).join("");
      const x0 = getMin(currentWord.map((x) => x[1].x0));
      const x1 = getMax(currentWord.map((x) => x[1].x1));
      const y0 = getMin(currentWord.map((x) => x[1].y0));
      const y1 = getMax(currentWord.map((x) => x[1].y1));
      const char: PdfChar = [word, { x0, x1, y0, y1 }];
      pushChar(char, wordStart, wordEnd);
      currentWord = [];
      wordStart = null;
      wordEnd = null;
    };

    for (let i = 0; i < chars.length; i++) {
      const char = chars[i];
      if (isSpace(char)) {
        pushWord();
        pushChar(char, i, i + 1);
      } else {
        buildWord(char, i);
      }
    }
    pushWord();

    return [processedChars, wordIndexMap];
  }

  let chars: PdfChar[] = [];
  $: words = processChars(copyChars(chars));
  $: processedChars = layout(
    position.page_width,
    position.page_height,
    words[0]
  );
// TODO: 在 PDF 文件的某一頁指定段落文字的高亮功能
  function getHighlightWordIndices(
    words: [PdfChar[], [number, number][]],
    start: number,
    end: number
  ): number[] {
    const wordIndices = words[1];
    const highlights = wordIndices
      .map<[number, [number, number]]>((x, i) => [i, x])
      .filter((x) => x[1][0] >= start && x[1][1] <= end)
      .map((x) => x[0]);
    return highlights;
  }

  $: highlightWordIndices =
    selectedOffset == null
      ? []
      : getHighlightWordIndices(
          words,
          selectedOffset[0] - position.char_index,
          selectedOffset[1] - position.char_index
        );

  $: scrollHighlights && scrollHighlightsIntoView(highlightWordIndices);
  let containerElem: HTMLElement;

  const baseFontSize = 16;
  let mWidth = 10;
  let mHeight = 24;

  // onMount(async ()) 這段程式碼來自一個名為 PdfChars.svelte 的 Svelte 組件。在這個組件中，當組件掛載（mount）到 DOM 時，會執行 onMount 函數中的程式碼。以下是對這段程式碼的逐行解釋：

  // 首先，程式碼創建一個 span 元素，並設置其內容為 "m"。然後，它將這個 span 元素的樣式設置為絕對定位、隱藏、不換行、字體為等寬字體，並設置字體大小為 baseFontSize。這個 span 元素被添加到 containerElem 元素中，然後程式碼獲取其邊界大小（寬度和高度），並將其存儲在 mWidth 和 mHeight 變數中。最後，這個 span 元素被從 containerElem 元素中移除。

  // 然後，程式碼發起一個對 /api/pdfchars 的 GET 請求，並將 file.filename 和 pageNumber 作為查詢參數。這個請求的響應被解析為 JSON 格式。

  // 最後，程式碼將解析出的 JSON 數據映射（map）為一個新的數組，並將其存儲在 chars 變數中。這個新的數組的每個元素都是一個包含兩個元素的數組：第一個元素是原始數組的第一個元素，第二個元素是一個包含 x0、y0、x1 和 y1 屬性的物件，這些屬性的值分別對應於原始數組的第二個元素的四個元素。

  // 這段程式碼的主要目的是獲取 PDF 文件中的字符資訊，並將其轉換為一種更方便處理的格式。
  onMount(async () => {
    // Measure a monospace 'm'
    const m = document.createElement("span");
    m.textContent = "m";
    m.style.position = "absolute";
    m.style.visibility = "hidden";
    m.style.whiteSpace = "pre";
    m.style.fontFamily = "monospace";
    m.style.fontSize = `${baseFontSize}px`;
    containerElem.appendChild(m);
    const mBounds = m.getBoundingClientRect();
    mWidth = mBounds.width;
    mHeight = mBounds.height;
    containerElem.removeChild(m);

    const response = await fetch(
      `/api/pdfchars?filename=${encodeURIComponent(
        file.filename
      )}&page=${pageNumber}`
    );
    const json = await response.json();
    chars = json.map((x) => [
      x[0],
      {
        x0: x[1][0],
        y0: x[1][1],
        x1: x[1][2],
        y1: x[1][3],
      },
    ]);
  });
</script>

<div class="absolute left-0 top-0 right-0 bottom-0" bind:this={containerElem}>
  {#each processedChars as char, i}
    <div
      class="absolute content-box text-transparent"
      style="left: {(char[1].x0 - (char[1].lpad || 0)) *
        zoom}px; top: {(position.page_height -
        char[1].y1 -
        (char[1].bpad || 0)) *
        zoom}px; width: {(char[1].x1 -
        char[1].x0 +
        (char[1].lpad || 0) +
        (char[1].rpad || 0)) *
        zoom}px; height: {(char[1].y1 -
        char[1].y0 +
        (char[1].bpad || 0) +
        (char[1].tpad || 0)) *
        zoom}px;
        padding-left: {(char[1].lpad || 0) * zoom}px;
        padding-right: {(char[1].rpad || 0) * zoom}px;
        padding-bottom: {(char[1].tpad || 0) * zoom}px;
        padding-top: {(char[1].bpad || 0) * zoom}px;"
    >
      <span
        class="inline-block whitespace-pre origin-top-left select-all"
        class:page-highlight={highlightWordIndices.includes(i)}
        style="font-family: monospace; font-size: {baseFontSize}px; transform: scale({((char[1]
          .x1 -
          char[1].x0) /
          mWidth /
          char[0].length) *
          zoom}, {((char[1].y1 - char[1].y0) / mHeight) * zoom});"
        >{char[0]}</span
      >
    </div>
  {/each}
</div>

<style>
  .page-highlight {
    background-color: rgb(255 255 0 / 72%);
    mix-blend-mode: darken;
  }
</style>

<!-- 這段程式碼來自一個名為 PdfChars.svelte 的 Svelte 組件。這個組件的主要功能是處理 PDF 文件中的字符資訊，並將其顯示在頁面上。以下是對這段程式碼的逐行解釋：

1. 程式碼首先導入了一些必要的模組和類型，包括 Svelte 的 onMount 和 tick 函數，以及一些自定義的類型和函數。

2. 然後，程式碼定義了一些變數，包括 file、pageNumber、position、selectedOffset、zoom 和 scrollHighlights。這些變數都是由父組件傳入的 props。

3. 接著，程式碼定義了一些函數，包括 x、y、scrollHighlightsIntoView 和 processChars。其中，x 和 y 函數用於將 PDF 文件中的座標轉換為百分比；scrollHighlightsIntoView 函數用於將頁面滾動到高亮的部分；processChars 函數用於處理 PDF 文件中的字符資訊。

4. 然後，程式碼定義了一些反應式語句（以 $: 開頭的語句）。這些語句會在相關變數變化時自動重新執行。

5. 最後，程式碼定義了一個 onMount 函數，該函數會在組件掛載到 DOM 時執行。在這個函數中，程式碼首先測量了一個等寬字體的 'm' 字符的寬度和高度，然後發起了一個 HTTP 請求來獲取 PDF 文件中的字符資訊，並將其存儲在 chars 變數中。

在 HTML 部分，程式碼使用了 Svelte 的 each 指令來遍歷 processedChars 陣列，並為每個字符創建一個 div 元素。每個 div 元素的位置和大小都是根據字符的座標和大小計算出來的。每個 div 內部包含一個 span 元素，該元素的內容就是字符本身。

在 CSS 部分，程式碼定義了一個名為 .page-highlight 的類，該類會將背景顏色設置為黃色，並使用混合模式來使背景顏色與文字顏色混合。 -->



<!-- #selection 請針對頁面 highlight 的運作機制詳盡解說 -->
<!-- 這個 Svelte 組件的主要功能是在 PDF 文件的某一頁上顯示文字，並提供文字的高亮功能。以下是對頁面高亮運作機制的詳細解釋：

1. `highlightWordIndices` 是一個反應式變數，當 `selectedOffset` 或 `words` 變化時，它會重新計算。如果 `selectedOffset` 為 null，則 `highlightWordIndices` 為空陣列。否則，它會調用 `getHighlightWordIndices` 函數，傳入 `words`、`selectedOffset[0] - position.char_index` 和 `selectedOffset[1] - position.char_index` 作為參數，並將返回的高亮詞彙索引存儲在 `highlightWordIndices` 中。

2. `getHighlightWordIndices` 函數接收三個參數：`words`、`start` 和 `end`。它首先從 `words` 中獲取詞彙索引，然後過濾出在 `start` 和 `end` 範圍內的詞彙索引，並返回這些索引。

3. 在 Svelte 的 HTML 模板中，每個字符都被包裹在一個 `div` 元素中，並有一個 `span` 子元素用於顯示字符。這個 `span` 元素有一個 `class:page-highlight` 屬性，該屬性的值由 `highlightWordIndices.includes(i)` 決定。如果 `highlightWordIndices` 包含當前字符的索引，則 `span` 元素將獲得 `page-highlight` 類別。

4. 在 CSS 樣式中，`page-highlight` 類別的元素將獲得黃色背景色，並使用 `mix-blend-mode: darken;` 來混合背景色和文字色，使高亮的文字更易於閱讀。

5. 當 `scrollHighlights` 為 true 時，將調用 `scrollHighlightsIntoView` 函數，該函數將滾動視窗，使第一個高亮的元素出現在視窗中央。然後將 `scrollHighlights` 設置為 false，以防止無限滾動。

總的來說，這個組件的高亮功能是通過反應式變數、過濾函數、條件類別和 CSS 樣式來實現的。當選擇的範圍（由 `selectedOffset` 決定）變化時，相應的文字將被高亮顯示，並且視窗將滾動到第一個高亮的文字處。 -->
