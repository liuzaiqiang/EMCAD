from __future__ import annotations

import heapq
import os
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)


def register_font() -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simkai.ttf"),
        Path(r"C:\Windows\Fonts\Noto Sans SC (TrueType).otf"),
    ]
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("XJUSans", str(path)))
            return "XJUSans"
    raise FileNotFoundError("No Chinese font found")


FONT = register_font()


def esc(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


styles = getSampleStyleSheet()
TITLE = ParagraphStyle(
    "XJUTitle", parent=styles["Title"], fontName=FONT, fontSize=17,
    leading=23, alignment=TA_CENTER, spaceAfter=8,
)
SUBTITLE = ParagraphStyle(
    "XJUSubtitle", parent=styles["Normal"], fontName=FONT, fontSize=9.5,
    leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
    spaceAfter=10,
)
H1 = ParagraphStyle(
    "XJUH1", parent=styles["Heading1"], fontName=FONT, fontSize=13,
    leading=18, spaceBefore=8, spaceAfter=6, textColor=colors.HexColor("#163a5f"),
)
H2 = ParagraphStyle(
    "XJUH2", parent=styles["Heading2"], fontName=FONT, fontSize=10.5,
    leading=15, spaceBefore=6, spaceAfter=4, textColor=colors.HexColor("#274e73"),
)
BODY = ParagraphStyle(
    "XJUBody", parent=styles["BodyText"], fontName=FONT, fontSize=9.3,
    leading=14, spaceAfter=3, alignment=TA_LEFT,
)
SMALL = ParagraphStyle(
    "XJUSmall", parent=BODY, fontSize=8.3, leading=12,
)
ANSWER = ParagraphStyle(
    "XJUAnswer", parent=BODY, backColor=colors.HexColor("#f4f7fa"),
    borderColor=colors.HexColor("#d6e0ea"), borderWidth=0.4, borderPadding=5,
    spaceBefore=2, spaceAfter=5,
)
CODE = ParagraphStyle(
    "XJUCode", parent=styles["Code"], fontName="Courier", fontSize=7.7,
    leading=10.2, leftIndent=7, rightIndent=7, backColor=colors.HexColor("#f7f7f7"),
    borderColor=colors.HexColor("#dddddd"), borderWidth=0.3, borderPadding=5,
    spaceBefore=3, spaceAfter=5,
)


def para(text: str, style=BODY) -> Paragraph:
    return Paragraph(esc(text).replace("\n", "<br/>"), style)


def code(text: str) -> Preformatted:
    return Preformatted(text, CODE)


def section(title: str) -> Paragraph:
    return Paragraph(esc(title), H1)


def page_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(18 * mm, 10 * mm, "新疆大学数据结构模拟训练（非官方试题）")
    canvas.drawRightString(192 * mm, 10 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_doc(path: Path, story: list[Any]):
    doc = BaseDocTemplate(
        str(path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=17 * mm,
        title=path.stem, author="Codex",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=page_header_footer)])
    doc.build(story)


def mc_questions(k: int):
    n = 5 + (k * 3) % 8
    cols = 7 + (k * 5) % 6
    base = 1000 + k * 37
    w = 2 + k % 3
    addr = base + (3 * cols + 4) * w
    table_size = 9 + 2 * (k % 4)
    leaves = 6 + k % 5
    q = [
        ("数据结构的逻辑结构与存储结构的关系，正确的是（ ）", ["逻辑结构完全由存储结构决定", "存储结构与逻辑结构没有关系", "同一逻辑结构只能采用一种存储结构", "抽象数据类型的操作由用户定义"], "D", "逻辑结构描述数据之间的关系，存储结构描述数据在存储器中的表示；抽象数据类型允许用户定义数据及其操作。"),
        (f"二维数组 A[0..{n-1}][0..{cols-1}] 按行优先存储，首地址为 {base}，每个元素占 {w} 个存储单元，则 A[3][4] 的地址为（ ）", [str(addr), str(addr - w), str(addr + w), str(addr + 2 * w)], "A", f"行优先地址=首地址+({3}×{cols}+{4})×{w}={addr}。"),
        ("若已知单链表中结点 p 的直接前驱，且插入结点不涉及查找，则插入一个结点的时间复杂度为（ ）", ["O(n)", "O(1)", "O(log n)", "O(n^2)"], "B", "已知前驱后只需修改有限个指针，操作次数与链表长度无关。"),
        ("输入序列为 1,2,3,4,5，采用一个栈，下面不可能得到的出栈序列是（ ）", ["2,1,4,5,3", "3,2,1,5,4", "3,1,2,4,5", "1,2,3,5,4"], "C", "出栈 3 后栈顶必为 2，不能越过 2 先弹出 1。"),
        (f"循环队列容量为 {table_size}，front 指向队头元素、rear 指向下一个入队位置，则当前元素个数为（ ）", [f"(rear-front+{table_size})%{table_size}", "rear-front-1", "rear-front+1", "rear-front"], "A", "采用牺牲一个存储单元区分空、满时，长度为 (rear-front+容量)%容量。"),
        ("含 n 个结点的二叉树采用二叉链表存储，所有空指针域的总数为（ ）", ["n-1", "n", "n+1", "2n"], "C", "共有 2n 个指针域，非空指针为 n-1 条，因此空指针为 n+1 条。"),
        (f"一棵严格二叉树有 {leaves} 个叶结点，则总结点数为（ ）", [str(2 * leaves - 2), str(2 * leaves), str(2 * leaves - 1), str(leaves * leaves)], "C", "严格二叉树满足 n0=n2+1，且总结点 n=n0+n2=2n0-1。"),
        ("森林转换为二叉树时，原树结点的第一个孩子作为左孩子，其后续兄弟结点作为（ ）", ["右孩子", "左孩子", "父结点", "根结点"], "A", "孩子兄弟表示法中，左指针表示第一个孩子，右指针表示下一个兄弟。"),
        ("关于哈夫曼树，正确的是（ ）", ["权值越大的叶子越深", "带权路径长度在所有二叉树中最小", "一定是完全二叉树", "中序序列一定有序"], "B", "哈夫曼算法每次合并权值最小的两个结点，得到最小带权路径长度。"),
        ("下列编码中不是前缀编码的是（ ）", ["(0,10,110,111)", "(0,1,00,11)", "(00,01,10,11)", "(1,01,000,001)"], "B", "编码 0 是 00 的前缀，编码 1 是 11 的前缀，因此不满足前缀码条件。"),
        (f"含 {n+2} 个顶点的无向连通图至少需要（ ）条边", [str(n), str(n + 1), str(n + 2), str(2 * n)], "B", "无向连通图的最少边数是顶点数减一，即 n+2-1=n+1。"),
        ("邻接矩阵适合表示（ ）", ["顶点数较少且边较多的图", "顶点数很多且边很少的图", "所有动态变化的图", "只有树结构"], "A", "邻接矩阵空间为 O(|V|^2)，稠密图中边的存在性判断快。"),
        ("Dijkstra 单源最短路算法的适用条件是（ ）", ["允许存在负权边", "只适用于无向图", "边权非负", "必须是完全图"], "C", "Dijkstra 依赖当前最小距离结点不再被改进，因此要求边权非负。"),
        ("散列表装填因子逐渐接近 1 时，通常会出现（ ）", ["冲突减少", "查找长度增加", "表长自动增加", "所有查找变为 O(1) 严格常数"], "B", "装填因子越高，冲突概率通常越大，平均查找长度上升。"),
        ("下列排序算法中，通常不能保证稳定性的是（ ）", ["直接插入排序", "冒泡排序", "归并排序", "快速排序"], "D", "快速排序可能交换相等关键字的相对位置，通常不稳定。"),
        ("大根堆的根结点保存的是（ ）", ["最大关键字", "最小关键字", "中位数", "最后插入元素"], "A", "大根堆满足每个结点关键字不小于其孩子，根为最大关键字。"),
        ("快速排序在最坏情况下的时间复杂度为（ ）", ["O(n)", "O(log n)", "O(n^2)", "O(n log n)"], "C", "每次划分极不均衡时递归规模为 n-1，比较次数累积为 O(n^2)。"),
        ("KMP 算法相对于朴素模式匹配的核心改进是（ ）", ["不需要模式串", "利用已匹配前缀信息避免主串回退", "只适用于数字串", "把时间复杂度降为 O(1)"], "B", "next/nextval 数组记录模式串前缀和后缀的相等信息，主串指针不回退。"),
        ("广义表 L=(a,(b,c),d)，执行 Tail(L) 的结果是（ ）", ["a", "((b,c),d)", "(b,c)", "(a,(b,c))"], "B", "Tail 去掉表头 a，保留其余元素组成的广义表。"),
        ("下列程序段的时间复杂度为（ ）\nfor i=1..n\n  for j=1..i\n    x=x+1", ["O(n)", "O(log n)", "O(n^2)", "O(n^3)"], "C", "执行次数为 1+2+...+n=n(n+1)/2，故为 O(n^2)。"),
        ("算法必须在执行有限步后结束，这体现了算法的（ ）", ["有穷性", "输入性", "可读性", "稳定性"], "A", "有穷性要求算法不能无限执行，每一步也应在有限时间内完成。"),
        ("带头结点单链表中的头结点主要作用是（ ）", ["保存第一个有效数据", "统一空表和非空表的边界处理", "保证链表有序", "使查找变为 O(1)"], "B", "头结点通常不保存有效业务数据，可减少首元结点插入、删除时的特殊分支。"),
        ("两个栈共享数组 A[0..m-1]，分别从两端向中间增长。设栈顶为 top1、top2，则栈满条件是（ ）", ["top1==top2", "top1+1==top2", "top1==m-1", "top2==0"], "B", "两个栈顶相邻时已无空闲单元，再入栈就会溢出。"),
        ("稀疏矩阵的三元组顺序表通常为每个非零元素保存（ ）", ["行号、列号和值", "行号和地址", "列号和指针", "值和颜色"], "A", "三元组用 (row,col,value) 描述每个非零元素。"),
        ("计算后缀表达式的值时，最适合使用的数据结构是（ ）", ["队列", "栈", "二叉搜索树", "散列表"], "B", "扫描到操作数时入栈，扫描到运算符时弹出操作数计算并将结果重新入栈。"),
        ("完全二叉树按层序从1开始编号，编号为 i 的结点若有左孩子，其编号为（ ）", ["i+1", "2i", "2i+1", "i/2"], "B", "一维数组存储完全二叉树时，左孩子下标为 2i，右孩子为 2i+1。"),
        ("在结点关键字互不相同的前提下，能够唯一确定一棵二叉树的遍历序列组合是（ ）", ["先序和中序", "先序和后序", "层序和先序", "两个相同的中序序列"], "A", "中序负责切分左右子树，先序首元素负责确定根，因此可以递归唯一还原。"),
        ("线索二叉树利用原二叉链表中的空指针域保存（ ）", ["结点权值", "遍历序列中的前驱或后继", "树的高度", "哈希地址"], "B", "线索化将部分空左、右指针改作特定遍历次序下的前驱、后继线索。"),
        ("二叉搜索树退化为单支树时，最坏查找时间复杂度为（ ）", ["O(1)", "O(log n)", "O(n)", "O(n log n)"], "C", "树高可能达到 n，查找需要沿单支路径逐个比较。"),
        ("AVL 树中任一结点的平衡因子允许取值为（ ）", ["仅0", "-1、0、1", "0、1、2", "任意整数"], "B", "AVL 树要求左右子树高度差的绝对值不超过1。"),
        ("有向图中全部顶点入度之和与出度之和的关系为（ ）", ["入度和是出度和两倍", "两者均等于边数", "两者之和等于边数", "无法确定"], "B", "每条有向边对一个顶点贡献1个出度，对另一个顶点贡献1个入度。"),
        ("对一个有向图执行拓扑排序，若最终输出顶点数小于图的顶点总数，则说明（ ）", ["图不连通", "图中存在有向环", "图没有边", "图是完全图"], "B", "有向环中的顶点无法出现入度为0的可选结点，因此不能输出全部顶点。"),
        ("对于边较多的稠密连通网，构造最小生成树时通常更适合采用（ ）", ["Prim 算法", "KMP 算法", "二分查找", "拓扑排序"], "A", "Prim 以顶点为中心扩张，采用邻接矩阵时对稠密图较方便。"),
        ("Kruskal 算法构造最小生成树时，判断加入一条边是否形成回路常使用（ ）", ["栈", "并查集", "循环队列", "模式串"], "B", "并查集可高效判断两个端点是否已属于同一连通分量。"),
        ("Floyd 算法主要用于求解（ ）", ["单源无权最短路", "所有顶点对之间的最短路", "最小生成树", "拓扑序列"], "B", "Floyd 通过逐步允许中间顶点更新任意顶点对距离，时间复杂度 O(n^3)。"),
        ("无向图采用邻接表存储时，空间复杂度通常为（ ）", ["O(|V|+|E|)", "O(|V|^2)", "O(log |V|)", "O(1)"], "A", "顶点表占 O(|V|)，每条无向边在边表中出现两次，仍为 O(|E|)。"),
        (f"在含 {16+k} 个元素的有序顺序表中进行折半查找，成功查找的比较次数数量级为（ ）", ["O(1)", "O(log n)", "O(n)", "O(n^2)"], "B", "每次比较把候选区间约缩小一半，因此比较次数为对数数量级。"),
        ("散列冲突采用拉链法处理时，同一散列地址上的关键字通常存放在（ ）", ["同一个链表中", "递归栈中", "有序数组的连续空位中", "二叉堆根部"], "A", "拉链法为每个桶维护同义词链表，冲突关键字挂在同一桶的链上。"),
        ("关于希尔排序，正确的是（ ）", ["一定稳定", "只能处理链表", "通常不稳定", "最坏时间恒为 O(n)"], "C", "不同增量组之间的移动可能改变相等关键字的相对次序。"),
        ("用数组实现堆排序时，除递归或少量临时变量外，其额外空间复杂度通常为（ ）", ["O(1)", "O(n)", "O(n log n)", "O(n^2)"], "A", "堆可直接在原数组上调整，属于原地排序。"),
    ]
    # Rotate options for visual variation while preserving the answer key.
    shift = k % 4
    out = []
    for i, (stem, opts, ans, exp) in enumerate(q):
        if i in {1, 4, 6, 10}:
            stem = stem.replace(str(n), str(n + (k % 3))).replace(str(cols), str(cols + (k % 2)))
        out.append({"stem": stem, "options": opts, "answer": ans, "explain": exp})
    # Forty-question bank, each paper draws a different twenty-question combination.
    start = (k * 7) % len(out)
    picked = [(start + j * 13) % len(out) for j in range(20)]
    return [out[i] for i in picked]


def tf_questions(k: int):
    items = [
        ("顺序表支持按下标随机访问，访问时间通常为 O(1)。", True, "数组地址可由下标和首地址直接计算。"),
        ("单链表适合直接进行二分查找，因为链表结点可以按下标访问。", False, "单链表不支持 O(1) 随机访问，二分查找缺少高效的中点定位。"),
        ("栈的插入和删除只能在同一端进行。", True, "该端称为栈顶，遵循后进先出。"),
        ("BFS 通常使用队列，DFS 通常使用栈或递归。", True, "二者的辅助结构不同，访问顺序也不同。"),
        ("无权图中，从源点开始的 BFS 可以得到最短边数路径。", True, "BFS 按层访问，首次到达结点时边数最少。"),
        ("二叉搜索树的中序遍历序列一定是关键字递减序列。", False, "通常是递增序列；递减需要按相反规则或反向中序。"),
        ("堆是一个完全二叉树，但其层序序列不一定整体有序。", True, "堆只要求父子之间满足堆序关系。"),
        ("归并排序的合并阶段需要额外辅助空间来暂存结果。", True, "常规数组归并需要 O(n) 辅助空间。"),
        ("KMP 的最坏时间复杂度为 O(mn)，其中 m、n 分别为模式串和主串长度。", False, "标准 KMP 为 O(m+n)。"),
        ("Dijkstra 算法可以直接保证含负权边图的最短路径正确。", False, "存在负权边时应谨慎，Dijkstra 的贪心前提不成立。"),
        ("数据的逻辑结构取决于数据元素之间的关系，而不是某一种具体编程语言。", True, "逻辑结构是抽象关系，可用不同语言和存储结构实现。"),
        ("循环队列只要 front==rear 就一定表示队列已满。", False, "在常见设计中 front==rear 表示空；区分空和满还需牺牲单元、计数器或标志位。"),
        ("哈夫曼编码属于前缀编码，可以从编码流中唯一译码。", True, "任一字符编码都不是另一个字符编码的前缀。"),
        ("一个有向无环图的拓扑序列一定唯一。", False, "同时存在多个入度为0的顶点时通常可以产生多个合法拓扑序列。"),
        ("含 n 个顶点的无向连通图边数至少为 n-1。", True, "生成树恰有 n-1 条边，是保持连通所需的最少边数。"),
        ("折半查找要求查找表有序且能够进行随机访问。", True, "需要高效定位区间中点，因此通常用于有序顺序表。"),
        ("线性探测散列表删除关键字时，直接把单元恢复为空可能截断后续同义词的查找链。", True, "通常需要设置删除标记，而不是简单改为空。"),
        ("简单选择排序在任何情况下都是稳定排序。", False, "交换最小元素与首元素时可能跨越相等关键字，通常不稳定。"),
        ("比较排序在一般模型下的最坏比较次数存在 O(n log n) 的下界。", True, "决策树至少需要区分 n! 种排列，其高度为 Omega(n log n)。"),
        ("邻接矩阵中判断两个指定顶点是否邻接通常为 O(1)。", True, "可直接访问矩阵对应单元。"),
    ]
    # Change two contextual words each set so papers are not visually identical.
    if k % 2:
        items[0] = ("顺序表在已知下标时可以进行 O(1) 的随机访问。", True, items[0][2])
        items[7] = ("常规数组归并排序的合并过程通常需要辅助数组。", True, items[7][2])
    start = (k * 3) % len(items)
    picked = [(start + j * 7) % len(items) for j in range(10)]
    return [items[i] for i in picked]


def fill_questions(k: int):
    rows = [
        ("数据的基本单位是____；数据对象是____的集合。", "数据元素；性质相同的数据元素", "数据元素是数据的基本单位，数据对象是同一性质数据元素的集合。"),
        ("数据的四种基本逻辑结构通常概括为____、____、____和____。", "集合；线性；树形；图状", "四类结构描述数据元素之间不同的关系密度和层次。"),
        ("评价算法的主要指标包括____、____、____和____。", "正确性；可读性；时间复杂度；空间复杂度", "正确性是首要条件，效率通常从时间和空间两方面评价，可读性影响实现与维护。"),
        ("二叉树中度为0的结点数 n0 与度为2的结点数 n2 满足____。", "n0=n2+1", "由树的边数等于总结点数减一，并按结点度数计数可得。"),
        (f"若 int A[4][{6+k%4}] 按行优先存储，首地址为1000、元素宽度为2，则 A[2][{2+k%3}] 地址为____。", str(1000 + (2 * (6 + k % 4) + (2 + k % 3)) * 2), "行优先地址=首地址+(行下标×列数+列下标)×元素宽度。"),
        ("两个栈共享同一数组并分别从两端增长时，栈满条件是____。", "top1+1==top2", "两栈顶相邻意味着中间已没有可分配的数组单元。"),
        (f"含 {8+k%5} 个顶点的无向连通图至少有____条边。", str(7+k%5), "连通图至少包含一棵生成树，生成树边数为顶点数减一。"),
        ("二叉搜索树按中序遍历得到的关键字序列通常是____序列。", "非递减有序", "左子树关键字不大于根，右子树关键字不小于根。"),
        ("散列表中，装填因子 α 通常定义为____。", "表中记录数/散列表长度", "装填因子反映散列表占用程度，并影响冲突概率和平均查找长度。"),
        (f"一棵严格二叉树有 {5+k%4} 个叶结点，则总结点数为____。", str(2*(5+k%4)-1), "严格二叉树满足度为0的结点数比度为2的结点数多1。"),
    ]
    all_rows = [{"stem": a, "answer": b, "explain": c} for a, b, c in rows]
    start = (k * 2) % len(all_rows)
    picked = [(start + j * 3) % len(all_rows) for j in range(5)]
    return [all_rows[i] for i in picked]


def tree_data(k: int):
    if k % 2 == 0:
        pre = ["A", "B", "D", "E", "C", "F", "G"]
        ino = ["D", "B", "E", "A", "F", "C", "G"]
    else:
        pre = ["A", "B", "D", "C", "E", "F", "G"]
        ino = ["D", "B", "A", "E", "C", "G", "F"]
    return pre, ino


def tree_post(pre: list[str], ino: list[str]) -> list[str]:
    if not pre:
        return []
    root = pre[0]
    p = ino.index(root)
    return tree_post(pre[1:1+p], ino[:p]) + tree_post(pre[1+p:], ino[p+1:]) + [root]


def huffman_steps(weights: list[int]):
    heap = list(weights)
    heapq.heapify(heap)
    steps = []
    wpl = 0
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        c = a + b
        steps.append((a, b, c))
        wpl += c
        heapq.heappush(heap, c)
    return steps, wpl


def dijkstra(edges: list[tuple[str, str, int]], s: str):
    g: dict[str, list[tuple[str, int]]] = {}
    for a, b, w in edges:
        g.setdefault(a, []).append((b, w))
        g.setdefault(b, []).append((a, w))
    dist = {v: 10**9 for v in g}
    prev = {v: None for v in g}
    dist[s] = 0
    used = set()
    for _ in g:
        cand = [(dist[v], v) for v in g if v not in used]
        _, u = min(cand)
        used.add(u)
        for v, w in g[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
    return dist, prev


def hash_insert(keys: list[int], m: int):
    table = [None] * m
    probes = []
    for key in keys:
        pos = key % m
        count = 1
        while table[pos] is not None:
            pos = (pos + 1) % m
            count += 1
            if count > m:
                raise ValueError("hash table full")
        table[pos] = key
        probes.append(count)
    return table, probes, sum(probes) / len(probes)


def application_questions(k: int):
    pre, ino = tree_data(k)
    post = tree_post(pre, ino)
    weights = [3 + ((k + i * 2) % 9) for i in range(6)]
    hs, wpl = huffman_steps(weights)
    a = 2 + k % 3
    b = 5 + (k * 2) % 4
    c = 2 + (k * 3) % 5
    d = 4 + (k * 5) % 5
    e = 3 + (k * 7) % 6
    f = 6 + (k * 2) % 5
    g = 2 + (k * 4) % 4
    edges = [("A", "B", a), ("A", "C", b), ("B", "C", c), ("B", "D", d), ("C", "D", e), ("C", "E", f), ("D", "E", g)]
    dist, prev = dijkstra(edges, "A")
    m = 11 + 2 * (k % 3)
    keys = [19 + k, 30 + 2 * k, 41 + 3 * k, 52 + k, 63 + 2 * k, 74 + 3 * k]
    table, probes, asl = hash_insert(keys, m)
    if k % 3 == 0:
        arr = [8 + k, 3 + k % 4, 6 + k, 1 + k % 3, 5 + k, 2 + k % 5]
        sorted_arr = arr[:]
        for i in range(1, len(sorted_arr)):
            x = sorted_arr[i]
            j = i - 1
            while j >= 0 and sorted_arr[j] > x:
                sorted_arr[j + 1] = sorted_arr[j]
                j -= 1
            sorted_arr[j + 1] = x
        sort_stem = f"对序列 {arr} 做直接插入排序，写出第3趟结束后的序列，并说明稳定性与最坏时间复杂度。"
        # Use a prefix after three passes for the answer.
        partial = arr[:]
        for i in range(1, min(4, len(partial))):
            x = partial[i]; j = i - 1
            while j >= 0 and partial[j] > x:
                partial[j + 1] = partial[j]; j -= 1
            partial[j + 1] = x
        sort_ans = f"前三趟结果为 {partial}；直接插入排序稳定，最坏时间复杂度 O(n^2)。"
    elif k % 3 == 1:
        arr = [31 + k, 14 + k % 5, 27 + k, 9 + k % 4, 22 + k, 18 + k % 6]
        first = sorted(arr[:2])
        sort_stem = f"对序列 {arr} 做一趟自底向上的二路归并，写出本趟各相邻长度2子序列归并后的结果，并说明空间复杂度。"
        merged = []
        for i in range(0, len(arr), 2):
            merged.extend(sorted(arr[i:i+2]))
        sort_ans = f"本趟结果为 {merged}；常规数组归并排序需要 O(n) 辅助空间，时间复杂度为 O(n log n)。"
    else:
        arr = [26 + k, 11 + k % 4, 34 + k, 7 + k % 3, 19 + k, 4 + k % 5]
        first_pass = arr[:]
        mi = min(range(len(first_pass)), key=first_pass.__getitem__)
        first_pass[0], first_pass[mi] = first_pass[mi], first_pass[0]
        sort_stem = f"对序列 {arr} 做一趟简单选择排序，写出本趟结束后的序列，并说明其交换次数上界和稳定性。"
        sort_ans = f"第一趟结果为 {first_pass}；每趟选择最小值，交换次数至多 n-1，但简单选择排序通常不稳定，时间复杂度 O(n^2)。"
    return [
        {"stem": f"由二叉树的先序序列 {','.join(pre)} 和中序序列 {','.join(ino)} 唯一确定该树，写出后序序列并说明还原过程。", "answer": f"后序序列为 {','.join(post)}。先序首元素确定根，在中序中切分左右子树，再递归处理左右部分。", "explain": "先序的第一个元素必为当前树根；中序中根左边属于左子树，右边属于右子树，递归即可还原。"},
        {"stem": f"对权值序列 {weights} 构造哈夫曼树，写出每次合并的权值并求 WPL。", "answer": f"合并过程：" + "；".join(f"{x}+{y}={z}" for x, y, z in hs) + f"；WPL={wpl}。", "explain": "每次从当前森林中取最小的两个权值合并，合并值累加即为所有叶子的带权路径长度。"},
        {"stem": f"无向带权图边为 {edges}，用 Dijkstra 从 A 出发，求 A 到各点最短距离及一条最短路径。", "answer": "；".join(f"A到{v}={dist[v]}" for v in sorted(dist)) + "。每次选取当前未确定且距离最小的顶点进行松弛。", "explain": "边权均非负，Dijkstra 的贪心选择成立；按距离从小到大确定顶点并更新邻接点。"},
        {"stem": f"散列表长度为 {m}，采用 h(k)=k mod {m}、线性探测处理冲突，依次插入关键字 {keys}，画出表并求成功查找 ASL。", "answer": f"最终表（下标0开始）={table}；各关键字探测次数={probes}；成功查找 ASL={asl:.2f}。", "explain": "插入时从同义地址开始逐格向后探测，成功查找的比较次数等于该关键字插入时的探测次数，ASL为其平均值。"},
        {"stem": sort_stem, "answer": sort_ans, "explain": "按题目指定的排序过程逐趟记录序列；稳定性要看相等关键字的相对次序是否保持，复杂度要按最坏输入分析。"},
    ]


def code_questions(k: int):
    x = 2 + k % 4
    code1 = f"""int Count(LinkNode *L, ElemType x)\n{{\n    int count = 0;\n    LinkNode *p = L->next;\n    while (p != L) {{\n        if (p->data % {x} == 0) count++;\n        p = p->next;\n    }}\n    return count;\n}}"""
    code2 = """void Visit(BTNode *t)\n{\n    if (t != NULL) {\n        Visit(t->lchild);\n        printf(\"%d \", t->data);\n        Visit(t->rchild);\n    }\n}"""
    return [
        {"stem": "阅读下列循环单链表代码，说明函数功能，并给出时间复杂度和空间复杂度。", "code": code1, "answer": f"统计循环链表中 data 能被 {x} 整除的结点数；每个结点访问一次，时间复杂度 O(n)，只使用指针和计数器，空间复杂度 O(1)。", "explain": "初始化 p=L->next，从首元结点开始，遇到头结点 L 停止，因此恰好遍历一周；条件判断成立时计数。"},
        {"stem": "阅读下列二叉树代码，说明遍历次序，并给出其时间复杂度。若树的中序序列为 D,B,E,A,F,C,G，写出输出序列。", "code": code2, "answer": "函数执行左子树、访问根、右子树，属于中序遍历；给定中序序列时输出为 D B E A F C G；每个结点访问一次，时间复杂度 O(n)，递归栈空间最坏 O(n)。", "explain": "访问语句位于两次递归调用之间，因此是中序；递归树对每个结点仅处理一次。"},
    ]


def algorithm_questions(k: int):
    second = (
        "请设计 KMP 模式匹配算法，给出 next 数组的求法、匹配过程和时间复杂度。"
        if k % 2 == 0 else
        "请设计无权图中从源点 s 到目标点 t 的 BFS 最短路径算法，要求输出路径并分析复杂度。"
    )
    if k % 2 == 0:
        ans2 = "先对模式串计算 next：next[0]=-1，令 j=next[j] 逐步回退，直到字符相等或 j=-1；匹配时若相等则 i、j 同时后移，否则 j=next[j]，j=-1 时 i、j 同时后移。时间复杂度 O(n+m)，空间复杂度 O(m)。"
    else:
        ans2 = "使用队列，从 s 入队并令 parent[s]=-1；每次出队 u，扫描其邻接点，首次访问 v 时记录 parent[v]=u 并入队；到达 t 后沿 parent 反向回溯再逆序输出。邻接表实现时间 O(V+E)，辅助空间 O(V)。"
    return [
        {"stem": "请设计一个算法，将两个按非递减顺序排列的单链表 A、B 合并为一个非递减单链表，要求尽量复用原结点，并分析复杂度。", "answer": "设置 dummy 头结点和尾指针 tail；当 A、B 均非空时比较当前结点，将较小者接到 tail 后并后移对应指针；循环结束后把剩余链表接到 tail 后，返回 dummy->next。时间复杂度 O(m+n)，额外空间 O(1)。相等时可优先接 A 以保持稳定性。", "explain": "每个结点只被比较和摘接一次；复用结点无需申请数组，dummy 头结点可统一处理结果为空和非空的情况。"},
        {"stem": second, "answer": ans2, "explain": "算法题必须写清初始化、循环条件、边界情况、输出方式和复杂度；仅写思路而没有可执行的伪代码通常不能得到满分。"},
    ]


def exam_header(k: int):
    return [
        Paragraph(f"新疆大学数据结构专业课模拟卷 {k:02d}", TITLE),
        Paragraph("适用范围：数据结构教程第5版相关章节；不含外排序、红黑树及其后续章节。仅供复习检测，非官方预测题。", SUBTITLE),
        para("建议用时：180分钟；满分：150分。请先独立完成试卷，再使用答案卷复盘。不同版本的真题在题型和总分表述上存在差异，本套统一采用近年常见题型进行训练。", SMALL),
        Spacer(1, 4),
    ]


def paper_story(k: int):
    story = exam_header(k)
    story.append(section("一、单项选择题（20题，每题2分，共40分）"))
    for i, q in enumerate(mc_questions(k), 1):
        opts = "；".join(f"{chr(65+j)}. {o}" for j, o in enumerate(q["options"]))
        story.append(para(f"{i}. {q['stem']}"))
        story.append(para(opts, SMALL))
    story.append(section("二、判断题（10题，每题1分，共10分）"))
    for i, (stem, _, _) in enumerate(tf_questions(k), 1):
        story.append(para(f"{i}. {stem}（正确/错误）"))
    story.append(section("三、填空题（5题，每题2分，共10分）"))
    for i, q in enumerate(fill_questions(k), 1):
        story.append(para(f"{i}. {q['stem']}"))
    story.append(section("四、应用与综合题（5题，每题8分，共40分）"))
    for i, q in enumerate(application_questions(k), 1):
        story.append(para(f"{i}. {q['stem']}"))
    story.append(section("五、代码分析题（2题，每题5分，共10分）"))
    for i, q in enumerate(code_questions(k), 1):
        story.append(para(f"{i}. {q['stem']}"))
        story.append(code(q["code"]))
    story.append(section("六、算法设计题（2题，每题20分，共40分）"))
    for i, q in enumerate(algorithm_questions(k), 1):
        story.append(para(f"{i}. {q['stem']}"))
    story.append(para("答题提醒：算法题需要写出关键伪代码或 C/C++ 风格代码，不能只写一句“采用某算法”；图、树、哈夫曼和哈希题应写中间过程。", ANSWER))
    return story


def answer_story(k: int):
    story = exam_header(k)
    story.append(section("答案与逐题解析"))
    story.append(para("使用方式：先遮住本页答案，按题号核对；错题不要只记选项，应把解析中的条件、公式和边界情况写入错题本。", SMALL))
    story.append(Paragraph("一、单项选择题", H2))
    for i, q in enumerate(mc_questions(k), 1):
        story.append(para(f"{i}. 答案：{q['answer']}；{q['explain']}", ANSWER))
    story.append(Paragraph("二、判断题", H2))
    for i, (_, ans, exp) in enumerate(tf_questions(k), 1):
        story.append(para(f"{i}. {'正确' if ans else '错误'}；{exp}", ANSWER))
    story.append(Paragraph("三、填空题", H2))
    for i, q in enumerate(fill_questions(k), 1):
        story.append(para(f"{i}. 答案：{q['answer']}；{q['explain']}", ANSWER))
    story.append(Paragraph("四、应用与综合题", H2))
    for i, q in enumerate(application_questions(k), 1):
        story.append(para(f"{i}. 参考答案：{q['answer']}", ANSWER))
        story.append(para(f"解析：{q['explain']}", SMALL))
    story.append(Paragraph("五、代码分析题", H2))
    for i, q in enumerate(code_questions(k), 1):
        story.append(para(f"{i}. 参考答案：{q['answer']}", ANSWER))
        story.append(para(f"解析：{q['explain']}", SMALL))
    story.append(Paragraph("六、算法设计题", H2))
    for i, q in enumerate(algorithm_questions(k), 1):
        story.append(para(f"{i}. 参考答案：{q['answer']}", ANSWER))
        story.append(para(f"评分要点与解析：{q['explain']}", SMALL))
    return story


def main():
    papers = []
    answers = []
    for k in range(1, 21):
        papers.extend(paper_story(k))
        if k != 20:
            papers.append(PageBreak())
        answers.extend(answer_story(k))
        if k != 20:
            answers.append(PageBreak())
    paper_path = OUT / "xju_data_structure_20_mock_exams.pdf"
    answer_path = OUT / "xju_data_structure_20_mock_answers_analysis.pdf"
    build_doc(paper_path, papers)
    build_doc(answer_path, answers)
    print(paper_path)
    print(answer_path)
    print(f"paper_bytes={paper_path.stat().st_size}")
    print(f"answer_bytes={answer_path.stat().st_size}")


if __name__ == "__main__":
    main()
