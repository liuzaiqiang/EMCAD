from pathlib import Path
import re
from PIL import Image
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, PageBreak, Image as RLImage,
                                KeepTogether)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape

ROOT = Path.cwd()
TMP = ROOT / 'tmp' / 'pdfs'
MD = TMP / 'EMCAD中文全译.md'
SRC = Path(r'C:\Users\lzq\Desktop\EMCAD：Efficient Multi-scale Convolutional Attention Decoding for Medical.pdf')
OUT = ROOT / 'docs' / 'EMCAD论文中文全译.pdf'

# Crop only the original plots, architecture diagrams, qualitative panels, and
# numeric tables. Captions are typeset in Chinese in the translated document.
crops = {
    'FIGURE 1': ('page2.png', (130, 105, 455, 355)),
    'FIGURE 2': ('page4.png', (120, 110, 900, 470)),
    'FIGURE 3': ('page7.png', (130, 100, 460, 365)),
    'FIGURE 4': ('page13.png', (80, 110, 925, 280)),
    'FIGURE 5': ('page13.png', (175, 330, 820, 790)),
    'TABLE 1': ('page6.png', (95, 135, 1080, 505)),
    'TABLE 2': ('page6.png', (95, 580, 1080, 965)),
    'TABLE 3': ('page7.png', (80, 455, 500, 745)),
    'TABLE 4': ('page7.png', (500, 110, 930, 250)),
    'TABLE 5': ('page8.png', (100, 135, 1030, 235)),
    'TABLE 6': ('page8.png', (100, 290, 575, 420)),
    'TABLE 7': ('page13.png', (75, 855, 510, 962)),
    'TABLE 8': ('page13.png', (510, 855, 925, 962)),
    'TABLE 9': ('page14.png', (90, 130, 1060, 305)),
    'TABLE 10': ('page14.png', (90, 410, 565, 500)),
    'TABLE 11': ('page14.png', (90, 555, 565, 805)),
}
for name, (filename, box) in crops.items():
    im = Image.open(TMP / filename).convert('RGB')
    im.crop(box).save(TMP / (name.lower().replace(' ', '_') + '.png'))

font_path = Path(r'C:\Windows\Fonts\msyh.ttc')
if not font_path.exists():
    font_path = Path(r'C:\Windows\Fonts\simhei.ttf')
pdfmetrics.registerFont(TTFont('Chinese', str(font_path), subfontIndex=0 if font_path.suffix.lower()=='.ttc' else 0))
pdfmetrics.registerFontFamily('Chinese', normal='Chinese', bold='Chinese', italic='Chinese', boldItalic='Chinese')

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='CNTitle', fontName='Chinese', fontSize=22, leading=31,
                          alignment=TA_CENTER, textColor=colors.HexColor('#163A5A'),
                          spaceAfter=12))
styles.add(ParagraphStyle(name='CNMeta', fontName='Chinese', fontSize=10, leading=16,
                          alignment=TA_CENTER, textColor=colors.HexColor('#455A64'), spaceAfter=5))
styles.add(ParagraphStyle(name='CNBody', fontName='Chinese', fontSize=10.2, leading=17,
                          alignment=TA_LEFT, firstLineIndent=20, spaceAfter=6,
                          wordWrap='CJK'))
styles.add(ParagraphStyle(name='CNQuote', parent=styles['CNBody'], leftIndent=15,
                          rightIndent=12, firstLineIndent=0, backColor=colors.HexColor('#F3F6F8'),
                          borderColor=colors.HexColor('#8AA7B8'), borderWidth=0.6,
                          borderPadding=7, spaceBefore=4, spaceAfter=9))
styles.add(ParagraphStyle(name='CNH1', fontName='Chinese', fontSize=17, leading=24,
                          textColor=colors.HexColor('#163A5A'), spaceBefore=18, spaceAfter=10,
                          keepWithNext=True, wordWrap='CJK'))
styles.add(ParagraphStyle(name='CNH2', fontName='Chinese', fontSize=14, leading=21,
                          textColor=colors.HexColor('#205A78'), spaceBefore=15, spaceAfter=8,
                          keepWithNext=True, wordWrap='CJK'))
styles.add(ParagraphStyle(name='CNH3', fontName='Chinese', fontSize=12, leading=18,
                          textColor=colors.HexColor('#317187'), spaceBefore=12, spaceAfter=6,
                          keepWithNext=True, wordWrap='CJK'))
styles.add(ParagraphStyle(name='CNH4', fontName='Chinese', fontSize=10.7, leading=17,
                          textColor=colors.HexColor('#37474F'), spaceBefore=9, spaceAfter=5,
                          keepWithNext=True, wordWrap='CJK'))
styles.add(ParagraphStyle(name='CNTOC', fontName='Chinese', fontSize=10, leading=16,
                          leftIndent=10, firstLineIndent=0, spaceAfter=1, wordWrap='CJK'))
styles.add(ParagraphStyle(name='CNRef', fontName='Chinese', fontSize=8.5, leading=12,
                          leftIndent=22, firstLineIndent=-22, spaceAfter=4, wordWrap='CJK'))
styles.add(ParagraphStyle(name='CNCaption', fontName='Chinese', fontSize=9, leading=14,
                          alignment=TA_LEFT, spaceBefore=3, spaceAfter=8, wordWrap='CJK'))

class Doc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ('CNH1','CNH2','CNH3','CNH4'):
            if self.page < 3:
                return
            level = {'CNH1':0,'CNH2':1,'CNH3':2,'CNH4':3}[flowable.style.name]
            text = flowable.getPlainText()
            key = 'h%d-%d' % (level, self.page)
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=(level>1))
            self.notify('TOCEntry', (level, text, self.page, key))

def page_chrome(canvas, doc):
    canvas.saveState()
    w, h = A4
    if doc.page > 1:
        canvas.setFont('Chinese', 8)
        canvas.setFillColor(colors.HexColor('#607D8B'))
        canvas.drawString(22*mm, h-14*mm, 'EMCAD 论文中文全译')
        canvas.setStrokeColor(colors.HexColor('#D8E1E6'))
        canvas.line(22*mm, h-16*mm, w-22*mm, h-16*mm)
        canvas.drawCentredString(w/2, 12*mm, str(doc.page-2))
    canvas.restoreState()

doc = Doc(str(OUT), pagesize=A4, leftMargin=22*mm, rightMargin=22*mm,
          topMargin=23*mm, bottomMargin=20*mm, title='EMCAD论文中文全译',
          author='OpenAI')
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
doc.addPageTemplates([PageTemplate(id='main', frames=frame, onPage=page_chrome)])

toc = TableOfContents()
toc.levelStyles = [
    ParagraphStyle(name='TOC0', fontName='Chinese', fontSize=10.5, leading=17,
                   leftIndent=0, firstLineIndent=0, spaceBefore=5, wordWrap='CJK'),
    ParagraphStyle(name='TOC1', fontName='Chinese', fontSize=9.5, leading=15,
                   leftIndent=14, firstLineIndent=0, wordWrap='CJK'),
    ParagraphStyle(name='TOC2', fontName='Chinese', fontSize=8.8, leading=13,
                   leftIndent=27, firstLineIndent=0, wordWrap='CJK'),
    ParagraphStyle(name='TOC3', fontName='Chinese', fontSize=8.2, leading=12,
                   leftIndent=40, firstLineIndent=0, wordWrap='CJK'),
]

def inline_markup(s):
    s = escape(s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', s)
    return s

story = [Spacer(1, 35*mm), Paragraph('EMCAD：面向医学图像分割的高效多尺度卷积注意力解码', styles['CNTitle']),
         Paragraph('完整中文译本', styles['CNMeta']), Spacer(1, 8*mm),
         Paragraph('原文题名：EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation', styles['CNMeta']),
         Paragraph('作者：Md Mostafijur Rahman、Mustafa Munir、Radu Marculescu｜德克萨斯大学奥斯汀分校', styles['CNMeta']),
         Paragraph('原文：arXiv:2405.06880v1（2024 年 5 月 11 日）', styles['CNMeta']), Spacer(1, 15*mm),
         Paragraph('译文说明', styles['CNH2']),
         Paragraph('EMCAD、MSCAM、MSCB、MSDC、LGAG、CAB、SAB、EUCB、PVTv2 等模型名和模块缩写均保留；Dice、IoU、HD95、FLOPs 等术语按医学图像分割领域惯例处理。图表保留原始数值与图形，题注和上下文说明翻译为中文。参考文献保留英文原始著录以便检索。', styles['CNBody']),
         PageBreak(), Paragraph('目录', styles['CNH1']), toc, PageBreak()]

md_text = MD.read_text(encoding='utf-8')
lines = md_text.splitlines()
skip_title = True
buffer=[]
ref_pages = PdfReader(str(SRC)).pages[8:11]
refs_raw = '\n'.join(p.extract_text() or '' for p in ref_pages)
refs_raw = re.sub(r'(?m)^\s*\d+\s*$', '', refs_raw)
refs_raw = refs_raw.replace('-\n', '').replace('\n', ' ')
refs = re.split(r'(?=\[\d+\])', refs_raw)
refs = [re.sub(r'\s+', ' ', x).strip() for x in refs if re.match(r'\s*\[\d+\]', x)]

def flush():
    global buffer
    if buffer:
        p = ' '.join(x.strip() for x in buffer).strip()
        if p:
            style = styles['CNQuote'] if p.startswith('&gt;') or p.startswith('>') else styles['CNBody']
            p = re.sub(r'^&gt;\s*|^>\s*', '', p)
            story.append(Paragraph(inline_markup(p), style))
        buffer=[]

for line in lines:
    if skip_title:
        if line.startswith('## 摘要'):
            skip_title=False
        else:
            continue
    if not line.strip():
        flush(); continue
    if line.startswith('#'):
        flush()
        n = len(line)-len(line.lstrip('#'))
        title = line[n:].strip()
        style_name = {1:'CNH1',2:'CNH1',3:'CNH2',4:'CNH3'}.get(n,'CNH3')
        story.append(Paragraph(inline_markup(title), styles[style_name]))
        continue
    marker = re.match(r'^\[\[(FIGURE \d+|TABLE \d+|REFERENCES)\]\]$', line.strip())
    if marker:
        flush()
        name = marker.group(1)
        if name == 'REFERENCES':
            for ref in refs:
                story.append(Paragraph(inline_markup(ref), styles['CNRef']))
        else:
            img_path = TMP / (name.lower().replace(' ', '_') + '.png')
            with Image.open(img_path) as im:
                iw, ih = im.size
            maxw, maxh = doc.width-5*mm, 125*mm
            scale = min(maxw/iw, maxh/ih, 1.0)
            image = RLImage(str(img_path), width=iw*scale, height=ih*scale, kind='proportional')
            image.hAlign='CENTER'
            story.append(KeepTogether([Spacer(1,2*mm), image, Spacer(1,2*mm)]))
        continue
    buffer.append(line)
flush()

doc.multiBuild(story)
print(f'created {OUT} ({OUT.stat().st_size} bytes), references={len(refs)}')
