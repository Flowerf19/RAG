# revised from https://github.com/opendatalab/MinerU/blob/7f0fe20004af7416db886f4b75c116bcc1c986b4/magic_pdf/pdf_parse_union_core.py#L177
# from fast_langdetect import detect_language
# import unicodedata
import re


def __is_overlaps_y_exceeds_threshold(bbox1, bbox2, overlap_ratio_threshold=0.8):
    """Check if two bboxes overlap on the y-axis, and the overlap height exceeds 80% of the shorter bbox's height"""
    _, y0_1, _, y1_1 = bbox1
    _, y0_2, _, y1_2 = bbox2

    overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2))
    height1, height2 = y1_1 - y0_1, y1_2 - y0_2
    max_height = max(height1, height2)
    min_height = min(height1, height2)

    return (overlap / min_height) > overlap_ratio_threshold

def merge_spans_to_line(spans):
    if len(spans) == 0:
        return []
    else:
        # Sort by y0 coordinate
        spans.sort(key=lambda span: span['bbox'][1])

        lines = []
        current_line = [spans[0]]
        for span in spans[1:]:
            # If current span type is "isolated" or current line already has "isolated"
            # Same applies to image and table types
            if span['type'] in ['isolated'] or any(
                    s['type'] in ['isolated'] for s in
                    current_line):
                # Start a new line
                lines.append(current_line)
                current_line = [span]
                continue

            # If current span overlaps with the last span of current line on y-axis, add to current line
            if __is_overlaps_y_exceeds_threshold(span['bbox'], current_line[-1]['bbox']):
                current_line.append(span)
            else:
                # Otherwise, start a new line
                lines.append(current_line)
                current_line = [span]

        # Add the last line
        if current_line:
            lines.append(current_line)

        return lines

# Sort spans in each line from left to right
def line_sort_spans_by_left_to_right(lines):
    line_objects = []
    for line in lines:
        # Sort by x0 coordinate
        line.sort(key=lambda span: span['bbox'][0])
        line_bbox = [
            min(span['bbox'][0] for span in line),  # x0
            min(span['bbox'][1] for span in line),  # y0
            max(span['bbox'][2] for span in line),  # x1
            max(span['bbox'][3] for span in line),  # y1
        ]
        line_objects.append({
            "bbox": line_bbox,
            "spans": line,
        })
    return line_objects

def fix_text_block(block):
    # Formula spans in text blocks should be converted to inline type
    for span in block['spans']:
        if span['type'] == "isolated":
            span['type'] = "inline"
    block_lines = merge_spans_to_line(block['spans'])
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    block['lines'] = sort_block_lines
    del block['spans']
    return block


def fix_interline_block(block):
    block_lines = merge_spans_to_line(block['spans'])
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    block['lines'] = sort_block_lines
    del block['spans']
    return block

def calculate_overlap_area_in_bbox1_area_ratio(bbox1, bbox2):
    """
    Calculate the ratio of overlap area between box1 and box2 to bbox1's area
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    if bbox1_area == 0:
        return 0
    else:
        return intersection_area / bbox1_area

def fill_spans_in_blocks(blocks, spans, radio):
    '''
    Place spans into blocks based on their positional relationships
    '''
    block_with_spans = []
    for block in blocks:
        block_type = block["category_type"]
        L = block['poly'][0]
        U = block['poly'][1]
        R = block['poly'][2]
        D = block['poly'][5]
        L, R = min(L, R), max(L, R)
        U, D = min(U, D), max(U, D)
        block_bbox = [L, U, R, D]
        block_dict = {
            'type': block_type,
            'bbox': block_bbox,
            'saved_info': block
        }
        block_spans = []
        for span in spans:
            span_bbox = span["bbox"]
            if calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > radio:
                block_spans.append(span)

        '''Inline formula adjustment: adjust height to match text height on the same line (prioritize left side, then right side)'''
        # displayed_list = []
        # text_inline_lines = []
        # modify_y_axis(block_spans, displayed_list, text_inline_lines)

        '''Convert incorrectly recognized displayed formulas to inline formulas'''
        # block_spans = modify_inline(block_spans, displayed_list, text_inline_lines)

        '''Remove bbox overlaps'''  # Removing overlaps affects span bbox, causing errors in subsequent fill operations
        # block_spans = remove_overlap_between_bbox_for_span(block_spans)

        block_dict['spans'] = block_spans
        block_with_spans.append(block_dict)

        # Remove spans that have been added to block_spans from the spans list
        if len(block_spans) > 0:
            for span in block_spans:
                spans.remove(span)

    return block_with_spans, spans

def fix_block_spans(block_with_spans):
    '''
    1. img_block and table_block have nested block relationships because they contain captions and footnotes.
       Need to place caption and footnote text_spans into the corresponding caption_block and footnote_block 
       within img_block and table_block
    2. Also need to remove the spans field from blocks
    '''
    fix_blocks = []
    for block in block_with_spans:
        block_type = block['type']

        # if block_type == BlockType.Image:
        #     block = fix_image_block(block, img_blocks)
        # elif block_type == BlockType.Table:
        #     block = fix_table_block(block, table_blocks)
        if block_type == "isolate_formula":
            block = fix_interline_block(block)
        else:
            block = fix_text_block(block)
        fix_blocks.append(block)
    return fix_blocks


# def detect_lang(text: str) -> str:

#     if len(text) == 0:
#         return ""
#     try:
#         lang_upper = detect_language(text)
#     except:
#         html_no_ctrl_chars = ''.join([l for l in text if unicodedata.category(l)[0] not in ['C', ]])
#         lang_upper = detect_language(html_no_ctrl_chars)
#     try:
#         lang = lang_upper.lower()
#     except:
#         lang = ""
#     return lang

def detect_lang(string):
    """
    Detect language from text based on Unicode character ranges.
    Supports: Chinese (zh), Vietnamese (vi), Japanese (ja), Korean (ko), 
    Thai (th), Arabic (ar), Hebrew (he), Cyrillic (ru), and defaults to English (en).
    
    :param string: Text string to detect language from
    :return: Language code string ('zh', 'vi', 'ja', 'ko', 'th', 'ar', 'he', 'ru', 'en')
    """
    if not string or not string.strip():
        return 'en'
    
    # Count characters in each language category
    char_counts = {
        'zh': 0,      # Chinese
        'vi': 0,      # Vietnamese (Latin + Vietnamese marks)
        'ja': 0,      # Japanese (Hiragana, Katakana)
        'ko': 0,      # Korean (Hangul)
        'th': 0,      # Thai
        'ar': 0,      # Arabic
        'he': 0,      # Hebrew
        'ru': 0,      # Cyrillic (Russian and related)
        'latin': 0    # Basic Latin (for English/French/German/etc.)
    }
    
    for ch in string:
        # Chinese: CJK Unified Ideographs
        if u'\u4e00' <= ch <= u'\u9fff':
            char_counts['zh'] += 1
        # Japanese: Hiragana and Katakana
        elif (u'\u3040' <= ch <= u'\u309f') or (u'\u30a0' <= ch <= u'\u30ff'):
            char_counts['ja'] += 1
        # Korean: Hangul
        elif (u'\uac00' <= ch <= u'\ud7af') or (u'\u1100' <= ch <= u'\u11ff'):
            char_counts['ko'] += 1
        # Thai
        elif u'\u0e00' <= ch <= u'\u0e7f':
            char_counts['th'] += 1
        # Arabic
        elif u'\u0600' <= ch <= u'\u06ff':
            char_counts['ar'] += 1
        # Hebrew
        elif u'\u0590' <= ch <= u'\u05ff':
            char_counts['he'] += 1
        # Cyrillic (Russian, Ukrainian, etc.)
        elif u'\u0400' <= ch <= u'\u04ff':
            char_counts['ru'] += 1
        # Vietnamese marks (combining diacritics)
        elif u'\u0300' <= ch <= u'\u036f':
            char_counts['vi'] += 1
        # Basic Latin
        elif u'\u0000' <= ch <= u'\u007f':
            char_counts['latin'] += 1
        # Latin Extended (includes Vietnamese base characters)
        elif u'\u0080' <= ch <= u'\u024f':
            char_counts['vi'] += 1
    
    # Determine language by highest count
    # Remove 'latin' from direct comparison since it's common
    max_lang = 'en'
    max_count = 0
    
    for lang, count in char_counts.items():
        if lang == 'latin':
            continue
        if count > max_count:
            max_count = count
            max_lang = lang
    
    # If no specific language detected but has Latin characters → English
    if max_count == 0 and char_counts['latin'] > 0:
        return 'en'
    
    # If Vietnamese detected (Vietnamese marks or extended Latin)
    if char_counts['vi'] > 0 and char_counts['latin'] > 0:
        return 'vi'
    
    return max_lang if max_count > 0 else 'en'

def ocr_escape_special_markdown_char(content):
    """
    Escape characters that have special meaning in markdown syntax
    """
    special_chars = ["*", "`", "~", "$"]
    for char in special_chars:
        content = content.replace(char, "\\" + char)

    return content

# def split_long_words(text):
#     segments = text.split(' ')
#     for i in range(len(segments)):
#         words = re.findall(r'\w+|[^\w]', segments[i], re.UNICODE)
#         for j in range(len(words)):
#             if len(words[j]) > 15:
#                 words[j] = ' '.join(wordninja.split(words[j]))
#         segments[i] = ''.join(words)
#     return ' '.join(segments)


def merge_para_with_text(para_block):
    para_text = ''
    for line in para_block['lines']:
        line_text = ""
        line_lang = ""
        for span in line['spans']:
            span_type = span['type']
            if span_type == "text":
                line_text += span['content'].strip()
        if line_text != "":
            line_lang = detect_lang(line_text)
        for span in line['spans']:
            span_type = span['type']
            content = ''
            if span_type == "text":
                content = span['content']
                content = ocr_escape_special_markdown_char(content)
                # language = detect_lang(content)
                # if language == 'en':  # Only perform word segmentation for English long words, Chinese segmentation loses text
                    # content = ocr_escape_special_markdown_char(split_long_words(content))
                # else:
                #     content = ocr_escape_special_markdown_char(content)
            elif span_type == 'inline':
                content = f" ${span['content'].strip('$')}$ "
            elif span_type == 'ignore-formula':
                content = f" ${span['content'].strip('$')}$ "
            elif span_type == 'isolated':
                content = f"\n$$\n{span['content'].strip('$')}\n$$\n"    
            elif span_type == 'footnote':
                content_ori = span['content'].strip('$')
                if '^' in content_ori:
                    content = f" ${content_ori}$ "
                else:
                    content = f" $^{content_ori}$ "

            if content != '':
                if 'zh' in line_lang:  # For documents with one character per span, single character language detection is inaccurate; use full line text for detection
                    para_text += content.strip()  # In Chinese context, no space separation needed between contents
                else:
                    para_text += content.strip() + ' '  # In English context, space separation needed between contents
    return para_text