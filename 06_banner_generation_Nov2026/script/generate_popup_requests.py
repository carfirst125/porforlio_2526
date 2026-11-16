# -*- coding: utf-8 -*-
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
import openpyxl

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_EXCEL_PATH = Path(r'C:\Users\huynq25\Downloads\AI Squad - Prompt AI (1).xlsx')
INPUT_DIR = PROJECT_DIR / 'input'

SEGMENTS = {
    'AFF': {
        'col_idx': 2,
        'format_type': 'ocb_omni_popup_aff_720x1280',
        'dominant_color': '#0F472F',
    },
    'MASS_AFF': {
        'col_idx': 3,
        'format_type': 'ocb_omni_popup_mass_aff_720x1280',
        'dominant_color': '#3DE6A0',
    },
    'MASS': {
        'col_idx': 4,
        'format_type': 'ocb_omni_popup_mass_720x1280',
        'dominant_color': '#B3F3BB',
    },
}

# Bucket 3: Chi duy nhat 2 san pham the co phoi the cung cap san
CARD_ASSET_MAP = {
    'Thẻ Passport Debit': ['Bucket_3/passport_card.png'],
    'Thẻ World 2in1 Credit': ['Bucket_3/prod_card_MCW.png'],
}

# Bucket 4: 3 san pham co logo chu typography thuong hieu duyet san
BRAND_TEXT_MAP = {
    'Flexi': {
        'AFF': ['Bucket_4/FLEXIText01.png'],
        'MASS_AFF': ['Bucket_4/FLEXIText02.png'],
        'MASS': ['Bucket_4/FLEXIText02.png'],
    },
    'Max saving': {
        'AFF': ['Bucket_4/MaxText03.png'],
        'MASS_AFF': ['Bucket_4/MaxText01.png'],
        'MASS': ['Bucket_4/MaxText02.png'],
    },
    'Flex Fund': {
        'AFF': ['Bucket_4/FlexLoan.png'],
        'MASS_AFF': ['Bucket_4/FlexLoan.png'],
        'MASS': ['Bucket_4/FlexLoan.png'],
    },
}

def sanitize_slug(name):
    s = name.strip()
    s = re.sub(r'[^a-zA-Z0-9_ ]', '', s)
    s = re.sub(r'\s+', '_', s)
    nfkd = unicodedata.normalize('NFKD', s)
    ascii_s = ''.join([c for c in nfkd if not unicodedata.combining(c)])
    return ascii_s.upper()

def main():
    parser = argparse.ArgumentParser(description='Generate Popup request JSONs from Excel.')
    parser.add_argument('--excel', type=Path, default=DEFAULT_EXCEL_PATH)
    parser.add_argument('--output', type=Path, default=INPUT_DIR)
    parser.add_argument('--num-version', type=int, default=2)
    args = parser.parse_args()

    if not args.excel.exists():
        print(f'[ERROR] Excel not found: {args.excel}', file=sys.stderr)
        sys.exit(1)

    wb = openpyxl.load_workbook(args.excel, data_only=True)
    if 'Request popup' not in wb.sheetnames:
        print('[ERROR] Sheet Request popup not found in Excel', file=sys.stderr)
        sys.exit(1)

    ws = wb['Request popup']
    args.output.mkdir(parents=True, exist_ok=True)

    count = 0
    for r in range(11, ws.max_row + 1):
        prod_name_raw = ws.cell(r, 1).value
        if not prod_name_raw:
            continue
        prod_name = str(prod_name_raw).strip()
        slug = sanitize_slug(prod_name)

        # 1. Bucket 3 (Product): The card thi lay asset phoi the, con lai la text description
        product_asset = None
        for k, v in CARD_ASSET_MAP.items():
            if k.lower() in prod_name.lower():
                product_asset = v
                break
        if product_asset is None:
            product_asset = [f'Điện thoại hoặc biểu tượng thể hiện tính năng {prod_name} trên ứng dụng OCB OMNI.']

        for seg_name, seg_info in SEGMENTS.items():
            brief_val = ws.cell(r, seg_info['col_idx']).value
            if not brief_val:
                brief_val = ws.cell(r, 2).value or prod_name

            brief_text = f'{prod_name}\n{str(brief_val).strip()}'

            # 2. Bucket 4 (Brand text)
            brand_text_asset = []
            brand_text_note = ''
            for k, seg_dict in BRAND_TEXT_MAP.items():
                if k.lower() in prod_name.lower():
                    brand_text_asset = seg_dict.get(seg_name, [])
                    brand_text_note = f'\nTên sản phẩm đã được cung cấp dưới dạng brand-text asset hoàn chỉnh. Dùng nguyên asset này trong vùng chữ; text AI tự sinh không lặp lại tên {prod_name} hoặc tagline đã có trong asset.'
                    break

            if brand_text_note:
                brief_text += brand_text_note

            cid = f'DB_POPUP_{slug}_720x1280_{seg_name}'

            theme_instruction = f'AI chọn bối cảnh phù hợp sản phẩm {prod_name} và phân khúc {seg_name}, theo design_brief.'
            if brand_text_asset:
                theme_instruction += f' và sinh text thì không được lặp lại từ tên sản phẩm {prod_name} đã có trong brand_text asset.'

            req_obj = {
                'creative_id': cid,
                'format_type': seg_info['format_type'],
                'design_brief': brief_text,
                'buckets': {
                    'theme': [theme_instruction],
                    'context': [],
                    'product': product_asset,
                    'brand_text': brand_text_asset
                },
                'reference': '',
                'text': {
                    'headline': '',
                    'message': '',
                    'cta_label': '',
                    'cta_deep_link': ''
                },
                'guidelines': '',
                'num_version': args.num_version,
                'design_options': {
                    'generate_person': False,
                    'has_logo': False,
                    'text_by_code': False,
                    'auto_generate_text': False,
                    'generate_text': True,
                    'generate_text_headline': True,
                    'generate_text_message': True,
                    'generate_text_cta_label': True,
                    'dominant_color': seg_info['dominant_color'],
                    'technique_selection': {
                        'mode': 'auto',
                        'manual_techniques': [],
                        'observe_brand': ['general', 'ocb_db'],
                        'auto_config': {
                            'max_background_techniques': 3,
                            'max_product_techniques': 3,
                            'max_decoration_techniques': 3,
                            'selection_constraints': ['prompts/ocb_omni_110926_techniques_sel.txt']
                        }
                    },
                    'dominant_color_tolerance_percent': 15,
                    'num_decorative_elements': 2
                },
                'generation': {
                    'canvas': {'width': 720, 'height': 1280},
                    'safe_area': {'width': 720, 'height': 1280, 'position': 'center', 'x': 0, 'y': 0},
                    'instruction': ''
                },
                'output': {
                    'size': {'width': 720, 'height': 1280},
                    'clone': []
                }
            }

            out_file = args.output / f'{cid}.json'
            out_file.write_text(json.dumps(req_obj, ensure_ascii=False, indent=2), encoding='utf-8')
            count += 1

    print(f'[SUCCESS] Created {count} Popup request JSON files (fully mapped Bucket 3 & Bucket 4).')

if __name__ == '__main__':
    main()
