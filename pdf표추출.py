# -*- coding: utf-8 -*-
"""bge-m3_faiss_v5_250305.py

PDF 문서를 파싱하고 벡터 저장소를 구축하는 파이프라인
"""

import os
import re
import json
import pickle
from pathlib import Path
from collections import defaultdict

import pdfplumber
from tqdm import tqdm
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import pandas as pd

# OpenAI 클라이언트 초기화 (API 키 설정 필요)
from openai import OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("OpenAI API 키를 입력하세요: ")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)  # 실제 API 키로 대체

# 입력 경로 및 파일 설정
input_path = './'  # 현재 디렉토리
input_file_name = '[한화손해보험]사업보고서(2025.03.11).pdf'  # 실제 PDF 파일명으로 변경
src = Path(input_path) / input_file_name

# PDF 파서 함수
def pdf_parser(pdf_path):
    """PDF 파일에서 페이지별로 텍스트와 테이블을 추출"""
    elements = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            element = {
                'Chapter': f'Chapter {i}',
                'Section': f'Section {i}',
                'Subsection': f'Subsection {i}',
                'Content': text,
                'Page_Numbers': [i + 1],
                'id': i,
            }
            if tables:
                element['Table'] = tables
            elements.append(element)
    return elements

# 테이블 ID 생성 함수
def table_id_generation(element):
    """테이블에 고유 ID 부여"""
    if "Table" not in element:
    return {}
    values = element['Table']
    keys = [f"element{element['id']}-table{i}" for i in range(len(values))]
    return dict(zip(keys, values))

# 테이블 메타 정보 추출 함수
def extract_table_info(table):
    """테이블의 메타데이터 추출"""
  table_dict = {
        'page': table.page.number,
      'bbox': table.bbox,
        'ncol': len(table.rows[0]) if table.rows else 0,
      'nrow': len(table.rows),
      'content': table.extract(),
      }
  return table_dict

# 테이블 비교 함수
def compare_tables(table_A, table_B):
    """두 테이블이 연속적인지 비교"""
    prev_info = extract_table_info(table_A)
    curr_info = extract_table_info(table_B)
  counter = 0
    if curr_info['page'] - prev_info['page'] == 1:
    counter += 1
    if (np.round(prev_info['bbox'][3], 0) > 780) and (np.round(curr_info['bbox'][1], 0) < 50):
    counter += 1
    if prev_info['ncol'] == curr_info['ncol']:
    counter += 1
    decision = 'same table' if counter >= 2 else 'different table'
    return [(counter, decision)]

# 테이블 그룹화 함수
def group_table_position(element_table):
    """연속된 테이블을 그룹화"""
    if len(element_table) <= 1:
        return [[0, 1]]
  pos = 0
  counter = 0
  result = []
    for i in range(1, len(element_table)):
    counter += 1
    table_comparison_result = compare_tables(element_table[i-1], element_table[i])[0][1]
        if table_comparison_result != 'same table':
            result.append([pos, pos + counter])
      pos += counter
            counter = 0
    result.append([pos, pos + counter + 1])
  return result

# 딕셔너리 병합 함수
def merge_dicts(dict_list):
    """여러 테이블의 메타데이터를 병합"""
    merged_dict = {
        'page': [],
        'bbox': [],
        'ncol': 0,
        'nrow': 0,
        'content': [],
    }
    for d in dict_list:
        merged_dict['page'].append(d['page'])
        merged_dict['bbox'].append(d['bbox'])
        merged_dict['ncol'] = max(merged_dict['ncol'], d['ncol'])
        merged_dict['nrow'] += d['nrow']
        merged_dict['content'].extend(d['content'])
    return merged_dict

# 테이블 위치 찾기 함수
def find_table_location_in_text(element_content):
    """콘텐츠 내 테이블 태그 위치 찾기"""
    start_pattern = r'<table>'
    table_start_position = re.finditer(start_pattern, element_content)
    start_positions = [(match.start(), match.end()) for match in table_start_position]
    end_pattern = r'</table>'
    table_end_position = re.finditer(end_pattern, element_content)
    end_positions = [(match.start(), match.end()) for match in table_end_position]
    if len(start_positions) != len(end_positions):
        return []
    table_location_in_text = [(start[0], end[1]) for start, end in zip(start_positions, end_positions)]
    return table_location_in_text

# 테이블 처리 함수
def main_table_processing(element):
    """테이블 그룹화 및 메타데이터 처리"""
    element_content = element['Content']
    element_table = element['Table']
  table_group = group_table_position(element_table)
  table_location_in_text = find_table_location_in_text(element_content)
  table_id_list = list(table_id_generation(element).keys())

    element_table_meta_list = []
    element_table_content_list = []
    element_table_id_list = []
    
    for i, (start, end) in enumerate(table_group):
        tbl_list = element_table[start:end]
        meta_dicts = [extract_table_info(tbl) for tbl in tbl_list]
        merged_meta_dict = merge_dicts(meta_dicts)
        element_table_meta_list.append(merged_meta_dict)
        
        if i < len(table_location_in_text):
    positions = table_location_in_text[start:end]
            if positions:
                min_val, max_val = min(pos[0] for pos in positions), max(pos[1] for pos in positions)
    original_content = element_content[min_val:max_val]
                modified_content = original_content.replace('</table>\n<table>', ', ')
                element_table_content_list.append({
                    "raw": original_content,
                    "modified": modified_content,
                    "location": (min_val, max_val)
                })
                element_table_id_list.append(table_id_list[start])
    
    return (table_group, table_location_in_text, element_table_meta_list, 
            element_table_content_list, element_table_id_list)

# Document 생성 함수
def generate_document(element):
    """테이블이 없는 element를 Document로 변환"""
  metadata = {
      'chapter': element['Chapter'],
      'section': element['Section'],
      'subsection': element['Subsection'],
      'page': element['Page_Numbers'],
  }
    if 'Table' in element:
        metadata['table'] = table_id_generation(element)
  document = Document(
        page_content=element['Content'],
        metadata=metadata,
        id='element' + str(element['id'])
    )
  return document

def generate_document_with_table(element, document_list_of_table_element):
    """테이블이 있는 element를 Document로 변환"""
  metadata = {
      'chapter': element['Chapter'],
      'section': element['Section'],
      'subsection': element['Subsection'],
      'page': element['Page_Numbers'],
  }
    if 'Table' in element:
        metadata['table'] = [tbl_doc.id for tbl_doc in document_list_of_table_element]
  element_modified = element.copy()
    for doc in document_list_of_table_element:
    page_content = json.loads(doc.page_content)
        element_modified['Content'] = element_modified['Content'].replace(
            page_content['raw'], page_content['modified'])
  document = Document(
        page_content=element_modified['Content'],
        metadata=metadata,
        id='element' + str(element['id'])
    )
  return document

# LLM을 사용한 테이블 요약 함수
def ask_table_summary_markdown(table_text, document_content):
    """LLM을 사용해 테이블 요약 및 마크다운 변환"""
    prompt = f"""
    다음 table_content와 원본 문서의 컨텍스트를 기반으로,
    **table_content에 대한 포괄적인 설명**을 한국어로 제공하며 주요 통찰과 트렌드를 포함하세요.
    그런 다음, **제공된 table_content만 마크다운 형식으로 재구성**하세요.

    ## 원본 문서의 컨텍스트 (document_content):
    {document_content}

    ## 테이블 내용 (table_content):
    ※ 주의: table_content는 부분적일 수 있습니다.
    {table_text}

    ### 지침:
    1. **테이블에 대한 포괄적인 설명**을 한국어로 제공하며 주요 통찰과 트렌드를 포함.
    2. 테이블의 **소제목, 제목, 캡션**을 명확히 식별:
       - 캡션은 테이블 하단에 위치하며, 존재하지 않으면 임의로 추가하지 않음.
       - 제목을 단위 표시(예: (단위:~))와 혼동하지 않음.
    3. **table_content를 완전히 재구성**:
       - 테이블이 여러 페이지에 걸쳐 분할된 경우, document_content를 사용해 전체 테이블을 재구성.
       - 테이블은 <table> 태그로 시작하고 </table> 태그로 끝남.
       - document_content에서 동일한 테이블의 명확한 연속성이 확인되지 않으면 그대로 유지.
    4. **다중 인덱스 열 처리**:
       - 다중 레벨 열이 있는 경우, 모든 레벨을 '&' 기호로 연결.
    5. 최종 테이블은 **유효한 마크다운 형식**이어야 하며 가독성을 보장:
       - HTML 형식으로 변환하지 않음. 유일한 유효 형식은 마크다운.

    ### 예상 출력:
    최종 출력은 **깔끔한 JSON 형식**이며, 키-값 쌍은 아래와 같이 구성:

    final_output = {
      "section_header": ["소제목"],
      "title": ["테이블 제목"],
      "captions": ["캡션"],
      "description": "테이블에 대한 포괄적인 설명 문자열",
      "table": "유효한 마크다운 형식의 테이블"
    }
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that describes tables and formats them in markdown."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# LLM 응답 파싱 함수
def parse_llm_json_response(response_text):
    """LLM 응답에서 JSON 파싱"""
    cleaned_text = re.sub(r"^```json\n?|```$", "", response_text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned_text)
  except json.JSONDecodeError as e:
      print(f"JSON 변환 실패: {e}")
        return None

# 임베딩 생성 함수
def embed_with_progress(texts, embedding_model, batch_size):
    """텍스트 리스트를 배치 단위로 임베딩 생성"""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i:i + batch_size]
        embeddings = embedding_model.embed_documents(batch_texts)
        all_embeddings.extend(embeddings)
    return all_embeddings

# FAISS 벡터 저장소 설정 함수
def setup_faiss_vectorstore(embeddings, doc_ids, doc_list):
    """FAISS 벡터 저장소 구성"""
n, dimension_size = embeddings.shape
faiss_index = faiss.IndexFlatL2(dimension_size)
faiss_index.add(embeddings)
index_to_docstore_id = {i: doc_ids[i] for i in range(len(doc_ids))}
    docstore_data = {doc.id: doc for doc in doc_list}
    docstore = InMemoryDocstore(docstore_data)
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
vectorstore = FAISS(
        embedding_function=embedding_model,
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore

# 메인 함수
def main():
    """메인 실행 함수"""
    print("PDF 파싱 및 FAISS 벡터 저장소 구축 시작")
    
    # 1. PDF 파싱
    elements = pdf_parser(src)
    
    # 2. Document 생성 (테이블 처리 포함)
    doc_list = []
    for element in elements:
        if 'Table' in element:
            table_group, _, element_table_meta_list, element_table_content_list, element_table_id_list = main_table_processing(element)
            table_docs = [
                Document(
                    page_content=json.dumps(content),
                    metadata={"table_id": table_id},
                    id=table_id
                )
                for content, table_id in zip(element_table_content_list, element_table_id_list)
            ]
            doc = generate_document_with_table(element, table_docs)
        else:
            doc = generate_document(element)
        doc_list.append(doc)
    
    # 3. LLM을 사용한 테이블 요약 (선택적)
    for doc in doc_list:
        if 'table' in doc.metadata:
            table_ids = doc.metadata['table']
            for table_doc in doc_list:
                if table_doc.id in table_ids:
                    table_text = json.loads(table_doc.page_content)['raw']
                    document_content = doc.page_content
                    llm_response = ask_table_summary_markdown(table_text, document_content)
                    parsed_response = parse_llm_json_response(llm_response)
                    if parsed_response:
                        table_doc.page_content = json.dumps(parsed_response)
    
    # 4. 임베딩 생성
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    texts = [doc.page_content for doc in doc_list]
    embeddings = embed_with_progress(texts, embedding_model, batch_size=32)
    embeddings = np.array(embeddings)
    
    # 5. FAISS 벡터 저장소 구축
    doc_ids = [doc.id for doc in doc_list]
    vectorstore = setup_faiss_vectorstore(embeddings, doc_ids, doc_list)
    
    # 6. 저장소 저장
    vectorstore.save_local("./faiss_index")
    
    print("처리 완료")

if __name__ == "__main__":
    main()