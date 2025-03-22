import os
import re
import time
import json
import hashlib
import logging
import traceback
import requests
import torch
import argparse
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
import asyncio
import time
import io
import base64
import aiohttp
from io import StringIO
from datetime import datetime
import glob
import sys
import tabula
import random

# 디버그 모드 설정
DEBUG_MODE = False

# OLLAMA_AVAILABLE 변수 정의
OLLAMA_AVAILABLE = False

# ollama 모듈 가져오기 시도
try:
    import ollama
    OLLAMA_AVAILABLE = True
    
    # ollama 라이브러리 사용 예제 (참고용)
    """
    # 모델 목록 조회
    models = ollama.list()
    
    # 채팅 응답 생성
    response = ollama.chat(model='gemma3:4b', messages=[
        {'role': 'user', 'content': '질문 내용'}
    ])
    print(response['message']['content'])
    
    # 스트리밍 응답 생성
    for chunk in ollama.chat(
        model='gemma3:4b',
        messages=[{'role': 'user', 'content': '질문 내용'}],
        stream=True,
    ):
        print(chunk['message']['content'], end='', flush=True)
    """
except ImportError:
    print("⚠️ ollama 모듈 가져오기 실패")
# 현재 파일 위치 기준 상대 경로 계산을 위한 변수 추가
# 현재 스크립트 파일의 디렉토리 경로 설정
# 현재 스크립트 파일의 절대 경로 가져오기
SCRIPT_DIR = os.getcwd()
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # 상위 디렉토리
print(f"현재 스크립트 파일 경로: {SCRIPT_DIR}")
print(f"상위 디렉토리 경로: {BASE_DIR}")


# .env 파일에서 환경변수 로드
load_dotenv()

# OpenMP 스레드 수 제한 (FAISS와 Java 충돌 방지)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# PDF 처리 라이브러리
import pypdf
import tabula
import pdfplumber

# RAG 관련 라이브러리
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import faiss

# RAG 평가 관련 메트릭 라이브러리
try:
    from rank_bm25 import BM25Okapi
    EVAL_LIBS_AVAILABLE = True
except ImportError:
    EVAL_LIBS_AVAILABLE = False

# 환경 설정
PDF_PATH = os.path.join(SCRIPT_DIR, "[한화손해보험]사업보고서(2025.03.11).pdf")
INDEX_DIR = os.path.join(SCRIPT_DIR, "Index/faiss_index_bge")  # 인덱스 디렉토리
METADATA_FILE = os.path.join(SCRIPT_DIR, "Index/document_metadata_bge.json")  # 메타데이터 파일
LOG_FILE = os.path.join(SCRIPT_DIR, "Log/hwgi_rag_streamlit.log")
CACHE_FILE = os.path.join(SCRIPT_DIR, "Log/query_cache_streamlit.json")
EVALUATION_FILE = os.path.join(SCRIPT_DIR, "Log/model_evaluations.json")  # 모델 평가 결과 저장 파일

# Ollama API 기본 URL 설정
OLLAMA_API_BASE = "http://localhost:11434/api"

# 사용 가능한 모델 설정
AVAILABLE_MODELS = ["gemma3:4b", "llama3.1:8b", "gemma3:12b"]

# 모델 설정
EMBEDDING_MODELS = {
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "index_dir": INDEX_DIR,
        "metadata_file": METADATA_FILE
    }
}

# 로깅 설정
def setup_logging(log_level=logging.DEBUG):
    logger = logging.getLogger('hwgi_rag')
    logger.setLevel(log_level)
    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setLevel(log_level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

logger = setup_logging()

# E5Embeddings 클래스를 BGE-M3 임베딩으로 대체
class BGEM3Embeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        print(f"✓ BGE-M3 임베딩 모델 '{model_name}' 초기화 중...")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        # 디바이스 설정
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✓ Apple Silicon GPU (MPS) 사용 가능")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("✓ NVIDIA GPU (CUDA) 사용 가능")
        else:
            self.device = torch.device("cpu")
            print("✓ CPU 사용")
        
        self.model.to(self.device)
        print(f"✓ 모델 로드 완료 (디바이스: {self.device})")
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리 함수"""
        # 특수문자 제거 및 공백 정리
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트의 임베딩을 반환합니다."""
        try:
            # 배치 크기 증가 (32 → 64)
            batch_size = 64
            all_embeddings = []
            
            # 전처리된 텍스트로 배치 처리
            for i in range(0, len(texts), batch_size):
                batch = [self._preprocess_text(text) for text in texts[i:i + batch_size]]
                with torch.inference_mode():
                    embeddings = self.model.encode(
                        batch,
                        convert_to_tensor=True,
                        device=self.device,
                        normalize_embeddings=True  # L2 정규화 적용
                    )
                    if self.device.type == "mps":
                        embeddings = embeddings.to("cpu")
                    all_embeddings.extend(embeddings.cpu().numpy().tolist())
            
            return all_embeddings
        except Exception as e:
            print(f"❌ 문서 임베딩 생성 중 오류: {e}")
            raise e
    
    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 텍스트의 임베딩을 반환합니다."""
        try:
            # 쿼리용 접두사 추가
            query_text = f"query: {text}"
            with torch.inference_mode():
                embedding = self.model.encode(
                    [query_text],
                    convert_to_tensor=True,
                    device=self.device,
                    normalize_embeddings=True
                )
                # MPS 디바이스에서 CPU로 이동 후 numpy 변환
                if self.device.type == "mps":
                    embedding = embedding.to("cpu")
                return embedding.cpu().numpy().tolist()[0]
        except Exception as e:
            print(f"❌ 쿼리 임베딩 생성 중 오류: {e}")
            raise e

# OpenAI 임베딩 클래스 정의
class OpenAIEmbeddings(Embeddings):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        print(f"✓ OpenAI 임베딩 모델 '{model_name}' 초기화 중...")
        load_dotenv()  # .env 파일에서 OPENAI_API_KEY 로드
        self.model_name = model_name
        self.client = OpenAI()
        print("✓ OpenAI 클라이언트 초기화 완료")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트의 임베딩을 반환합니다."""
        try:
            # 배치 크기 설정 (OpenAI API 제한 고려)
            batch_size = 100
            all_embeddings = []
            
            # 배치 단위로 처리
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float"
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            print(f"❌ OpenAI 문서 임베딩 생성 중 오류: {e}")
            raise e
    
    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 텍스트의 임베딩을 반환합니다."""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[text],
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ OpenAI 쿼리 임베딩 생성 중 오류: {e}")
            raise e

# --- PDF 처리 및 문서 분할 ---
class PDFProcessor:
    def __init__(self, pdf_path: str):
        # 경로가 상대 경로인 경우 현재 스크립트 위치 기준으로 절대 경로 변환
        if not os.path.isabs(pdf_path):
            self.pdf_path = os.path.join(SCRIPT_DIR, pdf_path)
        else:
            self.pdf_path = pdf_path
        self.text_content = []  # 텍스트 내용 저장
        self.tables = []  # 표 데이터 저장
        self.page_count = 0  # 총 페이지 수
        self.pdf_hash = self._calculate_pdf_hash()  # PDF 파일 해시
        self.hash_file = os.path.join(SCRIPT_DIR, "pdf_hash.json")  # 해시 저장 파일
        logger.info(f"PDFProcessor 초기화: '{self.pdf_path}' 파일 처리 준비")
    
    def _calculate_pdf_hash(self) -> str:
        """PDF 파일의 해시값을 계산합니다."""
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_hash = hashlib.md5(file.read()).hexdigest()
            return pdf_hash
        except Exception as e:
            logger.error(f"PDF 해시 계산 중 오류: {e}")
            return ""
    
    def _load_previous_hash(self) -> str:
        """이전에 처리한 PDF의 해시값을 로드합니다."""
        try:
            if os.path.exists(self.hash_file):
                with open(self.hash_file, 'r') as f:
                    data = json.load(f)
                    return data.get('pdf_hash', '')
            return ''
        except Exception as e:
            logger.error(f"이전 해시 로드 중 오류: {e}")
            return ''
    
    def _save_current_hash(self):
        """현재 PDF 해시값을 JSON 파일에 저장합니다."""
        try:
            os.makedirs(os.path.dirname(self.hash_file), exist_ok=True)
            data = {
                'pdf_hash': self.pdf_hash,
                'pdf_path': self.pdf_path,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(self.hash_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✓ 현재 PDF 해시 저장 완료: {self.hash_file}")
        except Exception as e:
            print(f"⚠️ 현재 해시 저장 중 오류: {e}")
            logger.error(f"현재 해시 저장 중 오류: {e}")
    
    def needs_processing(self) -> bool:
        """PDF가 새로운 데이터인지 확인합니다."""
        previous_hash = self._load_previous_hash()
        return previous_hash != self.pdf_hash
    
    def extract_text(self) -> List[Document]:
        logger.info("📄 PDF 텍스트 내용 추출 시작")
        print("📄 PDF 텍스트 내용 추출 중...")
        documents = []
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                self.page_count = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        doc_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                        
                        # 텍스트 내용 저장
                        self.text_content.append({
                            "page": page_num + 1,
                            "content": text,
                            "hash": doc_hash
                        })
                        
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={"page": page_num + 1, "source": "text", "hash": doc_hash}
                            )
                        )
            logger.info(f"✅ 총 {self.page_count}페이지에서 {len(documents)}개의 텍스트 문서 추출 완료")
            return documents
        except Exception as e:
            logger.error(f"❌ 텍스트 추출 중 오류: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def table_id_generation(self, element):
        """테이블 ID 생성 함수"""
        if "Table" not in element:
            return {}
        else:
            values = element['Table']  # list of tables
            keys = [f"element{element['id']}-table{i}" for i in range(len(values))]
            return dict(zip(keys, values))

    def extract_cell_color(self, table):
        """테이블의 첫 셀과 마지막 셀 색상 추출 함수"""
        page_image = table.page.to_image()
        cell_image_first = page_image.original.crop(table.cells[0])
        cell_image_last = page_image.original.crop(table.cells[-1])
        
        res_list = []
        for cell_image in [cell_image_first, cell_image_last]:
            width, height = cell_image.size
            background_pixel = cell_image.getpixel((width/5, height/5))
            res_list.append(background_pixel)
        
        return res_list

    def extract_image(self, table, resolution=300):
        """테이블 이미지 추출 함수"""
        scale_factor = resolution / 72
        x0, y0, x1, y1 = table.bbox
        
        x0 *= scale_factor
        y0 *= scale_factor
        x1 *= scale_factor
        y1 *= scale_factor
        
        img = table.page.to_image(resolution=resolution)
        table_img = img.original.crop((x0, y0, x1, y1))
        
        return table_img

    def extract_table_info(self, table):
        """테이블 메타 정보 추출 함수"""
        table_dict = {
            'page': self.extract_page_number(str(table.page)),
            'bbox': table.bbox,
            'ncol': len(table.columns),
            'nrow': len(table.rows),
            'content': table.extract(),
            'cell_color': self.extract_cell_color(table),
            'img': self.extract_image(table)
        }
        
        return table_dict

    def compare_tables(self, table_A, table_B):
        """두 테이블이 같은 테이블인지 비교"""
        prev_info = self.extract_table_info(table_A)
        curr_info = self.extract_table_info(table_B)
        
        counter = 0
        # 두 테이블의 페이지가 인접해 있는가?
        if curr_info['page'] - prev_info['page'] == 1:
            counter += 1
        
        # 테이블 위치가 이어지는가?
        if (np.round(prev_info['bbox'][3], 0) > 780) and (np.round(curr_info['bbox'][1], 0) == 50):
            counter += 1
        
        # 셀 색상이 같은가?
        if prev_info['cell_color'][1] == curr_info['cell_color'][0]:
            counter += 1
        
        # 컬럼 수가 같은가?
        if prev_info['ncol'] == curr_info['ncol']:
            counter += 1
        
        decision = 'same table' if counter == 4 else 'different table'
        return [(counter, decision)]

    def find_table_location_in_text(self, element_content):
        """콘텐츠 내 테이블 위치 찾기"""
        start_pattern = '<table>'
        table_start_position = re.finditer(start_pattern, element_content)
        start_positions = [(match.start(), match.end()) for match in table_start_position]
        
        end_pattern = '</table>'
        table_end_position = re.finditer(end_pattern, element_content)
        end_positions = [(match.start(), match.end()) for match in table_end_position]
        
        table_location_in_text = [(start[0], end[1]) 
                                for start, end in zip(start_positions, end_positions)]
        
        return table_location_in_text

    def group_table_position(self, element_table):
        """연속된 테이블의 포지션을 묶어주는 함수"""
        pos = 0
        counter = 0
        result = []
        
        for i in range(1, len(element_table)):
            counter += 1
            table_comparison_result = self.compare_tables(element_table[i-1], element_table[i])[0][1]
            
            if table_comparison_result != 'same table':
                result.append([pos, pos+counter])
                pos += counter
                counter = 0
        
        # 마지막 그룹 추가
        result.append([pos, pos + counter + 1])
        return result

    def merge_dicts(self, dict_list):
        """여러 개의 딕셔너리를 하나로 합치는 함수"""
        merged_dict = {
            'page': [],
            'bbox': [],
            'ncol': 0,
            'nrow': 0,
            'content': [],
            'cell_color': [],
            'img': [],
            'obj_type': ['table']
        }
        
        for d in dict_list:
            merged_dict['page'].append(d['page'])
            merged_dict['bbox'].append(d['bbox'])
            merged_dict['ncol'] = max(merged_dict['ncol'], d['ncol'])
            merged_dict['nrow'] += d['nrow']
            merged_dict['content'] += (d['content'])
            merged_dict['cell_color'].append(d['cell_color'])
            merged_dict['img'].append(d['img'])
        
        return merged_dict

    def extract_page_number(self, text):
        """테이블이 위치한 페이지 번호 추출 함수"""
        match = re.search(r"<Page:(\d+)>", text)
        return int(match.group(1)) if match else None

    def extract_tables(self) -> List[Document]:
        """PDF에서 표 데이터를 추출하고 Document로 변환"""
        logger.info("📊 PDF 표 데이터 추출 시작")
        print("📊 PDF 표 데이터 추출 중...")
        try:
            table_documents = []
            
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    if not tables:
                        continue
                    
                    for table_idx, table in enumerate(tables):
                        # 빈 행/열 제거 및 문자열 변환
                        table_content = []
                        for row in table:
                            if any(cell for cell in row):  # 빈 행 제외
                                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                                table_content.append(cleaned_row)
                        
                        if not table_content:
                            continue
                        
                        # CSV 형식으로 변환
                        table_text = '\n'.join([','.join(row) for row in table_content])
                        table_hash = hashlib.md5(table_text.encode('utf-8')).hexdigest()
                        
                        # 메타데이터 구성
                        metadata = {
                            'table_id': f'table_p{page_num}_t{table_idx + 1}',
                            'page': page_num,
                            'source': 'table',
                            'hash': table_hash,
                            'row_count': len(table_content),
                            'col_count': len(table_content[0]) if table_content else 0
                        }
                        
                        # 표 정보 저장
                        self.tables.append({
                            'table_id': metadata['table_id'],
                            'content': table_text,
                            'raw_data': table_content,
                            'hash': table_hash,
                            'metadata': metadata
                        })
                        
                        # Document 객체 생성
                        table_documents.append(
                            Document(
                                page_content=f"표 {metadata['table_id']}:\n{table_text}",
                                metadata=metadata
                            )
                        )
            
            logger.info(f"✅ {len(table_documents)}개의 표 처리 완료")
            return table_documents
            
        except Exception as e:
            logger.error(f"❌ 표 추출 중 오류: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def process(self) -> List[Document]:
        """PDF를 처리하고 문서 리스트를 반환합니다."""
        print(f"\n{'─'*60}")
        print("📌 1단계: PDF 문서 처리")
        print(f"{'─'*60}")
        
        if not self.needs_processing():
            logger.info("이전에 처리된 동일한 PDF 파일 감지. 변경 없음으로 판단.")
            print("✓ 이미 처리된 PDF 파일입니다. 새로운 처리가 필요 없습니다.")
            # 빈 리스트 반환하여 다음 단계에서 기존 인덱스 사용하도록 함
            return []
        
        logger.info("===== PDF 처리 시작 =====")
        text_docs = self.extract_text()
        table_docs = self.extract_tables()
        all_docs = text_docs + table_docs
        
        if not all_docs:
            print("⚠️ PDF에서 문서를 추출하지 못했습니다.")
            return []
        
        # 성공적으로 처리되면 현재 해시 저장
        self._save_current_hash()
        logger.info(f"📚 {len(all_docs)}개의 문서 조각 생성됨")
        print(f"📚 PDF 처리 완료: {len(text_docs)}개 텍스트 문서, {len(table_docs)}개 표 문서 생성")
        return all_docs
    
    def visualize_table(self, table_id: int):
        """특정 표를 시각화합니다 (matplotlib 사용)"""
        if not self.tables:
            print("⚠️ 표 데이터가 없습니다.")
            return
        
        # 유효한 table_id 확인
        table_index = table_id - 1
        if table_index < 0 or table_index >= len(self.tables):
            print(f"⚠️ 표 #{table_id}가 존재하지 않습니다.")
            return
        
        try:
            table_data = self.tables[table_index]
            df = pd.DataFrame(table_data["raw_data"])
            
            # 표 시각화
            fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
            ax.axis('off')
            ax.table(
                cellText=df.values,
                colLabels=df.columns,
                cellLoc='center',
                loc='center'
            )
            plt.title(f"표 {table_data['table_id']}", fontsize=14)
            plt.tight_layout()
            plt.show()
            
            # 테이블 정보 출력
            print(f"\n📊 표 {table_data['table_id']} 정보:")
            print(f"  - 행 수: {df.shape[0]}")
            print(f"  - 열 수: {df.shape[1]}")
            print(f"  - 열 이름: {', '.join(df.columns)}")
            
        except Exception as e:
            print(f"❌ 표 시각화 중 오류 발생: {e}")

class DocumentSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=150):  # 청크 사이즈 축소 (800 → 500), 겹침 비율 30% 유지
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # 구분자 최적화: 문단 > 문장 > 구두점 > 공백 순서로 시도
            separators=[
                "\n\n",  # 문단 구분
                "\n",    # 줄바꿈
                ".",     # 문장 끝
                "!",     # 감탄문
                "?",     # 의문문
                ";",     # 세미콜론
                ":",     # 콜론
                ",",     # 쉼표
                " ",     # 공백
                ""       # 마지막 수단
            ]
        )
        logger.info(f"DocumentSplitter 초기화: 청크 크기={chunk_size}, 겹침={chunk_overlap}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        logger.info(f"🔪 문서 분할 시작: {len(documents)}개 문서")
        print("🔪 문서를 청크로 분할 중...")
        try:
            chunks = self.text_splitter.split_documents(documents)
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = f"chunk_{i}"
            logger.info(f"✅ {len(chunks)}개의 청크 생성 완료")
            return chunks
        except Exception as e:
            logger.error(f"❌ 문서 분할 중 오류: {e}")
            return documents

# --- Query 확장 (Ollama API 이용) ---
class QueryExpander:
    def __init__(self, models: List[str] = AVAILABLE_MODELS):
        self.models = models
        logger.info(f"QueryExpander 초기화: 모델={', '.join(models)}")
        self.prompt_template = """당신은 한화손해보험 사업보고서 PDF 문서를 검색하는 시스템입니다.
주어진 원래 질문을 기반으로 사업보고서 내용을 효과적으로 검색하기 위한 3개의 변형 쿼리를 생성해주세요.

각 변형 쿼리는 다음 특성을 가져야 합니다:
1. 금융/보험 용어 중심: 원래 질문에서 금융, 보험, 재무와 관련된 핵심 키워드를 추출하여 구성
2. 사업보고서 맥락: 사업보고서에서 찾을 수 있는 정보(재무상태, 경영실적, 사업전략, 리스크, 지배구조 등)에 맞게 변형
3. 구체적인 정보 지향: 숫자, 비율, 금액, 날짜 등 구체적인 정보를 찾기 위한 표현 포함

예시:
- 원래 질문: "한화손해보험의 순이익은?"
  변형1: "한화손해보험 당기순이익 금액"
  변형2: "한화손해보험 영업이익 재무제표"
  변형3: "한화손해보험 수익 실적 연도별"

- 원래 질문: "한화손해보험의 주요 사업은?"
  변형1: "한화손해보험 주력 보험상품 종류"
  변형2: "한화손해보험 사업분야 매출 비중"
  변형3: "한화손해보험 핵심사업 전략 방향"

반드시 아래 형식의 유효한 JSON 배열로만 응답하며, 추가 설명이나 주석은 절대 포함시키지 마세요:
["변형1", "변형2", "변형3"]

원래 질문: {query}"""
    
    def _generate_with_ollama(self, prompt: str, model: str) -> str:
        if DEBUG_MODE:
            print(f"\n📝 쿼리 확장 - Ollama API 호출 중 ({model})")
        start_time = time.time()
        try:
            response = requests.post(
                f"{OLLAMA_API_BASE}/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.5, "top_p": 0.95, "num_predict": 500}  # 온도 낮춤 (0.7 → 0.5)
                }
            )
            response.raise_for_status()
            result = response.json().get("response", "")
            elapsed_time = time.time() - start_time
            if DEBUG_MODE:
                print(f"✓ 완료 ({elapsed_time:.2f}초)")
            return result
        except Exception as e:
            logger.error(f"❌ Ollama API 호출 중 오류: {e}")
            if DEBUG_MODE:
                print(f"❌ Ollama API 호출 실패: {e}")
            return ""
    
    def _generate_expansion_prompt(self, query: str) -> str:
        """쿼리 확장을 위한 프롬프트를 생성합니다."""
        return f"""다음 질문을 한화손해보험 사업보고서에서 정보를 찾기 위한 3-4개의 다양한 검색 쿼리로 확장해주세요.
원래 질문: {query}

주어진 질문은 한화손해보험 사업보고서에 관한 것입니다. 질문의 핵심 개념을 파악하고, 검색에 유용한 유사 표현과 관련 키워드를 포함하는 다양한 검색 쿼리를 생성해주세요.

다음과 같은 형식으로 작성해주세요 (번호 매기기):
1. [첫 번째 검색 쿼리]
2. [두 번째 검색 쿼리]
3. [세 번째 검색 쿼리]
4. [네 번째 검색 쿼리]

각 쿼리는 원래 질문의 핵심을 유지하되, 다른 표현이나 추가 키워드를 사용하여 검색 범위를 확장해야 합니다."""
    
    def expand_query(self, query: str) -> List[str]:
        if not query or query.strip() == "":
            return [query]
            
        print(f"\n{'─'*60}")
        print(f"📌 1단계: 쿼리 확장 및 분석")
        print(f"{'─'*60}")
        print(f"▶ 원본 쿼리: '{query}'")
        
        # 쿼리 길이에 따라 확장 전략 조정
        if len(query) < 5:  # 매우 짧은 쿼리의 경우
            print(f"⚠️ 쿼리가 너무 짧습니다. 확장 없이 그대로 사용합니다.")
            return [query]

        # 모든 쿼리에 원본 포함 (확장 안 되더라도)
        all_queries = [query]
        unique_queries = set([query])
        
        # 오류 발생 시 건너뛸 수 있도록 각 모델 개별 시도
        for model in self.models:
            try:
                print(f"\n🤖 {model} 모델로 쿼리 확장 중...")
                sys.stdout.flush()  # 버퍼 비우기
                
                prompt = self._generate_expansion_prompt(query)
                print(f"\n📝 쿼리 확장 - Ollama API 호출 중 ({model})")
                sys.stdout.flush()  # 버퍼 비우기
                
                start_time = time.time()
                responses = self._generate_with_ollama(prompt, model)
                elapsed_time = time.time() - start_time
                print(f"✓ 완료 ({elapsed_time:.2f}초)")
                sys.stdout.flush()  # 버퍼 비우기
                
                # 응답 파싱
                new_queries = []
                for line in responses.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('=='):
                        # 번호 또는 대시로 시작하는 항목 제거
                        clean_line = re.sub(r'^[\d\-\.\s]+', '', line).strip()
                        if clean_line and len(clean_line) > 5:
                            new_queries.append(clean_line)
                
                # 중복 제거
                old_count = len(unique_queries)
                unique_queries.update(new_queries)
                new_count = len(unique_queries)
                
                print(f"✓ {new_count - old_count}개의 고유 쿼리 생성됨")
                sys.stdout.flush()  # 버퍼 비우기
                
                # 쿼리가 충분히 생성된 경우 추가 확장 중단
                if new_count >= 5:
                    break
                    
            except Exception as e:
                print(f"⚠️ {model} 모델 쿼리 확장 중 오류: {e}")
                continue
        
        # 최종 쿼리 목록 생성 (원본 포함)
        all_queries = list(unique_queries)
        
        # 너무 많은 쿼리는 필터링 (최대 5개)
        if len(all_queries) > 5:
            # 원본 쿼리는 항상 포함
            filtered_queries = [query]
            # 나머지 중 가장 긴 쿼리 4개 선택 (보통 더 구체적)
            other_queries = [q for q in all_queries if q != query]
            other_queries.sort(key=len, reverse=True)
            filtered_queries.extend(other_queries[:4])
            all_queries = filtered_queries
        
        print(f"\n✅ 전체 확장 완료: {len(all_queries)}개의 고유 쿼리 생성됨")
        sys.stdout.flush()  # 버퍼 비우기
        return all_queries

# --- RAG 시스템 (벡터 검색) ---
class RAGSystem:
    def __init__(self, embedding_type: str = "bge-m3", use_hnsw: bool = True, ef_search: int = 200, ef_construction: int = 200, m: int = 64):
        print("🔧 RAG 시스템 초기화 중...")
        
        # 응답 캐시 초기화
        self._cache = {}
        self.cache_file = os.path.join(SCRIPT_DIR, "cache.json")
        
        # 항상 BGE-M3 임베딩 사용
        self.embedding_type = "bge-m3"
        self.model_config = EMBEDDING_MODELS["bge-m3"]
        print(f"  - 임베딩 모델: {self.model_config['name']} (bge-m3)")
        
        # 임베딩 모델 초기화
        self.embeddings = BGEM3Embeddings(model_name=self.model_config["name"])
        
        # QueryExpander 초기화 추가
        self.query_expander = QueryExpander(models=AVAILABLE_MODELS)
        
        # FAISS 인덱스 설정
        self.use_hnsw = use_hnsw
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.m = m
        
        self.vector_store = None
        self.index_dir = os.path.join(SCRIPT_DIR, "faiss_index")
        self.metadata_file = os.path.join(SCRIPT_DIR, "faiss_metadata.json")
        
        # 캐시 초기화 시도
        try:
            self.cache = self._load_cache()
            # 기존 캐시를 _cache에도 복사
            self._cache = self.cache.copy()
        except Exception as e:
            logger.warning(f"캐시 로드 중 오류 발생: {e}")
            print(f"⚠️ 캐시 파일 로드 실패, 새로운 캐시를 생성합니다")
            self.cache = {}
            self._cache = {}
            self._save_cache()  # 새로운 빈 캐시 파일 생성
        
        self.qa_prompt = """당신은 한화손해보험 사업보고서 내용을 분석하고 정확한 정보를 제공하는 금융 전문가입니다.
사용자의 질문에 대해 제공된 문서 내용을 기반으로 명확하고 사실적인 답변을 작성해주세요.

[지침]
1. 문서에서 찾은 핵심 정보를 먼저 나열하고, 그 내용을 바탕으로 답변을 작성하세요.
2. 숫자, 날짜, 금액 등 구체적인 수치는 정확히 인용하세요.
3. 불확실한 내용은 추측하지 말고 문서에 있는 내용만 사용하세요.
4. 답변은 3-4문장으로 간단명료하게 작성하세요.
5. 전문 용어는 가능한 쉽게 설명하세요.

질문: {question}

관련 문서:
{context}

답변 형식:
[핵심 정보]
• (찾은 핵심 정보들을 불릿으로 나열)

[답변]
(위 정보들을 바탕으로 3-4문장으로 답변)"""
        print("✅ RAG 시스템 초기화 완료")
    
    def _load_cache(self) -> Dict[str, str]:
        """캐시된 응답을 로드합니다."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"캐시 로드 중 오류: {e}")
            return {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                # self._cache로 저장하도록 수정
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
            # 그리고 동기화를 위해 self.cache도 업데이트
            self.cache = self._cache.copy()
        except Exception as e:
            logger.error(f"캐시 저장 중 오류: {e}")
            print(f"⚠️ 캐시 저장 중 오류: {e}")
    
    def _save_document_metadata(self, documents: List[Document], metadata_file: str):
        metadata = {
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'document_count': len(documents),
            'hashes': [doc.metadata.get('hash') for doc in documents if 'hash' in doc.metadata]
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def _generate_with_ollama(self, prompt: str, model: str, stream=False, **params) -> str:
        """Ollama API를 사용하여 텍스트 생성"""
        start_time = time.time()
        default_params = {"temperature": 0.7, "top_p": 0.9, "num_predict": 2048}
        default_params.update(params)  # 사용자 제공 매개변수로 기본값 업데이트
        
        # 1. 가능하면 ollama 라이브러리 사용
        if OLLAMA_AVAILABLE:
            try:
                if stream:
                    # 스트리밍 모드
                    print("\n응답: ", end="")
                    sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                    full_result = ""
                    
                    # 스트리밍 생성
                    for chunk in ollama.generate(
                        model=model,
                        prompt=prompt,
                        options=default_params,
                        stream=True
                    ):
                        chunk_content = chunk.get("response", "")
                        full_result += chunk_content
                        print(chunk_content, end="")
                        sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                    
                    print()  # 줄바꿈
                    sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                    elapsed_time = time.time() - start_time
                    print(f"✓ 스트리밍 답변 생성 완료 (ollama 라이브러리): {model} ({elapsed_time:.2f}초)")
                    return full_result
                else:
                    # 일반 모드
                    response = ollama.generate(
                        model=model,
                        prompt=prompt,
                        options=default_params
                    )
                    result = response.get("response", "")
                    elapsed_time = time.time() - start_time
                    print(f"✓ 답변 생성 완료 (ollama 라이브러리): {model} ({elapsed_time:.2f}초)")
                    return result
            except Exception as e:
                print(f"⚠️ ollama 라이브러리 호출 실패: {e}, REST API로 시도합니다.")
        
        # 2. 실패하면 REST API 사용
        try:
            if stream:
                # 스트리밍 모드
                print("\n응답: ", end="")
                sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                full_result = ""
                
                response = requests.post(
                    f"{OLLAMA_API_BASE}/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": True,
                        "options": default_params
                    },
                    stream=True
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            line_text = line.decode('utf-8')
                            json_data = json.loads(line_text)
                            chunk_content = json_data.get("response", "")
                            if chunk_content:
                                full_result += chunk_content
                                print(chunk_content, end="")
                                sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                        except json.JSONDecodeError:
                            continue
                
                print()  # 줄바꿈
                sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                elapsed_time = time.time() - start_time
                print(f"✓ 스트리밍 답변 생성 완료 (REST API): {model} ({elapsed_time:.2f}초)")
                return full_result
            else:
                # 일반 모드
                response = requests.post(
                    f"{OLLAMA_API_BASE}/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": default_params
                    }
                )
                response.raise_for_status()
                result = response.json().get("response", "")
                elapsed_time = time.time() - start_time
                print(f"✓ 답변 생성 완료 (REST API): {model} ({elapsed_time:.2f}초)")
                return result
        except Exception as e:
            logger.error(f"❌ Ollama API 호출 중 오류 (모델: {model}): {e}")
            print(f"❌ Ollama API 호출 실패 (모델: {model}): {e}")
            return f"[{model} 모델 응답 생성 실패] 오류: {str(e)}"
    
    def load_or_create_vector_store(self, documents: List[Document], force_update: bool = False) -> bool:
        if not force_update and os.path.exists(self.index_dir) and os.path.exists(self.metadata_file):
            try:
                self.vector_store = FAISS.load_local(self.index_dir, self.embeddings, allow_dangerous_deserialization=True)
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
                existing_hashes = set(existing_metadata.get('hashes', []))
                new_docs = [doc for doc in documents if doc.metadata.get('hash') not in existing_hashes]
                if new_docs:
                    self.vector_store.add_documents(new_docs)
                    self.vector_store.save_local(self.index_dir)
                    self._save_document_metadata(documents, self.metadata_file)
            except Exception as e:
                logger.error(f"❌ 인덱스 로드 중 오류: {e}")
                return self._create_new_vector_store(documents)
        else:
            return self._create_new_vector_store(documents)
        return True
    
    def _create_new_vector_store(self, documents, hnsw_space='l2'):
        """문서 임베딩 및 벡터 저장소 생성"""
        print(f"\n{'─'*60}")
        print(f"📊 벡터 저장소 생성 중... ")
        print(f"{'─'*60}")
        
        # 문서가 없는 경우 처리
        if not documents or len(documents) == 0:
            print("⚠️ 문서가 없습니다. 빈 벡터 저장소를 생성합니다.")
            try:
                # 임베딩 차원 결정
                embedding_dim = 768  # 기본 차원
                try:
                    # 테스트 문장으로 임베딩 차원 확인
                    embedding_dim = len(self.embeddings.embed_query("테스트"))
                    print(f"✓ 임베딩 차원 확인됨: {embedding_dim}")
                except Exception as e:
                    print(f"⚠️ 임베딩 차원 확인 실패: {e}, 기본값 {embedding_dim} 사용")
                
                # 빈 인덱스 구성 요소 생성
                empty_index, docstore, index_to_docstore_id = create_empty_faiss_index(embedding_dim)
                
                if empty_index is None:
                    print("❌ 빈 인덱스 생성 실패")
                    return None
                
                # 벡터 저장소 생성
                from langchain.vectorstores import FAISS
                vector_store = FAISS(
                    embedding_function=self.embeddings,
                    index=empty_index,
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id
                )
                
                print("✓ 빈 벡터 저장소 생성 완료")
                # 저장 정보 설정
                self.document_count = 0
                
                # 저장소 경로가 존재하는지 확인하고 생성
                if self.index_dir:
                    if not os.path.exists(self.index_dir):
                        os.makedirs(self.index_dir, exist_ok=True)
                    try:
                        vector_store.save_local(self.index_dir)
                        self._save_document_metadata([], self.metadata_file)
                        print(f"✓ 빈 벡터 저장소 저장 완료: {self.index_dir}")
                    except Exception as save_err:
                        print(f"⚠️ 빈 벡터 저장소 저장 실패: {save_err}")
                
                return vector_store
                
            except Exception as e:
                print(f"❌ 빈 벡터 저장소 생성 실패: {e}")
                return None
        
        print(f"🔢 문서 수: {len(documents)}")
        
        # 최대 문서 수 제한 (8000개)
        if len(documents) > 8000:
            print(f"⚠️ 문서가 너무 많습니다 ({len(documents)}개). 처음 8000개만 사용합니다.")
            documents = documents[:8000]
        
        # 문서 임베딩 전 안전 검사
        try:
            if not self.embeddings:
                print("❌ 임베딩 모델이 초기화되지 않았습니다.")
                return None
                
            print("🔄 문서 임베딩 및 벡터 저장소 생성 중...")
            start_time = time.time()
            
            # 먼저 임베딩 차원 확인
            embedding_dim = 768  # 기본 차원
            try:
                embedding_dim = len(self.embeddings.embed_query("테스트"))
                print(f"✓ 임베딩 차원: {embedding_dim}")
            except Exception as e:
                print(f"⚠️ 임베딩 차원 확인 실패: {e}, 기본값 {embedding_dim} 사용")
            
            # FAISS 벡터 저장소 생성 시도
            try:
                vector_store = None
                # HNSW 인덱스로 생성 시도
                try:
                    if hnsw_space == 'cosine':
                        print("🔄 HNSW 코사인 유사도 인덱스 생성 중...")
                        vector_store = FAISS.from_documents(
                            documents, 
                            self.embeddings,
                            normalize_L2=True,  # 코사인 유사도를 위한 정규화
                            space='inner_product',  # 내적 사용
                            m=64,  # HNSW 그래프의 이웃 수
                            ef_construction=128  # 구축 시 고려할 이웃 수
                        )
                    else:  # 'l2' 또는 기타
                        print("🔄 HNSW L2 거리 인덱스 생성 중...")
                        vector_store = FAISS.from_documents(
                            documents, 
                            self.embeddings,
                            normalize_L2=False,  # L2 거리는 정규화 필요 없음
                            space='l2',  # L2 거리 사용
                            m=64,
                            ef_construction=128
                        )
                    print("✓ HNSW 인덱스 생성 완료")
                except Exception as hnsw_error:
                    print(f"❌ HNSW 인덱스 생성 실패: {hnsw_error}")
                    print("⚠️ 기본 인덱스로 재시도 중...")
                    vector_store = None
                
                # 기본 방법으로 재시도
                if vector_store is None:
                    print("🔄 기본 인덱스 생성 중...")
                    try:
                        vector_store = FAISS.from_documents(documents, self.embeddings)
                        print("✓ 기본 인덱스 생성 완료")
                    except Exception as basic_error:
                        print(f"❌ 기본 인덱스 생성 실패: {basic_error}")
                        print("⚠️ 안전 모드로 재시도 중...")
                        
                        # 안전 모드로 벡터 저장소 생성 (직접 임베딩 생성)
                        try:
                            # 문서 내용 리스트 생성
                            texts = [doc.page_content for doc in documents]
                            
                            # 임베딩 생성
                            embeddings_list = self.embeddings.embed_documents(texts)
                            
                            # 빈 인덱스 생성
                            import numpy as np
                            import faiss
                            import uuid
                            
                            # 벡터 임베딩으로 인덱스 생성
                            index = faiss.IndexFlatL2(embedding_dim)
                            if len(embeddings_list) > 0:
                                index.add(np.array(embeddings_list, dtype=np.float32))
                            
                            # 문서 저장소 생성
                            docstore = {}
                            index_to_docstore_id = {}
                            
                            # 문서와 인덱스 매핑
                            for i, doc in enumerate(documents):
                                id = str(uuid.uuid4())
                                docstore[id] = doc
                                index_to_docstore_id[i] = id
                            
                            # FAISS 객체 생성
                            vector_store = FAISS(
                                embedding_function=self.embeddings,
                                index=index,
                                docstore=docstore,
                                index_to_docstore_id=index_to_docstore_id
                            )
                            print("✓ 안전 모드로 인덱스 생성 완료")
                        except Exception as safe_error:
                            print(f"❌ 안전 모드로도 생성 실패: {safe_error}")
                            return None
                
                elapsed_time = time.time() - start_time
                print(f"✓ 벡터 저장소 생성 완료 ({elapsed_time:.2f}초)")
                
                # 메타데이터 저장
                self.document_count = len(documents)
                
                # 저장소 경로가 존재하는지 확인하고 생성
                if self.index_dir and vector_store is not None:
                    if not os.path.exists(self.index_dir):
                        os.makedirs(self.index_dir, exist_ok=True)
                    try:
                        vector_store.save_local(self.index_dir)
                        self._save_document_metadata(documents, self.metadata_file)
                        print(f"✓ 벡터 저장소 저장 완료: {self.index_dir}")
                    except Exception as save_err:
                        print(f"⚠️ 벡터 저장소 저장 실패: {save_err}")
                
                return vector_store
                
            except Exception as e:
                print(f"❌ 벡터 저장소 생성 중 오류: {e}")
                print(f"⚠️ 남은 문서 수로 재시도: {len(documents)//2}개")
                
                if len(documents) > 1:
                    # 문서 수를 절반으로 줄여서 재시도
                    half_docs = documents[:len(documents)//2]
                    return self._create_new_vector_store(half_docs, hnsw_space)
                else:
                    print("❌ 재시도 실패: 문서가 너무 적습니다")
                    return None
                    
        except Exception as outer_e:
            print(f"❌ 벡터 저장소 생성 중 심각한 오류: {outer_e}")
            return None
    
    def search(self, query: str, top_k: int = 12) -> List[Document]:
        """쿼리에 관련된 문서를 검색합니다."""
        print("\n" + "─"*60)
        print(f"📌 3단계: 문서 검색 ({self.model_config['name']})")
        print("─"*60)
        
        try:
            # 멀티 쿼리 확장 수행
            logger.info(f"쿼리 확장 시작: '{query}'")
            expanded_queries = []
            expanded_queries.append(query)  # 원본 쿼리도 포함
            
            # QueryExpander 사용하여 쿼리 확장
            logger.info(f"🔍 쿼리 확장 중...")
            print("\n" + "─"*60)
            print(f"📌 2단계: 쿼리 확장 (핵심 모델)")
            print("─"*60)
            print(f"▶ 원본 쿼리: '{query}'")
            
            expanded = self.query_expander.expand_query(query)
            expanded_queries.extend(expanded)
            
            # 중복 제거 및 빈 문자열 제거
            expanded_queries = [q.strip() for q in expanded_queries if q.strip()]
            expanded_queries = list(dict.fromkeys(expanded_queries))  # 순서 유지하며 중복 제거
            
            print("✅ 전체 확장 완료: {}개의 고유 쿼리 생성됨".format(len(expanded_queries)))
            print("─"*60)
            for i, q in enumerate(expanded_queries, 1):
                print(f"  {i}. {q}")
            
            print("\n📊 확장된 쿼리로 검색 중...")
            
            # 각 확장 쿼리에 대해 검색 수행
            all_docs = []
            
            for i, exp_query in enumerate(expanded_queries):
                t_start = time.time()
                if not self.vector_store:
                    logger.error("벡터 저장소가 초기화되지 않았습니다.")
                    print("❌ 벡터 저장소가 초기화되지 않았습니다.")
                    return []
                
                # 각 확장 쿼리에 대해 검색
                docs = self.vector_store.similarity_search_with_score(
                    exp_query, k=top_k
                )
                
                t_end = time.time()
                elapsed = t_end - t_start
                print(f"🔍 쿼리 #{i+1}: \"{exp_query}\"")
                print(f"  ✓ {len(docs)}개 문서 검색됨 ({elapsed:.2f}초)")
                
                # 결과 병합
                all_docs.extend(docs)
            
            # 스코어 정규화 및 중복 제거
            doc_dict = {}
            for doc, score in all_docs:
                doc_id = doc.metadata.get('chunk_id', doc.page_content[:50])
                if doc_id not in doc_dict or score < doc_dict[doc_id][1]:
                    doc_dict[doc_id] = (doc, score)
            
            # 정규화된 점수 계산
            min_score = min([score for _, score in doc_dict.values()], default=0)
            max_score = max([score for _, score in doc_dict.values()], default=1)
            score_range = max_score - min_score if max_score > min_score else 1
            
            # 정규화된 점수로 정렬 (낮은 거리 = 높은 유사도)
            sorted_docs = []
            for doc_id, (doc, score) in doc_dict.items():
                normalized_score = 1 - ((score - min_score) / score_range) if score_range > 0 else 0
                sorted_docs.append((doc, normalized_score))
            
            # 점수 기준 내림차순 정렬
            sorted_docs = sorted(sorted_docs, key=lambda x: x[1], reverse=True)
            
            # 최종 결과 문서 추출
            final_docs = [doc for doc, _ in sorted_docs[:top_k]]
            
            print(f"\n✅ 검색 완료: {len(final_docs)}개 문서 선택됨")
            
            print("\n📄 검색된 문서:")
            
            for i, (doc, score) in enumerate(sorted_docs[:top_k], 1):
                print(f"\n{'='*80}")
                source_info = f"문서 #{i} | 페이지 {doc.metadata.get('page', '불명')}"
                if 'source' in doc.metadata:
                    source_info += f" | 유형: {doc.metadata['source']}"
                print(f"📑 {source_info} | 정규화 점수: {score:.4f}")
                print(f"{'─'*80}")
                
                # 검색어 하이라이트 처리
                content = doc.page_content
                for search_term in expanded_queries:
                    pattern = re.compile(re.escape(search_term), re.IGNORECASE)
                    content = pattern.sub(f"\033[93m{search_term}\033[0m", content)
                print(content)
            
            print(f"\n{'='*80}")
            return final_docs
            
        except Exception as e:
            logger.error(f"❌ 문서 검색 중 오류: {e}")
            print(f"❌ 문서 검색 중 오류 발생: {e}")
            return []

    def format_context_for_model(self, docs: List[Document]) -> str:
        """검색된 문서들을 모델에 전달할 수 있는 형식으로 변환"""
        formatted_docs = []
        for doc in docs:
            page_info = f"[페이지 {doc.metadata.get('page', '불명')}]"
            formatted_docs.append(f"{page_info} {doc.page_content}")
        return "\n\n".join(formatted_docs)
    
    def answer(self, query: str, model: str, context: str) -> Dict[str, Any]:
        """쿼리에 대한 답변을 생성합니다."""
        logger.debug(f"답변 생성 요청 - 쿼리: '{query}', 모델: {model}")
        
        # 1. 캐시에서 응답 확인
        cache_key = f"{model}:{hashlib.md5((query + context[:100]).encode()).hexdigest()}"
        cached_answer = self._load_cache().get(cache_key)
        
        if cached_answer and not os.environ.get('DISABLE_CACHE'):
            print(f"💾 캐시된 응답 사용: {model}")
            return {"answer": cached_answer, "model": model, "cached": True}
        
        # 2. 프롬프트 생성
        try:
            # 오늘 날짜 정보 추가
            today = datetime.now().strftime("%Y년 %m월 %d일")
            
            prompt_template = f"""당신은 한화손해보험의 전문가 AI 어시스턴트입니다.
주어진 정보를 바탕으로 질문에 정확하고 상세하게 답변해주세요.
오늘은 {today}입니다.

[정보]
{context}

[질문]
{query}

[답변]"""
            
            # 3. Ollama를 사용하여 응답 생성
            try:
                # 스트리밍 답변 생성
                result = self._generate_with_ollama(prompt_template, model, stream=True)
                
                # 응답 캐싱
                answer_content = result if isinstance(result, str) else result.get('answer', '')
                self._cache[cache_key] = answer_content
                self._save_cache()
                
                # 결과 반환
                return {
                    "answer": answer_content,
                    "model": model,
                    "cached": False
                }
                
            except Exception as e:
                logger.error(f"❌ 답변 생성 중 오류: {e}")
                return {"answer": f"답변을 생성하는 중 오류가 발생했습니다: {str(e)}", "model": model, "error": True}
                
        except Exception as e:
            logger.error(f"❌ 프롬프트 생성 중 오류: {e}")
            return {"answer": f"답변을 준비하는 중 오류가 발생했습니다: {str(e)}", "model": model, "error": True}

def main():
    # global 선언을 함수 시작 부분으로 이동
    global AVAILABLE_MODELS
    
    print("\n" + "="*60)
    print("📊 한화손해보험 사업보고서 RAG 시스템")
    print("(Ollama 모델 비교: Llama3.1 vs Gemma3)")
    print("="*60)
    print("🔄 Ollama 서버 연결 확인 중...")
    
    # 올라마 모델 확인 및 설정
    available_models = check_ollama_models()
    
    # 사용 가능한 모델로 AVAILABLE_MODELS 업데이트
    if available_models:
        AVAILABLE_MODELS = available_models
    
    # 명령줄 인수 가져오기
    args = parser.parse_args()
    
    # RAG 시스템 초기화 - 한 번만 생성
    print(f"\n💾 벡터 저장소 준비 중 (임베딩 모델: BGE-M3)...")
    sys.stdout.flush()  # 버퍼 비우기
    
    rag = RAGSystem(
        use_hnsw=True,
        ef_search=200,
        ef_construction=200,
        m=64
    )
    
    # PDF 파일 처리
    sys.stdout.flush()  # 버퍼 비우기
    print("\n🔍 PDF 문서 처리 시작...")
    sys.stdout.flush()  # 버퍼 비우기
    
    pdf_path = args.pdf
    if not os.path.exists(pdf_path):
        print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return
    
    pdf_processor = PDFProcessor(pdf_path)
    documents = pdf_processor.process()
    
    # 문서 분할
    splitter = DocumentSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(documents)
    
    # 벡터 저장소 생성 및 로드
    success = rag.load_or_create_vector_store(chunks, force_update=args.force_update)
    
    if not success:
        print("❌ 벡터 저장소 생성/업데이트 실패")
        return
    
    # 자동 평가 설정
    auto_evaluator = None
    if args.auto_eval:
        try:
            auto_evaluator = AutoEvaluator()
        except Exception as e:
            print(f"⚠️ 자동 평가 모듈 초기화 실패: {e}")
            print("⚠️ 자동 평가 기능이 비활성화됩니다.")
    
    # 모델 평가자 초기화
    evaluator = ModelEvaluator()
    
    # 상호작용 모드 또는 자동 테스트 모드 실행
    print("\n" + "="*80)
    print("🚀 RAG 시스템 준비 완료!")
    print(f"  - 임베딩 모델: {rag.model_config['name']}")
    print(f"  - 인덱스 디렉토리: {rag.index_dir}")
    
    if args.auto_eval and auto_evaluator:
        print(f"  - 자동 평가: 활성화 (gemma3:12b)")
    else:
        print(f"  - 자동 평가: 비활성화")
    
    # 자동 테스트 모드
    if args.auto_test:
        print("\n🧪 자동 테스트 모드 시작...")
        auto_question_generator = AutoQuestionGenerator()
        auto_test_manager = AutoTestManager(rag, auto_question_generator, auto_evaluator, evaluator, AVAILABLE_MODELS)
        auto_test_manager.run_auto_test(num_questions=args.num_questions, top_k=args.top_k)
        return
        
    # 상호작용 모드 안내
    print("\n💬 명령어 목록:")
    print("  - '자동 테스트' 또는 'auto': 자동 질문 생성 및 평가 모드 실행")
    print("  - '종료' 또는 'quit': 프로그램 종료")
    print("  - 그 외: 질문으로 처리")
    
    print("\n질문이나 명령어를 입력하세요:")
    print("="*80)
    
    while True:
        print("\n💡 표준 입력으로부터 쿼리를 읽는 중...")
        sys.stdout.flush()  # 버퍼 비우기
        
        try:
            query = input().strip()
            print(f"\n💬 입력: 💬 쿼리 입력 완료: '{query}'")
            sys.stdout.flush()  # 버퍼 비우기
            
            # 종료 명령
            if query.lower() in ('종료', 'quit', 'exit'):
                print("👋 프로그램을 종료합니다.")
                break
                
            # 자동 테스트 모드
            if query.lower() in ('자동 테스트', 'auto', 'auto test'):
                print("\n🧪 자동 테스트 모드 시작...")
                sys.stdout.flush()  # 버퍼 비우기
                
                try:
                    num_questions = int(input("생성할 질문 수를 입력하세요(1-10): ").strip())
                    num_questions = max(1, min(10, num_questions))
                except (ValueError, EOFError):
                    print("⚠️ 유효하지 않은 입력, 기본값 5개 질문으로 설정")
                    num_questions = 5
                
                print(f"✓ {num_questions}개 질문 생성 및 테스트 시작")
                sys.stdout.flush()  # 버퍼 비우기
                
                auto_question_generator = AutoQuestionGenerator()
                auto_test_manager = AutoTestManager(rag, auto_question_generator, auto_evaluator, evaluator, AVAILABLE_MODELS)
                auto_test_manager.run_auto_test(num_questions=num_questions, top_k=args.top_k)
                continue
            
            # 빈 입력 무시
            if not query:
                continue
                
            # PDF 검색 및 답변 생성
            print(f"\n🔍 문서 검색 중...")
            sys.stdout.flush()  # 버퍼 비우기
            
            docs = rag.search(query, top_k=args.top_k)
            print(f"✅ 검색 완료: {len(docs)}개의 문서가 검색되었습니다.")
            if not docs:
                print("❌ 관련 문서를 찾을 수 없습니다.")
                continue
            
            context = rag.format_context_for_model(docs)
            print("\n💡 문서 검색 결과를 모델에 전달하여 답변을 생성합니다...")
            
            results = {}
            auto_evaluations = {}
            
            for i, model in enumerate(AVAILABLE_MODELS):
                print(f"\n📌 [{i+1}/{len(AVAILABLE_MODELS)}] {model} 모델로 답변 생성 중...")
                result = rag.answer(query, model, context)
                answer = result["answer"]
                results[model] = answer
                print(f"✅ {model} 모델 답변 생성 완료.")
                
                # 자동 평가 옵션이 활성화된 경우
                if args.auto_eval:
                    print(f"\n📊 {model} 모델 답변 자동 평가 중...")
                    evaluation = auto_evaluator.evaluate_answer(query, context, answer)
                    auto_evaluations[model] = evaluation
                    print(f"✅ {model} 모델 자동 평가 완료.")
            
            print("\n💡 모든 모델 처리 완료, 결과 저장 중...")
            # 결과 저장 및 평가 ID 생성
            evaluation_id = evaluator.save_evaluation(
                query=query,
                context=context,
                results=results,
                metadata={
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "auto_evaluations": auto_evaluations if args.auto_eval else None
                }
            )
            print(f"✅ 평가 ID '{evaluation_id}' 생성 완료.")
            
            print("\n" + "="*80)
            print(f"💬 질문: {query}")
            print(f"📝 평가 ID: {evaluation_id}")
            print("="*80)
            
            for model, answer in results.items():
                print(f"\n{'─'*80}")
                print(f"📝 모델: {model}")
                if args.auto_eval and model in auto_evaluations:
                    eval_data = auto_evaluations[model]
                    score = eval_data.get("score")
                    score_text = f"점수: {score}/5" if score is not None else "점수: 평가 불가"
                    print(f"🤖 자동 평가: {score_text}")
                print(f"{'─'*80}")
                print(answer)
                
                # 자동 평가 세부 결과 표시
                if args.auto_eval and model in auto_evaluations:
                    eval_data = auto_evaluations[model]
                    print(f"\n📊 자동 평가 세부 결과:")
                    print(f"{'─'*40}")
                    print(f"평가 이유: {eval_data.get('reason', '평가 이유 추출 실패')}")
            
            # 자동 평가 결과를 ModelEvaluator에 저장 (수동 평가 입력 제거)
            if args.auto_eval:
                print("\n💡 자동 평가 결과 저장 중...")
                for model, eval_data in auto_evaluations.items():
                    score = eval_data.get("score")
                    if score is not None:
                        evaluator.add_evaluation_score(
                            evaluation_id=evaluation_id,
                            model=model,
                            score=score,
                            comments=eval_data.get("reason", "")[:200]  # 이유 요약
                        )
                print("✅ 자동 평가 결과 저장 완료.")
            
            # 멀티쿼리 결과를 JSON 파일로 저장하여 streamlit에서 조회 가능하게 함
            print("\n💡 쿼리 결과 JSON 파일 저장 중...")
            query_results_file = os.path.join("query_results", f"query_{evaluation_id}.json")
            os.makedirs("query_results", exist_ok=True)
            
            # 결과 저장
            query_result_data = {
                "query": query,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_id": evaluation_id,
                "results": results,
                "auto_evaluations": auto_evaluations if args.auto_eval else None,
                "context": context
            }
            
            with open(query_results_file, 'w', encoding='utf-8') as f:
                json.dump(query_result_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 쿼리 결과가 저장되었습니다: {query_results_file}")
            print(f"Streamlit에서 이 결과를 조회할 수 있습니다.")
            
        except EOFError:
            print("\n👋 EOF 감지, 프로그램 종료")
            break
        except Exception as e:
            logger.error(f"치명적 오류 발생: {e}")
            logger.error(traceback.format_exc())
            print(f"❌ 치명적인 오류 발생: {e}")
            print("세부 오류 내용:", traceback.format_exc())

def streamlit_main():
    import streamlit as st
    
    # Streamlit 페이지 설정
    st.set_page_config(
        page_title="한화손해보험 사업보고서 RAG 시스템",
        page_icon="📊",
        layout="wide"
    )
    
    # 제목 및 설명
    st.title("📊 한화손해보험 사업보고서 RAG 시스템")
    st.subheader("(Ollama 모델 비교: Llama3.1 vs Gemma3)")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 환경 설정")
    
    # PDF 파일 선택
    pdf_files = glob.glob(os.path.join(SCRIPT_DIR, "*.pdf"))
    if not pdf_files:
        st.sidebar.warning("PDF 파일을 찾을 수 없습니다.")
        selected_pdf = PDF_PATH  # 기본 PDF 경로 사용
    else:
        selected_pdf = st.sidebar.selectbox(
            "PDF 파일 선택",
            options=pdf_files,
            index=0
        )
    
    # 청크 크기 및 겹침 설정
    chunk_size = st.sidebar.slider("청크 크기", min_value=100, max_value=1000, value=500, step=50)
    chunk_overlap = st.sidebar.slider("청크 겹침", min_value=50, max_value=300, value=150, step=25)
    
    # 검색 결과 수 설정
    top_k = st.sidebar.slider("검색 결과 수 (Top-K)", min_value=3, max_value=20, value=10)
    
    # 인덱스 강제 업데이트 옵션
    force_update = st.sidebar.checkbox("벡터 인덱스 강제 업데이트", value=False)
    
    # HNSW 인덱스 사용 옵션
    use_hnsw = st.sidebar.checkbox("HNSW 인덱스 사용 (정확도 향상)", value=True)
    
    # 자동 평가 옵션
    auto_eval = st.sidebar.checkbox("자동 평가 활성화 (gemma3:12b 필요)", value=True)
    
    # 탭 설정
    tab1, tab2, tab3 = st.tabs(["💬 질문 응답", "🔄 자동 테스트", "🔍 디버그"])
    
    # RAG 시스템 초기화
    logger = setup_logging()
    evaluator = ModelEvaluator()
    auto_evaluator = AutoEvaluator(model_name="gemma3:12b")
    auto_question_generator = AutoQuestionGenerator(model_name="gemma3:12b")
    
    # Ollama 서버 연결 및 모델 확인
    print("🔄 Ollama 모델 확인 중...")
    available_models = check_ollama_models()
    
    # 사용 가능한 모델을 스트림릿에 표시
    if available_models:
        st.sidebar.success(f"✓ 사용 가능한 모델: {', '.join(available_models)}")
    else:
        st.sidebar.warning("⚠️ 사용 가능한 모델이 없습니다. 기본 모델을 사용합니다.")
    
    # 질문 응답 탭
    with tab1:
        st.header("질문 응답")
        
        # 저장된 쿼리 결과 확인
        query_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "query_results")
        os.makedirs(query_results_dir, exist_ok=True)
        query_results_files = glob.glob(os.path.join(query_results_dir, "query_*.json"))
        query_results_files = sorted(query_results_files, key=os.path.getmtime, reverse=True)
        
        if query_results_files:
            st.subheader("📝 저장된 쿼리 결과")
            selected_result_file = st.selectbox(
                "이전 쿼리 결과 선택",
                options=["새 쿼리 입력"] + query_results_files,
                format_func=lambda x: f"새 쿼리 입력" if x == "새 쿼리 입력" else f"{os.path.basename(x)} - {time.ctime(os.path.getmtime(x))}"
            )
            
            # 이전 쿼리 결과 로드
            if selected_result_file != "새 쿼리 입력":
                try:
                    with open(selected_result_file, 'r', encoding='utf-8') as f:
                        query_data = json.load(f)
                    
                    # 데이터 표시
                    st.subheader(f"질문: {query_data['query']}")
                    st.caption(f"생성 시간: {query_data['timestamp']}")
                    
                    # 문서 컨텍스트 표시
                    if 'context' in query_data:
                        with st.expander("🔍 검색된 문서 컨텍스트", expanded=False):
                            st.markdown(query_data['context'])
                    
                    # 각 모델의 응답 표시
                    if 'results' in query_data:
                        for model, answer in query_data['results'].items():
                            with st.expander(f"📝 모델: {model}", expanded=True):
                                st.markdown(answer)
                                
                                # 자동 평가 결과 표시
                                if query_data.get('auto_evaluations') and model in query_data['auto_evaluations']:
                                    eval_data = query_data['auto_evaluations'][model]
                                    score = eval_data.get('score')
                                    reason = eval_data.get('reason', '평가 정보 없음')
                                    
                                    st.divider()
                                    st.markdown("**🤖 자동 평가 결과:**")
                                    if score is not None:
                                        st.markdown(f"**점수**: {score}/5")
                                    st.markdown(f"**평가 이유**: {reason}")
                except Exception as e:
                    st.error(f"쿼리 결과 파일 로드 중 오류: {e}")
        
        # 새 쿼리 입력
        if not query_results_files or selected_result_file == "새 쿼리 입력":
            # 질문 입력
            if 'last_question' not in st.session_state:
                st.session_state.last_question = ""
                
            question = st.text_area("질문을 입력하세요:", height=100, value=st.session_state.last_question)
            
            if st.button("질문 제출", type="primary", disabled=(not selected_pdf)):
                if question:
                    st.session_state.last_question = question
                    
                    with st.spinner("질문에 답변 생성 중..."):
                        try:
                            # RAG 시스템 초기화
                            rag = RAGSystem(
                                use_hnsw=use_hnsw,
                                ef_search=200,
                                ef_construction=200,
                                m=64
                            )
                            
                            # PDF 처리
                            processor = PDFProcessor(selected_pdf)
                            documents = processor.process()
                            
                            # 기존 인덱스가 있고 PDF가 변경되지 않았다면 인덱스 로드만 실행
                            if not documents and not force_update:
                                success = rag.load_or_create_vector_store([], force_update=False)
                                if not success:
                                    st.error("❌ 기존 인덱스 로드 실패")
                                    return
                            else:
                                # 새 문서가 있는 경우 분할 후 인덱싱
                                splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                                chunks = splitter.split_documents(documents)
                                
                                # 가비지 컬렉션
                                import gc
                                gc.collect()
                                time.sleep(1)
                                
                                # 새 인덱스 생성 또는 업데이트
                                success = rag.load_or_create_vector_store(chunks, force_update=force_update)
                                if not success:
                                    st.error("❌ 벡터 저장소 생성/업데이트 실패")
                                    return
                            
                            # 쿼리 처리
                            retrieved_docs = rag.search(question, top_k=top_k)
                            context = rag.format_context_for_model(retrieved_docs)
                            
                            # 결과 변수
                            results = {}
                            auto_evaluations = {}
                            evaluation_id = None
                            
                            # 모델 평가 생성
                            if auto_eval:
                                evaluation_id = evaluator.save_evaluation(question, context, {}, {
                                    "top_k": top_k,
                                    "has_auto_eval": True
                                })
                            
                            # 각 모델에 대해 답변 생성
                            progress_bar = st.progress(0)
                            for i, model in enumerate(AVAILABLE_MODELS):
                                progress = (i / len(AVAILABLE_MODELS)) * 100
                                progress_bar.progress(int(progress))
                                
                                # 모델 응답 생성
                                result = rag.answer(question, model, context)
                                answer = result["answer"]
                                results[model] = answer
                                
                                # 자동 평가 실행
                                if auto_eval:
                                    auto_evaluation = auto_evaluator.evaluate_answer(question, context, answer)
                                    auto_evaluations[model] = auto_evaluation
                                    
                                    # 평가 결과 저장
                                    if evaluation_id:
                                        score = auto_evaluation.get("score", 0)
                                        comments = auto_evaluation.get("reason", "")
                                        evaluator.add_evaluation_score(evaluation_id, model, score, comments)
                            
                            progress_bar.progress(100)
                            
                            # 결과 저장
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            query_id = re.sub(r'\W+', '_', question)[:30]
                            query_results_file = os.path.join(query_results_dir, f"query_{timestamp}_{query_id}.json")
                            
                            query_result_data = {
                                "query": question,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "evaluation_id": evaluation_id,
                                "results": results,
                                "auto_evaluations": auto_evaluations if auto_eval else None,
                                "context": context
                            }
                            
                            with open(query_results_file, 'w', encoding='utf-8') as f:
                                json.dump(query_result_data, f, ensure_ascii=False, indent=2)
                            
                            # 결과 표시
                            st.success("✅ 응답이 생성되었습니다")
                            
                            # 검색된 문서 컨텍스트 표시
                            with st.expander("🔍 검색된 문서 컨텍스트", expanded=False):
                                st.markdown(context)
                            
                            # 각 모델의 응답 표시
                            for model, answer in results.items():
                                with st.expander(f"📝 모델: {model}", expanded=True):
                                    st.markdown(answer)
                                    
                                    # 자동 평가 결과 표시
                                    if auto_eval and model in auto_evaluations:
                                        eval_data = auto_evaluations[model]
                                        score = eval_data.get("score")
                                        reason = eval_data.get("reason", "평가 정보 없음")
                                        
                                        st.divider()
                                        st.markdown("**🤖 자동 평가 결과:**")
                                        if score is not None:
                                            st.markdown(f"**점수**: {score}/5")
                                        st.markdown(f"**평가 이유**: {reason}")
                            
                        except Exception as e:
                            st.error(f"❌ 오류 발생: {e}")
                            logger.error(f"치명적 오류 발생: {e}")
                            logger.error(traceback.format_exc())
    
    # 자동 테스트 탭
    with tab2:
        st.header("자동 테스트")
        st.write("RAG 시스템의 자동 테스트를 실행합니다.")
        
        # 자동 테스트 설정
        num_questions = st.slider("생성할 질문 수", min_value=1, max_value=20, value=5)
        
        if st.button("자동 테스트 실행", type="primary"):
            with st.spinner("자동 테스트 실행 중..."):
                try:
                    # RAG 시스템 초기화
                    rag = RAGSystem(
                        use_hnsw=use_hnsw,
                        ef_search=200,
                        ef_construction=200,
                        m=64
                    )
                    
                    # PDF 처리
                    processor = PDFProcessor(selected_pdf)
                    documents = processor.process()
                    
                    # 기존 인덱스가 있고 PDF가 변경되지 않았다면 인덱스 로드만 실행
                    if not documents and not force_update:
                        success = rag.load_or_create_vector_store([], force_update=False)
                        if not success:
                            st.error("❌ 기존 인덱스 로드 실패")
                            return
                    else:
                        # 새 문서가 있는 경우 분할 후 인덱싱
                        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        chunks = splitter.split_documents(documents)
                        
                        # 가비지 컬렉션
                        import gc
                        gc.collect()
                        time.sleep(1)
                        
                        # 새 인덱스 생성 또는 업데이트
                        success = rag.load_or_create_vector_store(chunks, force_update=force_update)
                        if not success:
                            st.error("❌ 벡터 저장소 생성/업데이트 실패")
                            return
                    
                    # 자동 테스트 관리자 초기화
                    auto_test_manager = AutoTestManager(
                        rag_system=rag,
                        auto_question_generator=auto_question_generator,
                        auto_evaluator=auto_evaluator,
                        evaluator=evaluator,
                        available_models=available_models
                    )
                    
                    # 자동 테스트 실행
                    st.session_state.auto_test_results = []
                    
                    # 프로그레스 바 추가
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 임의의 문서 추출
                    random_doc_indices = random.sample(range(len(chunks)), min(5, len(chunks)))
                    random_docs = [chunks[i] for i in random_doc_indices]
                    
                    # 생성할 질문 수 조정
                    status_text.text("문서에서 질문 생성 중...")
                    for i, doc in enumerate(random_docs):
                        # 문서에서 질문 생성
                        context = doc.page_content
                        doc_questions = auto_question_generator.generate_questions(context)
                        
                        # 질문마다 테스트 실행
                        for j, question in enumerate(doc_questions[:max(1, num_questions // len(random_docs))]):
                            current_progress = (i * max(1, num_questions // len(random_docs)) + j) / num_questions * 100
                            progress_bar.progress(int(current_progress))
                            status_text.text(f"질문 처리 중 ({i * max(1, num_questions // len(random_docs)) + j + 1}/{num_questions}): {question}")
                            
                            # 질문에 대한 검색 수행
                            retrieved_docs = rag.search(question, top_k=top_k)
                            context = rag.format_context_for_model(retrieved_docs)
                            
                            # 모델별 답변 생성
                            results = {}
                            auto_evaluations = {}
                            
                            for model in available_models:
                                # 모델 응답 생성
                                result = rag.answer(question, model, context)
                                answer = result["answer"]
                                results[model] = answer
                                
                                # 자동 평가 실행
                                auto_evaluation = auto_evaluator.evaluate_answer(question, context, answer)
                                auto_evaluations[model] = auto_evaluation
                            
                            # 결과 저장
                            test_result = {
                                "question": question,
                                "context": context,
                                "results": results,
                                "auto_evaluations": auto_evaluations
                            }
                            st.session_state.auto_test_results.append(test_result)
                    
                    progress_bar.progress(100)
                    status_text.text("자동 테스트 완료!")
                    
                    # 결과 표시
                    st.success(f"✅ {len(st.session_state.auto_test_results)}개의 질문에 대한 자동 테스트가 완료되었습니다")
                    
                    # 성능 요약
                    st.subheader("📊 모델 성능 요약")
                    model_scores = {model: [] for model in available_models}
                    
                    for result in st.session_state.auto_test_results:
                        for model, eval_data in result["auto_evaluations"].items():
                            if "score" in eval_data:
                                model_scores[model].append(eval_data["score"])
                    
                    # 평균 점수 계산 및 표시
                    summary_data = []
                    for model, scores in model_scores.items():
                        if scores:
                            avg_score = sum(scores) / len(scores)
                            summary_data.append({
                                "모델": model,
                                "평균 점수": f"{avg_score:.2f}",
                                "테스트 수": len(scores)
                            })
                    
                    if summary_data:
                        st.table(summary_data)
                    
                    # 개별 결과 표시
                    st.subheader("🔍 개별 질문 결과")
                    for i, result in enumerate(st.session_state.auto_test_results):
                        with st.expander(f"질문 {i+1}: {result['question']}", expanded=i==0):
                            # 검색된 문서 컨텍스트 표시
                            with st.expander("🔍 검색된 문서 컨텍스트", expanded=False):
                                st.markdown(result["context"])
                            
                            # 각 모델의 응답 표시
                            for model, answer in result["results"].items():
                                with st.expander(f"📝 모델: {model}", expanded=True):
                                    st.markdown(answer)
                                    
                                    # 자동 평가 결과 표시
                                    if model in result["auto_evaluations"]:
                                        eval_data = result["auto_evaluations"][model]
                                        score = eval_data.get("score")
                                        reason = eval_data.get("reason", "평가 정보 없음")
                                        
                                        st.divider()
                                        st.markdown("**🤖 자동 평가 결과:**")
                                        if score is not None:
                                            st.markdown(f"**점수**: {score}/5")
                                        st.markdown(f"**평가 이유**: {reason}")
                    
                except Exception as e:
                    st.error(f"❌ 자동 테스트 중 오류 발생: {e}")
                    logger.error(f"치명적 오류 발생: {e}")
                    logger.error(traceback.format_exc())
    
    # 디버그 탭
    with tab3:
        st.header("디버그 정보")
        
        # 시스템 정보
        st.subheader("시스템 정보")
        st.json({
            "Python 버전": sys.version,
            "실행 경로": sys.executable,
            "운영체제": platform.platform(),
            "메모리 정보": psutil.virtual_memory()._asdict() if "psutil" in sys.modules else "psutil 모듈 없음"
        })
        
        # Ollama 모델 정보
        st.subheader("Ollama 모델 정보")
        try:
            response = requests.get(f"{OLLAMA_API_BASE}/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            st.table([{
                "모델명": model.get("name"),
                "크기": f"{model.get('size') / 1024 / 1024 / 1024:.2f} GB" if model.get('size') else "알 수 없음",
                "수정일": datetime.fromtimestamp(model.get('modified_at', 0)).strftime('%Y-%m-%d %H:%M:%S') if model.get('modified_at') else "알 수 없음"
            } for model in models])
        except Exception as e:
            st.error(f"Ollama 서버 연결 오류: {e}")
        
        # 평가 통계
        st.subheader("모델 평가 통계")
        summary = evaluator.get_evaluation_summary()
        if summary:
            summary_data = []
            for model, stats in summary.items():
                summary_data.append({
                    "모델": model,
                    "평가 횟수": stats.get('total_evaluations', 0),
                    "평균 점수": f"{stats.get('avg_score', 0):.2f}",
                    "평균 속도": f"{stats.get('avg_speed', 0):.2f} 토큰/초" if stats.get('avg_speed') else "알 수 없음"
                })
            st.table(summary_data)
        else:
            st.info("아직 평가 데이터가 없습니다.")

def create_empty_faiss_index(dimension=768):
    """빈 FAISS 인덱스 생성 유틸리티 함수"""
    try:
        import faiss
        import numpy as np
        from langchain.docstore.in_memory import InMemoryDocstore
        
        # 빈 FAISS 인덱스 생성
        empty_index = faiss.IndexFlatL2(dimension)
        
        # 빈 문서 저장소 생성
        docstore = InMemoryDocstore({})
        
        # 빈 매핑 생성
        index_to_docstore_id = {}
        
        return empty_index, docstore, index_to_docstore_id
        
    except Exception as e:
        print(f"❌ 빈 FAISS 인덱스 생성 중 오류: {e}")
        # 최소한의 기본 객체 반환
        return None, None, None

# 모델 평가 모듈
class ModelEvaluator:
    def __init__(self, evaluation_file: str = None):
        if evaluation_file is None:
            evaluation_file = os.path.join(SCRIPT_DIR, "model_evaluations.json")
        self.evaluation_file = evaluation_file
        self.evaluations = self._load_evaluations()
    
    def _load_evaluations(self) -> Dict:
        """기존 평가 데이터를 로드합니다."""
        try:
            if os.path.exists(self.evaluation_file):
                with open(self.evaluation_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {"evaluations": [], "statistics": {}}
        except Exception as e:
            logger.error(f"평가 데이터 로드 중 오류: {e}")
            return {"evaluations": [], "statistics": {}}
    
    def save_evaluation(self, query: str, context: str, results: Dict[str, str], 
                       metadata: Dict = None) -> str:
        """모델 응답을 저장하고 평가 ID를 반환합니다."""
        try:
            evaluation_id = hashlib.md5(f"{query}_{time.time()}".encode()).hexdigest()[:8]
            
            evaluation_entry = {
                "id": evaluation_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "context_summary": context[:200] + "..." if len(context) > 200 else context,
                "results": results,
                "metadata": metadata or {},
                "evaluations": {model: {
                    "score": None,  # 1-5: 전반적인 응답 품질
                    "comments": ""  # 평가 코멘트
                } for model in results.keys()}
            }
            
            self.evaluations["evaluations"].append(evaluation_entry)
            self._save_evaluations()
            
            return evaluation_id
        except Exception as e:
            logger.error(f"평가 저장 중 오류: {e}")
            return None
    
    def add_evaluation_score(self, evaluation_id: str, model: str, 
                           score: int, comments: str = "") -> bool:
        """특정 평가 항목에 점수와 코멘트를 추가합니다."""
        try:
            for entry in self.evaluations["evaluations"]:
                if entry["id"] == evaluation_id:
                    if model in entry["evaluations"]:
                        entry["evaluations"][model]["score"] = score
                        entry["evaluations"][model]["comments"] = comments
                        self._save_evaluations()
                        return True
            return False
        except Exception as e:
            logger.error(f"평가 점수 추가 중 오류: {e}")
            return False
    
    def get_evaluation_summary(self) -> Dict:
        """모든 평가의 통계 요약을 생성합니다."""
        try:
            # 모든 모델의 평가 데이터를 저장할 딕셔너리 초기화
            model_stats = {}
            
            # 모든 평가 데이터를 순회하면서 각 모델의 평가 정보 수집
            for entry in self.evaluations["evaluations"]:
                for model, eval_data in entry["evaluations"].items():
                    if model not in model_stats:
                        model_stats[model] = {
                            "total_evaluations": 0,
                            "total_score": 0.0,
                            "avg_score": 0.0
                        }
                    
                    if eval_data.get("score") is not None:  # score가 있는 경우만 처리
                        model_stats[model]["total_evaluations"] += 1
                        model_stats[model]["total_score"] += eval_data["score"]
            
            # 평균 점수 계산
            for model_data in model_stats.values():
                if model_data["total_evaluations"] > 0:
                    model_data["avg_score"] = round(
                        model_data["total_score"] / model_data["total_evaluations"], 
                        2
                    )
                del model_data["total_score"]  # 중간 계산에 사용된 필드 제거
            
            # 통계 저장 및 반환
            self.evaluations["statistics"] = model_stats
            self._save_evaluations()
            return model_stats
            
        except Exception as e:
            logger.error(f"평가 요약 생성 중 오류: {e}")
            return {}
    
    def _save_evaluations(self):
        """평가 데이터를 파일에 저장합니다."""
        try:
            with open(self.evaluation_file, 'w', encoding='utf-8') as f:
                json.dump(self.evaluations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"평가 데이터 저장 중 오류: {e}")

class AutoEvaluator:
    def __init__(self, model_name: str = "gemma3:12b"):
        self.model_name = model_name
        self.eval_prompt_template = """당신은 질문-답변 쌍의 품질을 평가하는 엄격한 평가자입니다.
사용자 질문, 관련 문서 컨텍스트, 모델 답변이 주어집니다.
문서 컨텍스트를 참조하여 답변이 정확하고 관련성이 높은지 평가해주세요.

다음 기준으로 1-5점 척도로 평가하세요:
1: 완전히 잘못된 정보를 제공하거나 질문과 관련 없는 답변
2: 부분적으로 관련은 있지만 부정확하거나 불완전한 답변
3: 기본적인 질문에 답변했지만 세부 정보가 부족하거나 약간의 오류가 있음
4: 정확하고 관련성 높은 답변이지만 완벽하지 않음
5: 완벽하게 정확하고 포괄적이며 문서 컨텍스트에 충실한 답변

[질문]
{question}

[문서 컨텍스트]
{context}

[모델 답변]
{answer}

[평가]
위 답변에 대한 평가를 1-5점 척도로 수행하고, 그 이유를 설명해주세요.
답변 형식:
점수: (1-5 사이의 정수만 입력)
이유: (평가 이유 설명)"""
        
    def _generate_with_ollama(self, prompt: str, stream=False) -> str:
        """Ollama API를 통해 평가 생성"""
        print(f"\n🤖 자동 평가 생성 중... ({self.model_name})")
        start_time = time.time()
        
        # Ollama 라이브러리 사용 가능 여부 확인
        if OLLAMA_AVAILABLE:
            try:
                if stream:
                    # 스트리밍 모드
                    print("\n평가 응답: ", end="")
                    sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                    full_result = ""
                    
                    # 스트리밍 생성
                    for chunk in ollama.generate(
                        model=self.model_name,
                        prompt=prompt,
                        options={"temperature": 0.2, "top_p": 0.95, "num_predict": 1000},
                        stream=True
                    ):
                        chunk_content = chunk.get("response", "")
                        full_result += chunk_content
                        print(chunk_content, end="")
                        sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                    
                    print()  # 줄바꿈
                    sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                    elapsed_time = time.time() - start_time
                    print(f"✓ 평가 생성 완료 ({elapsed_time:.2f}초)")
                    return full_result
                else:
                    # 일반 모드
                    response = ollama.generate(
                        model=self.model_name,
                        prompt=prompt,
                        options={"temperature": 0.2, "top_p": 0.95, "num_predict": 1000}
                    )
                    result = response.get("response", "")
                    elapsed_time = time.time() - start_time
                    print(f"✓ 평가 생성 완료 ({elapsed_time:.2f}초)")
                    return result
            except Exception as e:
                print(f"⚠️ ollama 라이브러리 호출 실패: {e}, REST API로 시도합니다.")
        
        # REST API 사용
        try:
            if stream:
                # 스트리밍 모드
                print("\n평가 응답: ", end="")
                full_result = ""
                
                response = requests.post(
                    f"{OLLAMA_API_BASE}/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": True,
                        "options": {"temperature": 0.2, "top_p": 0.95, "num_predict": 1000}
                    },
                    stream=True
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            line_text = line.decode('utf-8')
                            json_data = json.loads(line_text)
                            chunk_content = json_data.get("response", "")
                            if chunk_content:
                                full_result += chunk_content
                                print(chunk_content, end="")
                                sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                        except json.JSONDecodeError:
                            continue
                
                print()  # 줄바꿈
                sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                elapsed_time = time.time() - start_time
                print(f"✓ 스트리밍 평가 생성 완료 ({elapsed_time:.2f}초)")
                return full_result
            else:
                # 일반 모드
                response = requests.post(
                    f"{OLLAMA_API_BASE}/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2, "top_p": 0.95, "num_predict": 1000}
                    }
                )
                response.raise_for_status()
                result = response.json().get("response", "")
                elapsed_time = time.time() - start_time
                print(f"✓ 평가 생성 완료 ({elapsed_time:.2f}초)")
                return result
        except Exception as e:
            logger.error(f"❌ 자동 평가 생성 중 오류: {e}")
            print(f"❌ 자동 평가 생성 중 오류: {e}")
            return "평가 실패"
    
    def evaluate_answer(self, question: str, context: str, answer: str, stream: bool = True) -> Dict[str, Any]:
        """답변의 품질을 자동으로 평가"""
        # 컨텍스트 길이 제한 (평가 모델의 컨텍스트 윈도우 고려)
        max_context_length = 1500
        if len(context) > max_context_length:
            context_parts = context.split("\n\n")
            optimized_context = []
            current_length = 0
            
            for part in context_parts:
                if current_length + len(part) <= max_context_length:
                    optimized_context.append(part)
                    current_length += len(part)
                else:
                    break
            
            context = "\n\n".join(optimized_context)
        
        prompt = self.eval_prompt_template.format(
            question=question,
            context=context,
            answer=answer
        )
        
        evaluation_result = self._generate_with_ollama(prompt, stream=stream)
        
        # 점수 추출
        score_pattern = r"점수:\s*(\d+)"
        score_match = re.search(score_pattern, evaluation_result)
        
        score = None
        reason = evaluation_result
        
        if score_match:
            try:
                score = int(score_match.group(1))
                # 점수 범위 확인 (1-5)
                if score < 1 or score > 5:
                    score = None
            except ValueError:
                score = None
            
            # 이유 추출
            reason_pattern = r"이유:(.*?)(?=$|점수:)"
            reason_match = re.search(reason_pattern, evaluation_result, re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()
        
        return {
            "score": score,
            "reason": reason,
            "raw_evaluation": evaluation_result
        }

class AutoQuestionGenerator:
    def __init__(self, model_name: str = "gemma3:12b"):
        self.model_name = model_name
        self.question_prompt_template = """당신은 한화손해보험 사업보고서에 관한 질문을 생성하는 AI입니다.
제공된 문서 내용을 기반으로 구체적이고 명확한 질문을 생성해주세요.

다음 유형의 질문을 생성하세요:
1. 재무/실적 관련 질문 (수익, 손실, 성장률 등)
2. 사업 전략 관련 질문
3. 리스크 관리 관련 질문
4. 지배구조 관련 질문
5. 상품/서비스 관련 질문

[문서 내용]
{context}

위 내용을 기반으로 다양한 유형의 질문 5개를 생성해주세요.
질문은 구체적이어야 하며, 문서 내용에서 답변할 수 있는 것이어야 합니다.
다음 형식으로 JSON 배열만 출력하세요:
["질문1", "질문2", "질문3", "질문4", "질문5"]"""

    def _generate_with_ollama(self, prompt: str, stream=False) -> str:
        """Ollama API를 통해 질문 생성"""
        print(f"\n🤖 자동 질문 생성 중... ({self.model_name})")
        start_time = time.time()
        
        # Ollama 라이브러리 사용 가능 여부 확인
        if OLLAMA_AVAILABLE:
            try:
                if stream:
                    # 스트리밍 모드
                    print("\n질문 생성 응답: ", end="")
                    sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                    full_result = ""
                    
                    # 스트리밍 생성
                    for chunk in ollama.generate(
                        model=self.model_name,
                        prompt=prompt,
                        options={"temperature": 0.7, "top_p": 0.95, "num_predict": 1000},
                        stream=True
                    ):
                        chunk_content = chunk.get("response", "")
                        full_result += chunk_content
                        print(chunk_content, end="")
                        sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                    
                    print()  # 줄바꿈
                    sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                    elapsed_time = time.time() - start_time
                    print(f"✓ 질문 생성 완료 ({elapsed_time:.2f}초)")
                    return full_result
                else:
                    # 일반 모드
                    response = ollama.generate(
                        model=self.model_name,
                        prompt=prompt,
                        options={"temperature": 0.7, "top_p": 0.95, "num_predict": 1000}
                    )
                    result = response.get("response", "")
                    elapsed_time = time.time() - start_time
                    print(f"✓ 질문 생성 완료 ({elapsed_time:.2f}초)")
                    return result
            except Exception as e:
                print(f"⚠️ ollama 라이브러리 호출 실패: {e}, REST API로 시도합니다.")
        
        # REST API 사용
        try:
            if stream:
                # 스트리밍 모드
                print("\n질문 생성 응답: ", end="")
                full_result = ""
                
                response = requests.post(
                    f"{OLLAMA_API_BASE}/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": True,
                        "options": {"temperature": 0.7, "top_p": 0.95, "num_predict": 1000}
                    },
                    stream=True
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            line_text = line.decode('utf-8')
                            json_data = json.loads(line_text)
                            chunk_content = json_data.get("response", "")
                            if chunk_content:
                                full_result += chunk_content
                                print(chunk_content, end="")
                                sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                        except json.JSONDecodeError:
                            continue
                
                print()  # 줄바꿈
                sys.stdout.flush()  # 출력 버퍼 즉시 비우기
                elapsed_time = time.time() - start_time
                print(f"✓ 스트리밍 질문 생성 완료 ({elapsed_time:.2f}초)")
                return full_result
            else:
                # 일반 모드
                response = requests.post(
                    f"{OLLAMA_API_BASE}/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.7, "top_p": 0.95, "num_predict": 1000}
                    }
                )
                response.raise_for_status()
                result = response.json().get("response", "")
                elapsed_time = time.time() - start_time
                print(f"✓ 질문 생성 완료 ({elapsed_time:.2f}초)")
                return result
        except Exception as e:
            logger.error(f"❌ 자동 질문 생성 중 오류: {e}")
            print(f"❌ 자동 질문 생성 중 오류: {e}")
            return "[]"

    def generate_questions(self, context: str, stream: bool = True) -> List[str]:
        """문서 컨텍스트를 기반으로 질문 생성"""
        # 컨텍스트 길이 제한
        max_context_length = 2000
        if len(context) > max_context_length:
            context_parts = context.split("\n\n")
            optimized_context = []
            current_length = 0
            
            for part in context_parts:
                if current_length + len(part) <= max_context_length:
                    optimized_context.append(part)
                    current_length += len(part)
                else:
                    break
            
            context = "\n\n".join(optimized_context)
        
        prompt = self.question_prompt_template.format(context=context)
        response = self._generate_with_ollama(prompt, stream=stream)
        
        try:
            # JSON 추출
            json_pattern = r'\[.*\]'
            json_match = re.search(json_pattern, response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                questions = json.loads(json_str)
                return questions[:5]  # 최대 5개 질문 반환
            else:
                # JSON 형식이 아닌 경우 줄바꿈으로 구분된 질문으로 처리
                lines = [line.strip() for line in response.split('\n') 
                         if line.strip() and not line.strip().startswith('[') and not line.strip().endswith(']')]
                questions = []
                for line in lines:
                    # 번호 패턴 제거 (예: "1. ", "2) ", "질문 1: " 등)
                    clean_line = re.sub(r'^(\d+[\.\):]|\*)\s*', '', line).strip()
                    if clean_line and len(clean_line) > 10:  # 최소 길이 제한
                        questions.append(clean_line)
                return questions[:5]  # 최대 5개 질문 반환
        except json.JSONDecodeError:
            logger.error("❌ JSON 디코딩 오류")
            # 번호로 시작하는 질문 목록을 추출
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            questions = []
            for line in lines:
                # 번호 패턴 제거 (예: "1. ", "2) ", "질문 1: " 등)
                if re.match(r'^\d+[\.\):]|^\*', line):
                    clean_line = re.sub(r'^(\d+[\.\):]|\*)\s*', '', line).strip()
                    if clean_line and len(clean_line) > 10:  # 최소 길이 제한
                        questions.append(clean_line)
            return questions[:5]  # 최대 5개 질문 반환
        
        return []  # 실패 시 빈 목록 반환

class AutoTestManager:
    def __init__(self, rag_system, auto_question_generator, auto_evaluator, evaluator, available_models):
        self.rag = rag_system
        self.question_generator = auto_question_generator
        self.auto_evaluator = auto_evaluator
        self.evaluator = evaluator
        self.available_models = available_models
        self.results = {
            "tests": [],
            "summary": {}
        }
    
    def run_auto_test(self, num_questions: int = 5, top_k: int = 10, stream: bool = True):
        """자동 질문 생성 및 평가 테스트 실행"""
        print(f"\n{'='*80}")
        print(f"🚀 자동 테스트 시작 (질문 수: {num_questions})")
        print(f"{'='*80}")
        
        # 문서에서 몇 개의 페이지를 샘플링하여 컨텍스트 생성
        try:
            # 벡터 저장소에서 무작위 문서 샘플링
            docs = self.rag.vector_store.similarity_search("한화손해보험", k=20)
            
            # 전체 테스트 결과 저장용 딕셔너리
            all_test_results = []
            model_scores = {model: [] for model in self.available_models}
            
            # 각 문서 셋에 대해 테스트 실행
            for i in range(min(num_questions, 3)):  # 최대 3번의 테스트 세트 생성
                print(f"\n📚 테스트 세트 #{i+1} 준비 중...")
                
                # 샘플 문서에서 컨텍스트 생성
                sample_docs = random.sample(docs, min(5, len(docs)))
                context = "\n\n".join([f"[페이지 {doc.metadata.get('page', '불명')}] {doc.page_content}" for doc in sample_docs])
                
                # 자동 질문 생성
                print("\n📝 컨텍스트에서 질문 생성 중...")
                questions = self.question_generator.generate_questions(context, stream=stream)
                
                if not questions:
                    print("❌ 질문 생성 실패, 다음 컨텍스트로 넘어갑니다.")
                    continue
                
                print(f"\n✅ {len(questions)}개 질문 생성 완료:")
                for j, q in enumerate(questions):
                    print(f"  {j+1}. {q}")
                
                # 각 질문에 대해 RAG 시스템으로 응답 생성 및 평가
                for j, question in enumerate(questions[:min(5, len(questions))]):
                    print(f"\n{'='*50}")
                    print(f"📊 테스트 #{i+1}-{j+1}: '{question}'")
                    
                    test_results = {
                        "question": question,
                        "context_summary": context[:200] + "..." if len(context) > 200 else context,
                        "models": {}
                    }
                    
                    # 모든 모델에 대해 답변 생성 및 평가
                    for model in self.available_models:
                        print(f"\n🔄 모델 '{model}' 테스트 중...")
                        
                        # 검색 및 답변 생성
                        try:
                            sources, answer = self.rag.query(
                                question, 
                                model_name=model, 
                                top_k=top_k
                            )
                            
                            # 자동 평가 수행
                            evaluation = self.auto_evaluator.evaluate_answer(
                                question, 
                                context, 
                                answer,
                                stream=stream
                            )
                            
                            # 평가 결과 저장
                            test_results["models"][model] = {
                                "answer": answer,
                                "score": evaluation.get("score"),
                                "reason": evaluation.get("reason"),
                                "raw_evaluation": evaluation.get("raw_evaluation")
                            }
                            
                            # 모델 점수 집계
                            if evaluation.get("score") is not None:
                                model_scores[model].append(evaluation.get("score"))
                                
                            print(f"📊 평가 점수: {evaluation.get('score', '평가 불가')}/5")
                            
                        except Exception as e:
                            logger.error(f"모델 {model} 테스트 중 오류: {e}")
                            print(f"❌ 모델 {model} 테스트 중 오류: {e}")
                            test_results["models"][model] = {
                                "error": str(e)
                            }
                    
                    # 테스트 결과 저장
                    all_test_results.append(test_results)
                    
                    print(f"{'='*50}")
            
            # 요약 통계 계산
            summary = {}
            for model in self.available_models:
                scores = model_scores[model]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    summary[model] = {
                        "avg_score": round(avg_score, 2),
                        "num_evaluations": len(scores)
                    }
                else:
                    summary[model] = {
                        "avg_score": 0,
                        "num_evaluations": 0
                    }
            
            # 결과 저장
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = f"auto_test_results_{timestamp}.json"
            final_results = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_questions": sum(1 for _ in all_test_results),
                "tests": all_test_results,
                "summary": summary
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            
            # 요약 출력
            print(f"\n{'='*80}")
            print(f"📊 자동 테스트 결과 요약")
            print(f"{'='*80}")
            print(f"테스트 질문 수: {sum(1 for _ in all_test_results)}")
            
            for model, stats in summary.items():
                print(f"\n{model}:")
                if stats["num_evaluations"] > 0:
                    print(f"  - 평균 점수: {stats['avg_score']:.2f}/5.00 ({stats['num_evaluations']}개 평가)")
                else:
                    print(f"  - 평가 없음")
            
            print(f"\n결과 파일 저장됨: {results_file}")
            print(f"{'='*80}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ 자동 테스트 중 오류: {e}")
            logger.error(traceback.format_exc())
            print(f"❌ 자동 테스트 중 오류: {e}")
            print(traceback.format_exc())
            return None

def check_ollama_models():
    """
    올라마 서버에서 사용 가능한 모델을 확인합니다.
    여러 방법을 시도하고 성공 시 사용 가능한 모델 목록을 반환합니다.
    모두 실패하면 기본 모델 목록을 반환합니다.
    """
    global AVAILABLE_MODELS
    
    # 기본 모델 목록 (실패 시 사용)
    default_models = ["gemma3:1b", "gemma3:4b", "gemma3:12b"]
    supported_models = []
    
    # 방법 1: Ollama Python 클라이언트 사용 (가장 신뢰할 수 있는 방법)
    if OLLAMA_AVAILABLE:
        try:
            print("🔄 Ollama Python 라이브러리로 모델 조회 시도 중...")
            ollama_models = ollama.list()
            available_models = [model['name'] for model in ollama_models.get('models', [])]
            print(f"✓ Ollama Python 라이브러리 연결 성공: {len(available_models)}개 모델 발견")
            
            if available_models:
                print(f"✓ 설치된 모델 목록: {', '.join(available_models)}")
                
                # 우선 우리가 원하는 모델(AVAILABLE_MODELS)이 있는지 확인
                for model_name in AVAILABLE_MODELS:
                    if any(model_name in model for model in available_models):
                        supported_models.append(model_name)
                
                # 원하는 모델이 없으면 gemma 또는 llama 모델을 찾아봄
                if not supported_models:
                    for model in available_models:
                        if "gemma" in model.lower() or "llama" in model.lower():
                            supported_models.append(model)
                
                # 그래도 없으면 설치된 모든 모델 사용
                if not supported_models:
                    supported_models = available_models
                
                print(f"✓ 사용할 모델: {', '.join(supported_models)}")
                return supported_models
            else:
                print("⚠️ Python 라이브러리: 설치된 Ollama 모델이 없습니다.")
        except Exception as e:
            print(f"⚠️ Ollama Python 라이브러리 호출 실패: {e}")
    
    # 방법 2: Ollama REST API 사용
    try:
        print("🔄 Ollama REST API로 모델 조회 시도 중...")
        response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        installed_models = [model.get("name") for model in models]
        print(f"✓ Ollama REST API 연결 성공: {len(installed_models)}개 모델 발견")
        
        if installed_models:
            # 우선 우리가 원하는 모델(AVAILABLE_MODELS)이 있는지 확인
            for model_name in AVAILABLE_MODELS:
                if any(model_name in model for model in installed_models):
                    supported_models.append(model_name)
            
            # 원하는 모델이 없으면 gemma 또는 llama 모델을 찾아봄
            if not supported_models:
                for model in installed_models:
                    if "gemma" in model.lower() or "llama" in model.lower():
                        supported_models.append(model)
            
            # 그래도 없으면 설치된 모든 모델 사용
            if not supported_models:
                supported_models = installed_models
            
            print(f"✓ 사용할 모델: {', '.join(supported_models)}")
            return supported_models
        else:
            print("⚠️ REST API: 설치된 Ollama 모델이 없습니다.")
    except Exception as e:
        print(f"⚠️ Ollama REST API 연결 실패: {e}")
    
    # 방법 3: 명령줄 도구 사용
    try:
        print("🔄 명령줄 'ollama list' 실행 시도 중...")
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True, timeout=5)
        output_lines = result.stdout.strip().split('\n')
        
        # 헤더 제거 후 모델 이름만 추출 (첫 번째 컬럼)
        if len(output_lines) > 1:
            cmd_models = [line.split()[0] for line in output_lines[1:]]
            print(f"✓ 명령줄 'ollama list' 실행 성공: {len(cmd_models)}개 모델 발견")
            
            if cmd_models:
                # 우선 우리가 원하는 모델(AVAILABLE_MODELS)이 있는지 확인
                for model_name in AVAILABLE_MODELS:
                    if any(model_name in model for model in cmd_models):
                        supported_models.append(model_name)
                
                # 원하는 모델이 없으면 gemma 또는 llama 모델을 찾아봄
                if not supported_models:
                    for model in cmd_models:
                        if "gemma" in model.lower() or "llama" in model.lower():
                            supported_models.append(model)
                
                # 그래도 없으면 설치된 모든 모델 사용
                if not supported_models:
                    supported_models = cmd_models
                
                print(f"✓ 사용할 모델: {', '.join(supported_models)}")
                return supported_models
            else:
                print("⚠️ 명령줄: 설치된 Ollama 모델이 없습니다.")
        else:
            print("⚠️ 명령줄: 설치된 Ollama 모델이 없습니다.")
    except Exception as e:
        print(f"⚠️ 명령줄 'ollama list' 실행 실패: {e}")
    
    # 모든 방법 실패 시 기본 모델 사용
    print(f"⚠️ 모든 모델 조회 방법 실패. 기본 모델 사용: {', '.join(default_models)}")
    return default_models

if __name__ == "__main__":
    # 명령줄 인자 설정
    parser = argparse.ArgumentParser(description="한화손해보험 사업보고서 RAG 시스템 (Streamlit)")
    
    # 기타 설정
    parser.add_argument("--pdf", type=str, default="./[한화손해보험]사업보고서(2025.03.11).pdf", help="PDF 파일 경로")
    parser.add_argument("--embeddings", type=str, default="bge-m3", choices=["e5", "openai", "bge-m3"], help="임베딩 모델")
    parser.add_argument("--chunk-size", type=int, default=500, help="문서 청크 크기")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="문서 청크 겹침")
    parser.add_argument("--top-k", type=int, default=12, help="검색 결과 수")
    parser.add_argument("--force-update", action="store_true", help="인덱스 강제 업데이트")
    parser.add_argument("--use-multi-query", action="store_true", help="다중 쿼리 확장 사용")
    parser.add_argument("--hybrid-weight", type=float, default=0.5, help="하이브리드 검색 가중치 (0-1)")
    parser.add_argument("--auto-eval", action="store_true", help="자동 평가 활성화")
    parser.add_argument("--auto-test", action="store_true", help="자동 테스트 실행")
    parser.add_argument("--num-questions", type=int, default=5, help="자동 테스트 질문 수")

    args = parser.parse_args()
    
    # 현재 실행 환경이 Streamlit인지 확인
    if 'STREAMLIT_RUNTIME' in os.environ:
        # Streamlit 환경에서 실행 중
        streamlit_main()
    else:
        # 일반 Python 환경에서 실행 중
        try:
            main()
        except Exception as e:
            logger = setup_logging()
            logger.error(f"치명적 오류 발생: {e}")
            logger.error(traceback.format_exc())
            print(f"❌ 치명적인 오류 발생: {e}")
            print("세부 오류 내용:", traceback.format_exc())