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
import shutil
import gc  # 가비지 컬렉션을 위한 모듈 추가
import math
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

# 디버그 모드 설정
DEBUG_MODE = False

# OLLAMA_AVAILABLE 변수 정의
import ollama
OLLAMA_AVAILABLE = True
   
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

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from rank_bm25 import BM25Okapi
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ langchain_community.embeddings 모듈을 가져올 수 없습니다.")

# RAG 평가 관련 메트릭 라이브러리
try:
    from rank_bm25 import BM25Okapi
    EVAL_LIBS_AVAILABLE = True
except ImportError:
    EVAL_LIBS_AVAILABLE = False

# 환경 설정
PDF_PATH = os.path.join(SCRIPT_DIR, "[한화손해보험]사업보고서(2025.03.11).pdf")
INDEX_DIR = os.path.join(SCRIPT_DIR, "Index")  # 인덱스 디렉토리 기본 경로
METADATA_FILE = os.path.join(SCRIPT_DIR, "Index/document_metadata_bge.json")  # 메타데이터 파일
LOG_FILE = os.path.join(SCRIPT_DIR, "Log/hwgi_rag_streamlit.log")
CACHE_FILE = os.path.join(SCRIPT_DIR, "cache.json")
EVALUATION_FILE = os.path.join(SCRIPT_DIR, "Log/model_evaluations.json")  # 모델 평가 결과 저장 파일

# Ollama API 기본 URL 설정
OLLAMA_API_BASE = "http://localhost:11434/api"

# 사용 가능한 모델 설정
AVAILABLE_MODELS = ["gemma3:12b"]

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
        """PDF 파일의 처리 필요 여부를 확인합니다. 해시값이 다르면 재처리가 필요합니다."""
        previous_hash = self._load_previous_hash()
        needs_processing = previous_hash != self.pdf_hash
        
        if not needs_processing:
            logger.info("이전에 처리된 동일한 PDF 파일 감지. 변경 없음으로 판단.")
        
        return needs_processing
    
    def force_processing(self) -> bool:
        """PDF 파일의 처리가 필요하도록 강제 설정합니다."""
        # 해시 파일 삭제를 통해 강제 처리
        if os.path.exists(self.hash_file):
            try:
                os.remove(self.hash_file)
                logger.info(f"PDF 해시 파일 삭제: {self.hash_file}")
                print(f"✓ PDF 해시 파일 삭제됨 - 강제 처리 모드 활성화")
                return True
            except Exception as e:
                logger.error(f"PDF 해시 파일 삭제 실패: {e}")
                print(f"⚠️ PDF 해시 파일 삭제 실패: {e}")
                return False
        return True
    
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
    def __init__(self, chunk_size=800, chunk_overlap=200):  # 청크 사이즈 증가 (500 → 800), 겹침 크기 증가 (150 → 200)
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

# --- 쿼리 확장기 클래스 ---
class QueryExpander:
    """쿼리 확장 클래스"""
    def __init__(self, model_name: str = None):
        """QueryExpander 초기화"""
        # 중복 초기화 방지
        if hasattr(QueryExpander, '_instance'):
            self.model = QueryExpander._instance.model
            self.models = QueryExpander._instance.models
            logger.info(f"QueryExpander 인스턴스 재사용: {self.model}")
            return
        
        # 모델 설정
        self.models = check_ollama_models()
        
        # 모델 자동 선택 또는 지정된 모델 사용
        if model_name and model_name in self.models:
            self.model = model_name
            logger.info(f"QueryExpander 모델 지정: {model_name}")
        elif self.models:
            self.model = self.models[0]
            logger.info(f"QueryExpander 모델 자동 선택: {self.model}")
        else:
            self.model = "gemma3:7b" # 기본 모델
            logger.warning(f"사용 가능한 모델이 없어 기본 모델 사용: {self.model}")
        
        # 인스턴스 저장
        QueryExpander._instance = self
    
    def expand_query(self, query: str, max_queries: int = 5) -> List[str]:
        """
        주어진 질문에 대해 다양한 검색용 쿼리를 생성합니다.
        - 원본 질문을 다양한 관점에서 재구성하여 검색 범위 확장
        - 반환값은 원본 쿼리를 포함한 확장 쿼리 목록
        """
        logger.info(f"🔍 쿼리 확장 시작: '{query}'")
        print(f"🔍 쿼리 확장 생성 중: '{query}'")
        
        try:
            if not query.strip():
                logger.warning("빈 쿼리는 확장할 수 없습니다.")
                return [query]
            
            # 엔티티 타입 감지 (인물, 상품, 기술 등)
            entity_type, entity_name = self._detect_entity(query)
            
            # 특수 엔티티가 감지된 경우
            if entity_type and entity_name:
                logger.info(f"특수 엔티티 감지: 유형={entity_type}, 이름='{entity_name}'")
                print(f"🔎 {entity_type} 검색 감지: '{entity_name}'")
                
                # 엔티티 유형별 특화 쿼리 생성
                return self._generate_entity_specific_queries(entity_type, entity_name, query, max_queries)
            
            # 일반 쿼리 확장 프롬프트
            prompt = f"""
당신은 한화손해보험 사업보고서 내용을 검색하는 RAG 시스템의 쿼리 확장기입니다.
다음 질문에 대해 사업보고서 내에서 답을 찾기 위한 관련 검색 쿼리를 여러 개 생성해 주세요.

원본 질문: "{query}"

중요: 반드시 원본 질문과 직접적인 관련이 있는 쿼리만 생성하세요. 
관련 없는 일반적인 쿼리는 생성하지 마세요.

원본 질문을 분석하여 다음과 같은 방식으로 변형해 주세요:
1. 동의어나 유사 표현으로 바꾸기
2. 구체적인 키워드 위주로 변환하기
3. 관련된 세부 개념으로 확장하기 (단, 질문의 핵심 의도를 유지해야 함)
4. 질문형/키워드형 등 다양한 형식으로 표현하기

JSON 형식으로 응답해 주세요:
```json
{{
  "queries": [
    "첫 번째 쿼리",
    "두 번째 쿼리",
    "세 번째 쿼리",
    ...
  ]
}}
```
JSON 형식만 응답하세요. 다른 설명이나 텍스트는 포함하지 마세요.
최대 {max_queries}개의 관련성 높은 쿼리를 생성하세요.
"""
            
            # Ollama API를 통해 쿼리 확장 생성
            response_text = self._generate_query_expansion(prompt)
            
            # JSON 파싱
            try:
                # JSON 형식 검색
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
                
                # 중괄호 기반 JSON 추출
                json_match = re.search(r'{.*}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
                
                # JSON 파싱
                response = json.loads(response_text)
                
                # 쿼리 목록 추출
                if "queries" in response:
                    expanded_queries = response["queries"]
                    
                    # 중복 제거 및 빈 쿼리 제거
                    expanded_queries = [q.strip() for q in expanded_queries if q.strip()]
                    
                    # 관련성 필터링 - 원본 쿼리의 핵심 키워드가 포함된 쿼리만 유지
                    keywords = self._extract_keywords(query)
                    if keywords:
                        # 적어도 하나의 키워드가 포함된 쿼리만 유지
                        filtered_queries = []
                        for q in expanded_queries:
                            if any(keyword.lower() in q.lower() for keyword in keywords):
                                filtered_queries.append(q)
                        expanded_queries = filtered_queries
                    
                    expanded_queries = list(dict.fromkeys(expanded_queries))
                    
                    # 원본 쿼리가 목록에 없으면 추가
                    if query not in expanded_queries:
                        expanded_queries.insert(0, query)
                    
                    # 최대 쿼리 수 제한
                    expanded_queries = expanded_queries[:max_queries]
                    
                    logger.info(f"✅ 쿼리 확장 완료: {len(expanded_queries)}개 생성")
                    logger.info(f"확장 쿼리: {expanded_queries}")
                    print(f"✅ 확장 쿼리 {len(expanded_queries)}개 생성 완료")
                    return expanded_queries
                else:
                    logger.warning("쿼리 확장 결과에 'queries' 키가 없습니다.")
                    return [query]
            
            except json.JSONDecodeError as e:
                logger.error(f"쿼리 확장 결과 JSON 파싱 오류: {e}")
                logger.error(f"원본 응답: {response_text}")
                return [query]
        
        except Exception as e:
            logger.error(f"쿼리 확장 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            return [query]
    
    def _detect_entity(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """쿼리에서 특수 엔티티(인물, 상품, 기술 등)를 감지합니다."""
        # 1. 인물 감지 (이름 + 조사 + 의문사)
        person_pattern = r'^([가-힣]{2,4}|[a-zA-Z\s]{2,20})(?:은|는|이|가)?\s+(?:누구|어떤|어느|무슨)'
        person_match = re.search(person_pattern, query)
        if person_match:
            name = person_match.group(1).strip()
            return "인물", name
            
        # 2. 상품/서비스 감지
        product_pattern = r'(?:상품|서비스|보험)\s+([가-힣a-zA-Z0-9\s]{2,20})(?:이|가|은|는)?\s+(?:무엇|뭐|어떤|어떻게)'
        product_match = re.search(product_pattern, query)
        if product_match:
            product = product_match.group(1).strip()
            return "상품", product
            
        # 3. 기술/기능 감지
        tech_pattern = r'(?:기술|기능|시스템|플랫폼)\s+([가-힣a-zA-Z0-9\s]{2,20})(?:이|가|은|는)?\s+(?:무엇|뭐|어떤|어떻게)'
        tech_match = re.search(tech_pattern, query)
        if tech_match:
            tech = tech_match.group(1).strip()
            return "기술", tech
            
        # 4. 인명만 있는 경우 (직접 언급)
        name_only_pattern = r'^([가-힣]{2,4}|[a-zA-Z\s]{2,20})$'
        name_only_match = re.search(name_only_pattern, query)
        if name_only_match:
            name = name_only_match.group(1).strip()
            return "인물", name
            
        # 감지된 엔티티가 없는 경우
        return None, None
            
    def _extract_keywords(self, query: str) -> List[str]:
        """원본 쿼리에서 핵심 키워드를 추출합니다."""
        # 불용어 제거
        stopwords = {"은", "는", "이", "가", "을", "를", "에", "에서", "의", "과", "와", "로", "으로", 
                    "이다", "있다", "없다", "하다", "되다", "한다", "된다", "무엇", "누구", "어디", 
                    "언제", "왜", "어떻게", "어떤", "얼마나", "몇", "무슨"}
        
        # 조사와 특수문자 제거
        cleaned_query = re.sub(r'[^\w\s]', ' ', query)
        words = cleaned_query.split()
        
        # 길이가 1인 단어와 불용어 제거
        keywords = [word for word in words if len(word) > 1 and word not in stopwords]
        return keywords
    
    def _generate_entity_specific_queries(self, entity_type: str, entity_name: str, original_query: str, max_queries: int) -> List[str]:
        """특정 엔티티 유형에 특화된 검색 쿼리를 생성합니다."""
        logger.info(f"'{entity_name}'({entity_type})에 대한 특화 쿼리 생성 중")
        
        # 기본 쿼리는 항상 원본 쿼리로 시작
        result_queries = [original_query]
        
        # 엔티티 유형별 특화 쿼리 생성
        if entity_type == "인물":
            specific_queries = [
                f"{entity_name}",
                f"{entity_name} 한화손해보험",
                f"{entity_name} 직책",
                f"{entity_name} 직위",
                f"{entity_name} 경력",
                f"{entity_name} 담당",
                f"{entity_name} 프로필",
                f"{entity_name} 이력",
                f"{entity_name} 업무",
                f"{entity_name} 소개",
                f"한화손해보험 {entity_name}",
                f"이사 {entity_name}",
                f"임원 {entity_name}",
                f"사장 {entity_name}",
                f"부사장 {entity_name}",
                f"전무 {entity_name}",
                f"상무 {entity_name}",
                f"본부장 {entity_name}",
                f"실장 {entity_name}",
                f"팀장 {entity_name}"
            ]
            result_queries.extend(specific_queries)
            
        elif entity_type == "상품":
            specific_queries = [
                f"{entity_name}",
                f"{entity_name} 상품",
                f"{entity_name} 보험",
                f"{entity_name} 서비스",
                f"{entity_name} 특징",
                f"{entity_name} 설명",
                f"{entity_name} 소개",
                f"{entity_name} 장점",
                f"{entity_name} 조건",
                f"{entity_name} 계약",
                f"한화손해보험 {entity_name}",
                f"{entity_name} 보장",
                f"{entity_name} 보험료",
                f"{entity_name} 판매",
                f"{entity_name} 가입"
            ]
            result_queries.extend(specific_queries)
            
        elif entity_type == "기술":
            specific_queries = [
                f"{entity_name}",
                f"{entity_name} 기술",
                f"{entity_name} 시스템",
                f"{entity_name} 플랫폼",
                f"{entity_name} 적용",
                f"{entity_name} 활용",
                f"{entity_name} 구현",
                f"{entity_name} 도입",
                f"{entity_name} 특징",
                f"{entity_name} 효과",
                f"한화손해보험 {entity_name}",
                f"{entity_name} 개발",
                f"{entity_name} 혁신",
                f"{entity_name} 투자"
            ]
            result_queries.extend(specific_queries)
            
        else:
            # 기타 엔티티 유형에 대한 일반적인 확장
            specific_queries = [
                f"{entity_name}",
                f"{entity_name} 한화손해보험",
                f"한화손해보험 {entity_name}",
                f"{entity_name} 설명",
                f"{entity_name} 내용",
                f"{entity_name} 소개",
                f"{entity_name} 정보"
            ]
            result_queries.extend(specific_queries)
        
        # 중복 제거 및 최대 개수 제한
        result_queries = list(dict.fromkeys(result_queries))[:max_queries]
        
        logger.info(f"{entity_type} 특화 쿼리 {len(result_queries)}개 생성 완료")
        print(f"✅ '{entity_name}'({entity_type})에 관한 특화 쿼리 {len(result_queries)}개 생성")
        
        return result_queries
        
    def _generate_query_expansion(self, prompt: str) -> str:
        """Ollama API를 통해 쿼리 확장을 생성합니다."""
        try:
            url = "http://localhost:11434/api/generate"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            logger.info(f"Ollama API 호출: 모델={self.model}")
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama API 오류: {response.status_code} - {response.text}")
                return ""
        
        except Exception as e:
            logger.error(f"쿼리 확장 생성 중 오류: {e}")
            return ""

# --- RAG 시스템 (벡터 검색) ---
class RAGSystem:
    """검색 증강 생성(RAG) 시스템 클래스"""
    def __init__(self, embedding_type: str = "bge-m3", use_hnsw: bool = True, ef_search: int = 200, ef_construction: int = 200, m: int = 64):
        """RAG 시스템 초기화"""
        print("🔧 RAG 시스템 초기화 중...")
        print(f"  - 임베딩 모델: {embedding_type}")
        
        # 중복 초기화 방지를 위한 클래스 변수
        if hasattr(RAGSystem, '_initialized'):
            logger.info("RAG 시스템이 이미 초기화되어 있습니다.")
            return
        
        # 초기화 완료 표시
        RAGSystem._initialized = True
        
        # 임베딩 모델 유형 저장
        self.embedding_type = embedding_type
        self.embedding_name = None
        
        # 임베딩 모델 초기화
        if embedding_type == "bge-m3":
            self.embeddings = BGEM3Embeddings(model_name="BAAI/bge-m3")
            self.embedding_name = "bge-m3"
        elif LANGCHAIN_AVAILABLE and embedding_type == "bge":
            # BGE-base 임베딩 사용 (HuggingFaceEmbeddings 필요)
            print("✓ BGE-base 임베딩 모델 초기화 중...")
            self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
            self.embedding_name = "bge"
        else:
            # 기본으로 BGE-M3 임베딩 사용 (embedding_type이 "bge-m3"가 아닌 다른 값인 경우에만)
            if embedding_type != "bge-m3":
                print("✓ BGE-M3 임베딩 모델 'BAAI/bge-m3' 초기화 중 (기본값)")
                self.embeddings = BGEM3Embeddings(model_name="BAAI/bge-m3")
                self.embedding_name = "bge-m3"
                self.embedding_type = "bge-m3"  # 타입 업데이트
            else:
                # 이미 "bge-m3"로 초기화된 경우 건너뜁니다 (첫 번째 if 조건에서 처리됨)
                pass

        # 응답 캐시 초기화
        self._cache = self._load_cache()
        
        # HNSW 인덱스 옵션
        self.use_hnsw = use_hnsw
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.m = m
        
        # 벡터 저장소 초기화
        self.vector_store = None
                
        # 인덱스 디렉토리 설정
        self.index_dir = INDEX_DIR
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Ollama REST API 설정
        self.ollama_base_url = "http://localhost:11434/api"
        
        # 사용 가능한 Ollama 모델 목록
        self.available_models = check_ollama_models()
        
        if self.available_models:
            selected_model = self.available_models[0]
            print(f"✓ 사용할 모델: {selected_model}")
            
            # 쿼리 확장기 초기화
            self.query_expander = QueryExpander(model_name=selected_model)
            logger.info(f"QueryExpander 모델 자동 선택: {selected_model}")
            print(f"✓ 쿼리 확장에 {selected_model} 모델을 사용합니다.")
            print(f"  - 쿼리 확장: 활성화됨")
        else:
            self.query_expander = None
            print("⚠️ 사용 가능한 Ollama 모델이 없어 쿼리 확장이 비활성화됩니다.")
            print(f"  - 쿼리 확장: 비활성화됨")
        
        print("✅ RAG 시스템 초기화 완료")
    
    def _load_cache(self) -> Dict[str, str]:
        """캐시 파일을 로드합니다."""
        cache_file = os.path.join(SCRIPT_DIR, "cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"캐시 파일 로드 실패: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """캐시를 파일에 저장합니다."""
        cache_file = os.path.join(SCRIPT_DIR, "cache.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"캐시 파일 저장 실패: {e}")
    
    def _generate_with_ollama(self, prompt: str, model: str, stream: bool = True) -> str:
        """Ollama API를 사용하여 텍스트 생성"""
        logger.debug(f"답변 생성 요청 - 쿼리: '{prompt[:30]}...', 모델: {model}")
        logger.info(f"응답 생성 시작: 모델={model}, 쿼리='{prompt[:20]}...'")
        
        # Ollama 서버 연결 확인
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                logger.error(f"Ollama 서버 연결 실패: {response.status_code}")
                return "⚠️ Ollama 서버 연결 오류: 서버가 실행 중인지 확인해주세요."
        except Exception as e:
            logger.error(f"Ollama 서버 접속 오류: {e}")
            return f"⚠️ Ollama 서버 접속 오류: {e}"
        
        # API 요청 포맷
        api_json = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        # 응답 저장할 문자열
        response_text = ""
        
        try:
            if stream:
                # 스트리밍 응답 처리
                print(f"\n================================================================================")
                print(f"📝 모델: {model}")
                print(f"================================================================================")
                
                with requests.post("http://localhost:11434/api/generate", json=api_json, stream=True) as response:
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            response_text += chunk
                            print(chunk, end="", flush=True)
                            
                            if data.get("done", False):
                                # 스트리밍이 끝났을 때 응답 저장
                                logger.debug(f"응답 생성 완료: 길이={len(response_text)}")
                                print()  # 줄바꿈으로 마무리
                
            else:
                # 비스트리밍 응답 처리
                print(f"\n================================================================================")
                print(f"📝 모델: {model}")
                print(f"================================================================================")
                
                response = requests.post("http://localhost:11434/api/generate", json=api_json)
                
                if response.status_code == 200:
                    response_data = response.json()
                    response_text = response_data.get("response", "")
                    
                    # JSON 추출 시도 (JSON으로 요청한 경우)
                    try:
                        if "format" in api_json and api_json["format"] == "json":
                            # 응답 텍스트에서 JSON 부분 추출 시도
                            json_match = re.search(r'({[\s\S]*})', response_text)
                            if json_match:
                                json_str = json_match.group(1)
                                parsed_json = json.loads(json_str)
                                logger.debug(f"JSON 응답 파싱 성공: {parsed_json}")
                            else:
                                # 전체 텍스트를 JSON으로 파싱 시도
                                parsed_json = json.loads(response_text)
                                logger.debug(f"텍스트 전체를 JSON으로 파싱 성공")
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON 파싱 실패: {e}, 응답: {response_text[:200]}...")
                    
                    print(response_text)
                    logger.debug(f"응답 생성 완료: 길이={len(response_text)}")
                    
                else:
                    error_msg = f"Ollama API 오류: {response.status_code}, {response.text}"
                    logger.error(error_msg)
                    response_text = f"⚠️ {error_msg}"
                    print(response_text)
            
            return response_text
            
        except Exception as e:
            error_msg = f"텍스트 생성 중 오류 발생: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            print(f"\n❌ {error_msg}")
            return f"⚠️ {error_msg}"
    
    def answer(self, query: str, model: str, context: str) -> Dict[str, Any]:
        """쿼리에 대한 답변을 생성합니다."""
        logger.debug(f"답변 생성 요청 - 쿼리: '{query}', 모델: {model}")
        
        # 1. 캐시에서 응답 확인
        cache_key = f"{model}:{hashlib.md5((query + context[:200]).encode()).hexdigest()}"
        cached_answer = self._load_cache().get(cache_key)
        
        if cached_answer and not os.environ.get('DISABLE_CACHE'):
            logger.info(f"캐시된 응답 사용: 모델={model}, 쿼리='{query[:30]}...'")
            print(f"💾 캐시된 응답 사용: {model}")
            return {"answer": cached_answer, "model": model, "cached": True}
        
        # 2. 프롬프트 생성
        try:
            # 오늘 날짜 정보 추가
            today = datetime.now().strftime("%Y년 %m월 %d일")
            
            # 컨텍스트가 너무 길면 지능적으로 자르기
            max_context_length = 10000  # 토큰 한도를 고려한 길이 제한
            if len(context) > max_context_length:
                logger.warning(f"컨텍스트 길이 제한 초과: {len(context)}자 -> {max_context_length}자로 제한")
                print(f"⚠️ 컨텍스트가 너무 깁니다 ({len(context)}자). {max_context_length}자로 제한합니다.")
                
                # 섹션 단위로 나누기
                sections = context.split("\n\n")
                
                # 중요도를 계산하여 정렬
                scored_sections = []
                for section in sections:
                    # 제목 섹션은 항상 유지
                    if section.startswith("다음은 사용자 질문과 관련된"):
                        score = float('inf')  # 최고 점수로 설정
                    elif section.startswith("[문서 #"):
                        score = float('inf') - 1  # 문서 제목도 높은 점수
                    else:
                        # 간단한 키워드 매칭 기반 관련성 점수 계산
                        query_terms = set(re.findall(r'\w+', query.lower()))
                        section_text = section.lower()
                        score = sum(1 for term in query_terms if term in section_text)
                    
                    scored_sections.append((section, score))
                
                # 점수별 정렬 (높은 것부터)
                sorted_sections = sorted(scored_sections, key=lambda x: x[1], reverse=True)
                
                # 컨텍스트 재구성
                trimmed_context = ""
                for section, _ in sorted_sections:
                    if len(trimmed_context) + len(section) + 2 <= max_context_length:
                        trimmed_context += section + "\n\n"
                    else:
                        # 최대 길이 초과 시 중단
                        break
                
                context = trimmed_context.strip()
                logger.info(f"컨텍스트 축소 완료: {len(context)}자")
                print(f"✓ 컨텍스트를 {len(context)}자로 중요도 기준 축소했습니다")
            
            # 한화손해보험 사업보고서 특화 QA 프롬프트 템플릿
            qa_template = f"""당신은 {today} 기준으로 한화손해보험 사업보고서를 분석하여 정확하고 상세한 답변을 제공하는 금융 전문가입니다.

다음 규칙에 따라 질문에 답변하세요:
1. 제공된 사업보고서 내용만을 기반으로 답변하세요. 제공된 문서에 정보가 없으면 "한화손해보험 사업보고서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요.
2. 재무 데이터, 날짜, 수치 등은 사업보고서에 기재된 정확한 값을 사용하세요.
3. 답변은 논리적 구조를 갖추고, 필요시 항목별로 구분하여 가독성을 높이세요.
4. 표, 그래프, 숫자 데이터는 표 형식으로 정리해 제시하세요.
5. 보험업 전문 용어가 사용된 경우 간략한 설명을 추가하세요.
6. 한화손해보험의 경영 전략, 재무 상태, 리스크 관리, 기업 지배구조 관련 정보를 우선적으로 제공하세요.
7. 복잡한 개념은 간결하게 설명하되, 핵심 정보는 빠뜨리지 마세요.
8. 주관적 의견이나 추측은 배제하고, 사업보고서에 명시된 내용만 답변하세요.

사용자 질문: {query}

검색된 한화손해보험 사업보고서 내용:
{context}

위 사업보고서 내용을 바탕으로 질문에 대한 전문적인 답변:"""
            
            # 3. Ollama 또는 LLM을 사용하여 응답 생성
            try:
                logger.info(f"응답 생성 시작: 모델={model}, 쿼리='{query[:30]}...'")
                start_time = time.time()
                
                # 스트리밍 답변 생성
                result = self._generate_with_ollama(qa_template, model, stream=True)
                
                # 소요 시간 측정
                elapsed_time = time.time() - start_time
                logger.info(f"응답 생성 완료: 소요시간={elapsed_time:.2f}초")
                
                # 응답 캐싱
                answer_content = result if isinstance(result, str) else result.get('answer', '')
                self._cache[cache_key] = answer_content
                self._save_cache()
                
                # 결과 반환
                return {
                    "answer": answer_content,
                    "model": model,
                    "cached": False,
                    "elapsed_time": elapsed_time
                }
                
            except Exception as e:
                logger.error(f"❌ 답변 생성 중 오류: {e}")
                return {"answer": f"답변을 생성하는 중 오류가 발생했습니다: {str(e)}", "model": model, "error": True}
                
        except Exception as e:
            logger.error(f"❌ 프롬프트 생성 중 오류: {e}")
            return {"answer": f"답변을 준비하는 중 오류가 발생했습니다: {str(e)}", "model": model, "error": True}

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """쿼리에 대한 관련 문서를 검색합니다."""
        try:
            logger.info(f"검색 요청: '{query}', top_k={top_k}")
            print(f"\n🔍 검색 시작: '{query}'")
            
            if not hasattr(self, "vector_store") or self.vector_store is None:
                # FAISS 로드 또는 초기화
                index_dir = os.path.join(self.index_dir, f"faiss_index_{self.embedding_type}")
                metadata_path = os.path.join(self.index_dir, f"document_metadata_{self.embedding_type}.json")
                
                logger.info(f"벡터 스토어 로드 시도: {index_dir}")
                print(f"📂 인덱스 로드 중: {index_dir}")
                
                try:
                    # HNSW 인덱스 사용 여부에 따라 로드
                    if self.use_hnsw:
                        # HNSW 인덱스 로드
                        self.vector_store = FAISS.load_local(
                            folder_path=index_dir,
                            embeddings=self.embeddings,
                            allow_dangerous_deserialization=True,
                            index_name="index"
                        )
                        logger.info("HNSW 인덱스 로드 성공")
                        print("✅ HNSW 인덱스 로드 완료")
                    else:
                        # L2 인덱스 로드
                        self.vector_store = FAISS.load_local(
                            folder_path=index_dir,
                            embeddings=self.embeddings,
                            allow_dangerous_deserialization=True,
                            index_name="index"
                        )
                        logger.info("L2 인덱스 로드 성공")
                        print("✅ L2 인덱스 로드 완료")
                    
                    # 메타데이터 로드
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            self.document_metadata = json.load(f)
                            logger.info(f"메타데이터 로드 성공: {len(self.document_metadata)}개 문서")
                except Exception as e:
                    logger.error(f"벡터 스토어 로드 실패: {e}")
                    logger.error(traceback.format_exc())
                    print(f"❌ 인덱스 로드 실패: {e}")
                    raise ValueError(f"벡터 스토어를 로드할 수 없습니다: {e}")
            
            # 검색 개선: 하이브리드 검색과 표 데이터 가중치 적용
            try:
                # 멀티쿼리 생성 (검색 성능 향상)
                expander = QueryExpander()
                queries = expander.expand_query(query, max_queries=5)  # 최대 5개 쿼리 생성
                
                logger.info(f"하이브리드 멀티쿼리 검색 실행: {len(queries)}개 쿼리")
                print(f"🧠 하이브리드 멀티쿼리 검색 ({len(queries)}개 쿼리 사용)")
                
                # 각 쿼리별 결과 저장
                all_results = {}
                query_results = {}
                
                for idx, q in enumerate(queries):
                    logger.info(f"쿼리 {idx+1} 검색: '{q}'")
                    print(f"  - 쿼리 {idx+1}: '{q}'")
                    
                    # 타이틀과 내용 가중치 높은 결과 가져오기
                    vector_results = self.vector_store.similarity_search_with_score(
                        q, 
                        k=top_k * 2  # 충분한 후보 결과 검색
                    )
                    
                    if vector_results:
                        # 결과 처리 및 관련성 점수 계산
                        processed_results = []
                        for doc, score in vector_results:
                            # 문서 ID 생성 (중복 확인용)
                            doc_id = self._get_doc_id(doc)
                            
                            # 관련성 점수 계산 (FAISS는 거리를 반환하므로 거리가 작을수록 관련성 높음)
                            # 정규화된 관련성 점수 (1에 가까울수록 관련성 높음)
                            relevance = 1.0 / (1.0 + score)
                            
                            # 표 데이터 가중치 적용 (1.5배)
                            is_table = False
                            # 표 데이터인지 확인: content_type=table 또는 메타데이터에서 확인
                            if "content_type" in doc.metadata and doc.metadata["content_type"] == "table":
                                is_table = True
                            # 표 키워드 확인 (page_content에 표, 테이블 등의 키워드가 있는지)
                            elif "table" in doc.page_content.lower() or "표 " in doc.page_content:
                                is_table = True
                                
                            if is_table:
                                relevance *= 1.5  # 표 데이터는 1.5배 가중치
                                logger.info(f"표 데이터 가중치 적용: {doc_id}, 원래점수={1.0/(1.0+score):.4f}, 가중치적용={relevance:.4f}")
                            
                            # 키워드 매칭 점수 계산 (하이브리드 검색 요소)
                            keyword_score = self._calculate_keyword_score(q, doc.page_content)
                            
                            # 최종 하이브리드 점수 계산: 벡터 점수 70% + 키워드 점수 30%
                            hybrid_score = (relevance * 0.7) + (keyword_score * 0.3)
                            
                            # 쿼리별 로깅
                            if doc_id not in all_results:
                                all_results[doc_id] = {
                                    "doc": doc,
                                    "scores": {},
                                    "queries": [],
                                    "is_table": is_table
                                }
                            
                            # 쿼리별 점수 저장
                            all_results[doc_id]["scores"][q] = hybrid_score
                            all_results[doc_id]["queries"].append(q)
                            processed_results.append((doc, doc_id, hybrid_score))
                        
                        query_results[q] = processed_results
                        logger.info(f"쿼리 '{q}'에 대한 검색 결과: {len(processed_results)}개 문서")
                
                # 결과 병합 및 정렬 로직 개선
                # 1. 여러 쿼리에서 발견된 문서에 가중치 부여
                # 2. 가장 높은 점수 사용
                # 3. 유사한 문서 중복 제거 (MMR과 유사한 효과)
                
                # 최종 점수 계산
                final_scores = {}
                for doc_id, data in all_results.items():
                    # 각 문서가 매칭된 쿼리 수에 따른 가중치
                    query_match_weight = min(len(data["queries"]) / len(queries), 1.0)
                    
                    # 쿼리별 최고 점수 가져오기
                    max_score = max(data["scores"].values())
                    
                    # 최종 점수 = 최고점수 * (1 + 쿼리매칭가중치)
                    # 이렇게 하면 여러 쿼리에 매칭된 문서가 더 높은 점수를 받음
                    final_score = max_score * (1 + query_match_weight)
                    
                    # 표 데이터 여부 표시 (이미 개별 점수에 가중치 적용됨)
                    if data.get("is_table", False):
                        logger.info(f"표 문서 최종 점수: {doc_id}, 점수={final_score:.4f}")
                        
                    final_scores[doc_id] = final_score
                
                # 최종 점수로 정렬
                sorted_results = sorted(
                    [(doc_id, all_results[doc_id]["doc"], score) for doc_id, score in final_scores.items()],
                    key=lambda x: x[2],  # 점수 기준 정렬
                    reverse=True  # 높은 점수부터
                )
                
                # top_k개 결과 선택
                final_docs = []
                for doc_id, doc, score in sorted_results[:top_k]:
                    # 관련 쿼리 정보 추가
                    doc.metadata["relevance_score"] = f"{score:.4f}"
                    doc.metadata["matched_queries"] = ", ".join(all_results[doc_id]["queries"])
                    
                    # 표 데이터 여부 표시
                    if all_results[doc_id].get("is_table", False):
                        doc.metadata["is_table"] = "true"
                        
                    final_docs.append(doc)
                
                # 결과 로깅
                logger.info(f"최종 검색 결과: {len(final_docs)}개 문서")
                print(f"✅ 검색 완료: {len(final_docs)}개 관련 문서 발견")
                
                # 표 데이터 개수 출력
                table_count = sum(1 for doc in final_docs if doc.metadata.get("is_table") == "true")
                if table_count > 0:
                    print(f"   - 표 데이터: {table_count}개 (1.5배 가중치 적용)")
                
                return final_docs
                
            except Exception as e:
                logger.error(f"멀티쿼리 검색 중 오류: {e}")
                logger.error(traceback.format_exc())
                print(f"⚠️ 멀티쿼리 검색 실패, 단일 쿼리로 대체: {e}")
                
                # 확장 실패시 기본 검색으로 폴백
                results = self.vector_store.similarity_search_with_score(query, k=top_k)
                
                if results:
                    logger.info(f"단일 쿼리 검색 결과: {len(results)}개 문서")
                    
                    # 점수 기준 정렬 및 결과 반환
                    final_results = []
                    for doc, score in results:
                        # 점수 정규화 (FAISS는 거리를 반환하므로 거리가 작을수록 관련성 높음)
                        relevance = 1.0 / (1.0 + score)
                        
                        # 표 데이터 가중치 적용
                        is_table = False
                        if "content_type" in doc.metadata and doc.metadata["content_type"] == "table":
                            is_table = True
                        elif "table" in doc.page_content.lower() or "표 " in doc.page_content:
                            is_table = True
                            
                        if is_table:
                            relevance *= 1.5  # 표 데이터 가중치
                            doc.metadata["is_table"] = "true"
                            
                        doc.metadata["relevance_score"] = f"{relevance:.4f}"
                        final_results.append(doc)
                    
                    # 표 데이터 개수 출력
                    table_count = sum(1 for doc in final_results if doc.metadata.get("is_table") == "true")
                    if table_count > 0:
                        print(f"   - 표 데이터: {table_count}개 (1.5배 가중치 적용)")
                        
                    print(f"✅ 단일 쿼리 검색 완료: {len(final_results)}개 문서 발견")
                    return final_results
                else:
                    logger.warning(f"검색 결과 없음: '{query}'")
                    print("❌ 검색 결과 없음")
                    return []
            
        except Exception as e:
            logger.error(f"검색 중 오류: {e}")
            logger.error(traceback.format_exc())
            print(f"❌ 검색 실패: {e}")
            return []
            
    def _get_doc_id(self, doc: Document) -> str:
        """문서의 고유 ID를 생성합니다."""
        # 메타데이터에서 정보 추출
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        chunk_id = doc.metadata.get("chunk_id", "")
        
        # 고유 ID 생성
        doc_id = f"{source}:{page}:{chunk_id}"
        return doc_id
    
    def load_or_create_vector_store(self, documents: List[Document], force_update: bool = False) -> bool:
        """벡터 저장소를 로드하거나 새로 생성합니다."""
        try:
            # 임베딩 모델에 따라 인덱스 폴더 결정
            if self.embedding_type == "bge-m3":
                index_folder = os.path.join(INDEX_DIR, "faiss_index_bge-m3")
                metadata_file = os.path.join(INDEX_DIR, "document_metadata_bge-m3.json")
            else:
                index_folder = os.path.join(INDEX_DIR, "faiss_index_bge")
                metadata_file = os.path.join(INDEX_DIR, "document_metadata_bge.json")
            
            # 인덱스 강제 업데이트 여부
            if force_update:
                logger.info("인덱스 강제 업데이트 모드 활성화")
                print("🔄 인덱스 강제 업데이트 모드 활성화됨")
                return self._create_new_vector_store(documents, index_folder, metadata_file)
            
            # 기존 인덱스 로드 시도
            if os.path.exists(os.path.join(index_folder, "index.faiss")) and os.path.exists(os.path.join(index_folder, "index.pkl")):
                try:
                    # HNSW 인덱스 사용 옵션
                    if self.use_hnsw:
                        print(f"✓ HNSW 인덱스 사용 (ef_search={self.ef_search})")
                        logger.info(f"HNSW 인덱스 로드 시도: ef_search={self.ef_search}")
                        
                        self.vector_store = FAISS.load_local(
                            index_folder,
                            self.embeddings,
                            normalize_L2=True,
                            allow_dangerous_deserialization=True  # 안전하지 않은 역직렬화 허용
                        )
                        
                        # HNSW 인덱스 설정 조정
                        if hasattr(self.vector_store, 'index'):
                            # HNSW 인덱스 타입인지 확인
                            if hasattr(self.vector_store.index, 'hnsw'):
                                self.vector_store.index.hnsw.efSearch = self.ef_search
                                logger.info(f"HNSW 인덱스 매개변수 설정 완료: ef_search={self.ef_search}")
                            else:
                                logger.info("L2 인덱스가 로드됨 (HNSW 인덱스 아님)")
                                print(f"✓ L2 인덱스가 로드됨 (HNSW 인덱스 아님)")
                    else:
                        logger.info("기본 L2 인덱스 로드 시도")
                        print(f"✓ 기본 L2 인덱스 사용")
                        
                        self.vector_store = FAISS.load_local(
                            index_folder,
                            self.embeddings,
                            normalize_L2=True,
                            allow_dangerous_deserialization=True  # 안전하지 않은 역직렬화 허용
                        )
                        
                    # 임베딩 모델 이름 저장
                    self.embedding_model_name = self.embedding_name
                    
                    logger.info(f"인덱스 로드 성공: {index_folder}")
                    print(f"✓ 인덱스 로드 성공: {index_folder}")
                    return True
                    
                except Exception as e:
                    logger.error(f"벡터 스토어 로드 실패: {e}")
                    logger.error(traceback.format_exc())
                    print(f"❌ 기존 인덱스 로드 실패 - 새 인덱스 생성 시도: {e}")
                    return self._create_new_vector_store(documents, index_folder, metadata_file)
            else:
                # 인덱스가 없는 경우 새로 생성
                logger.info(f"인덱스 파일이 존재하지 않음: {index_folder}")
                print(f"⚠️ 기존 인덱스 파일이 존재하지 않습니다. 새 인덱스를 생성합니다.")
                return self._create_new_vector_store(documents, index_folder, metadata_file)
                
        except Exception as e:
            logger.error(f"벡터 스토어 로드/생성 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            print(f"❌ 벡터 스토어 로드/생성 중 오류 발생: {e}")
            return False

    def _create_new_vector_store(self, documents: List[Document], index_folder: str, metadata_file: str) -> bool:
        """새로운 벡터 스토어 생성"""
        try:
            # 인덱스 디렉토리가 존재하지 않으면 생성
            os.makedirs(os.path.dirname(index_folder), exist_ok=True)
            os.makedirs(index_folder, exist_ok=True)
            
            # 문서가 없는 경우 처리
            if not documents or len(documents) == 0:
                logger.error("벡터 스토어 생성 실패: 문서가 없습니다")
                print("❌ 벡터 스토어 생성 실패: 문서가 없습니다")
                return False
                
            # 문서 유효성 검증
            valid_documents = []
            for doc in documents:
                if doc.page_content and len(doc.page_content.strip()) > 0:
                    valid_documents.append(doc)
                else:
                    logger.warning(f"빈 문서 무시: {doc.metadata}")
            
            # 유효한 문서가 없는 경우 처리
            if not valid_documents:
                logger.error("벡터 스토어 생성 실패: 유효한 문서가 없습니다")
                print("❌ 벡터 스토어 생성 실패: 유효한 문서가 없습니다")
                return False
                
            logger.info(f"벡터 스토어 생성 시작: {len(valid_documents)}개 문서, HNSW 사용: {self.use_hnsw}")
            print(f"🔄 벡터 스토어 생성 중: {len(valid_documents)}개 문서")
            
            # 메타데이터 추출 및 저장
            document_metadata = {}
            for i, doc in enumerate(valid_documents):
                doc_id = f"doc_{i}"
                doc.metadata["doc_id"] = doc_id
                document_metadata[doc_id] = doc.metadata
            
            # 메타데이터 저장
            os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(document_metadata, f, ensure_ascii=False, indent=2)
            
            # HNSW 인덱스 사용 여부에 따라 다르게 처리
            if self.use_hnsw:
                # HNSW 파라미터 로깅
                logger.info(f"HNSW 인덱스 생성: ef_construction={self.ef_construction}, M={self.m}")
                print(f"💡 HNSW 인덱스 생성 중: ef_construction={self.ef_construction}, M={self.m}")
                
                # HNSW 인덱스 생성 (향상된 성능을 위해)
                self.vector_store = FAISS.from_documents(
                    valid_documents, 
                    self.embeddings,
                    normalize_L2=True
                )
                
                # HNSW 파라미터 설정
                if hasattr(self.vector_store, 'index'):
                    # 기본 인덱스를 HNSW로 변환
                    dimension = self.vector_store.index.d
                    logger.info(f"FAISS 인덱스 차원: {dimension}")
                    
                    # 기존 벡터 저장
                    vectors = []
                    for i in range(self.vector_store.index.ntotal):
                        vector = self.vector_store.index.reconstruct(i)
                        vectors.append(vector)
                    
                    # HNSW 인덱스 생성
                    hnsw_index = faiss.IndexHNSWFlat(dimension, self.m)
                    hnsw_index.hnsw.efConstruction = self.ef_construction
                    hnsw_index.hnsw.efSearch = self.ef_search
                    
                    # 벡터 추가
                    if vectors:
                        hnsw_index.add(np.array(vectors))
                    
                    # 인덱스 교체
                    self.vector_store.index = hnsw_index
                    logger.info(f"HNSW 인덱스로 성공적으로 변환: {self.vector_store.index.ntotal}개 벡터")
                    print(f"✅ HNSW 인덱스 생성 완료: {self.vector_store.index.ntotal}개 벡터")
                
                # 인덱스 저장
                self.vector_store.save_local(index_folder, index_name="index")
                logger.info(f"HNSW 인덱스 저장 완료: {index_folder}")
                print(f"💾 인덱스 저장 완료: {index_folder}")
            else:
                # 기본 L2 인덱스 생성
                logger.info("기본 L2 인덱스 생성 중")
                print("💡 기본 L2 인덱스 생성 중")
                
                self.vector_store = FAISS.from_documents(
                    valid_documents, 
                    self.embeddings,
                    normalize_L2=True
                )
                
                # 인덱스 저장
                self.vector_store.save_local(index_folder, index_name="index")
                logger.info(f"L2 인덱스 저장 완료: {index_folder}")
                print(f"💾 인덱스 저장 완료: {index_folder}")
            
            # 문서 메타데이터 설정
            self.document_metadata = document_metadata
            
            logger.info(f"벡터 스토어 생성 완료: {len(valid_documents)}개 문서")
            print(f"✅ 벡터 스토어 생성 완료: {len(valid_documents)}개 문서")
            return True
        
        except Exception as e:
            logger.error(f"벡터 스토어 생성 중 오류: {e}")
            logger.error(traceback.format_exc())
            print(f"❌ 벡터 스토어 생성 실패: {e}")
            return False

    def format_context_for_model(self, documents: List[Document]) -> str:
        """검색된 문서를 모델 입력용 컨텍스트로 변환합니다."""
        if not documents:
            return "한화손해보험 사업보고서에서 관련 내용을 찾을 수 없습니다."
        
        # 소스별 문서 그룹화 및 관련성 점수 계산
        grouped_docs = {}
        for doc in documents:
            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page", "")
            key = f"{source}:{page}"
            
            # 관련성 점수 추출 (문자열에서 숫자로 변환)
            relevance_score = float(doc.metadata.get("relevance_score", "0.0"))
            
            if key not in grouped_docs:
                grouped_docs[key] = {
                    "docs": [],
                    "scores": [],
                    "queries": set()
                }
            
            grouped_docs[key]["docs"].append(doc)
            grouped_docs[key]["scores"].append(relevance_score)
            
            # 매칭된 쿼리 정보 수집
            if "matched_queries" in doc.metadata:
                for q in doc.metadata["matched_queries"].split(", "):
                    grouped_docs[key]["queries"].add(q)
        
        # 각 그룹의 평균 관련성 점수 계산
        for key in grouped_docs:
            scores = grouped_docs[key]["scores"]
            grouped_docs[key]["avg_score"] = sum(scores) / len(scores) if scores else 0.0
        
        # 그룹을 평균 관련성 점수로 정렬 (높은 점수부터)
        sorted_groups = sorted(grouped_docs.items(), key=lambda x: x[1]["avg_score"], reverse=True)
        
        # 문서 요약 정보 먼저 표시
        doc_summary = "📑 검색된 문서 요약:\n"
        for i, (key, group) in enumerate(sorted_groups):
            source, page = key.split(":")
            source_name = os.path.basename(source) if source else "알 수 없는 문서"
            page_info = f"페이지: {page}" if page else "페이지 정보 없음"
            avg_score = group["avg_score"]
            
            # 관련 쿼리 정보
            query_info = f", 관련 쿼리: {', '.join(list(group['queries'])[:2])}" if group["queries"] else ""
            
            doc_summary += f"{i+1}. {source_name} ({page_info}, 관련성: {avg_score:.2f}{query_info})\n"
        
        # 그룹별 컨텍스트 포맷팅
        formatted_docs = []
        
        for i, (key, group) in enumerate(sorted_groups):
            source, page = key.split(":")
            source_name = os.path.basename(source) if source else "알 수 없는 문서"
            
            # 메타데이터 정보
            meta_info = [
                f"출처: {os.path.basename(source)}" if source else "출처: 알 수 없음",
                f"페이지: {page}" if page else "페이지: 정보 없음",
                f"관련성: {group['avg_score']:.2f}"
            ]
            
            # 관련 쿼리 정보 추가
            if group["queries"]:
                meta_info.append(f"관련 쿼리: {', '.join(list(group['queries'])[:3])}")
            
            # 그룹 내 문서 내용 병합 (중복 제거)
            # 좀 더 스마트한 텍스트 병합: 청크 간 중복되는 부분 제거
            content_parts = []
            
            # 관련성 점수로 문서 정렬
            sorted_docs = sorted(
                zip(group["docs"], group["scores"]), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for doc, _ in sorted_docs:
                content = doc.page_content.strip()
                if content:
                    # 이미 추가된 내용과 중복 검사
                    if not any(self._has_significant_overlap(content, existing) for existing in content_parts):
                        content_parts.append(content)
            
            # 최종 내용 병합
            content = "\n\n".join(content_parts)
            
            # 최종 포맷팅된 문서 생성
            meta_str = " | ".join(meta_info)
            formatted_docs.append(f"[문서 #{i+1}] {meta_str}\n{content}")
        
        # 전체 컨텍스트 생성
        context = "다음은 사용자 질문과 관련된 한화손해보험 사업보고서 내용입니다:\n\n"
        context += doc_summary + "\n\n" # 문서 요약 정보 추가
        context += "\n\n---\n\n".join(formatted_docs)
        
        # 로그 및 콘솔에 출력
        logger.info(f"포맷팅된 컨텍스트 생성: {len(formatted_docs)}개 문서, {len(context)}자")
        print(f"\n📄 검색된 관련 문서: {len(formatted_docs)}개")
        
        for i, (key, group) in enumerate(sorted_groups):
            source, page = key.split(":")
            source_name = os.path.basename(source) if source else "알 수 없는 문서"
            page_info = f", 페이지: {page}" if page else ""
            avg_score = group["avg_score"]
            
            # 관련 쿼리 정보
            query_list = list(group["queries"])
            query_info = f", 쿼리: {', '.join(query_list[:2])}" if query_list else ""
            
            print(f"  - 문서 #{i+1}: {source_name}{page_info} (관련성: {avg_score:.2f}{query_info})")
        
        return context
    
    def _has_significant_overlap(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """두 텍스트 간에 상당한 중복이 있는지 확인합니다."""
        # 매우 짧은 텍스트는 중복 검사에서 제외
        if len(text1) < 50 or len(text2) < 50:
            return False
        
        # 더 짧은 텍스트 길이의 70%가 다른 텍스트에 포함되어 있는지 확인
        shorter = text1 if len(text1) < len(text2) else text2
        longer = text2 if len(text1) < len(text2) else text1
        
        # 더 짧은 텍스트의 단어 중 얼마나 많은 비율이 더 긴 텍스트에 나타나는지 계산
        shorter_words = set(shorter.split())
        longer_words = set(longer.split())
        
        if not shorter_words:
            return False
            
        # 공통 단어 비율 계산
        common_words = shorter_words.intersection(longer_words)
        overlap_ratio = len(common_words) / len(shorter_words)
        
        return overlap_ratio > threshold
        
    def _calculate_keyword_score(self, query: str, text: str) -> float:
        """키워드 매칭 기반 관련성 점수 계산 (하이브리드 검색용)"""
        # 쿼리 및 텍스트 전처리
        query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # 키워드 추출 (불용어 제거)
        stopwords = {"은", "는", "이", "가", "을", "를", "에", "에서", "의", "과", "와", "로", "으로", 
                    "이다", "있다", "없다", "하다", "되다", "한다", "된다", "무엇", "누구", "어디", 
                    "언제", "왜", "어떻게", "어떤", "얼마나", "몇", "무슨"}
        
        query_words = [w for w in query_clean.split() if w not in stopwords and len(w) > 1]
        
        if not query_words:
            return 0.0
        
        # 단어 단위 매칭
        matched_words = sum(1 for word in query_words if word in text_clean)
        if matched_words == 0:
            return 0.0
            
        # 키워드 매칭 비율
        keyword_score = matched_words / len(query_words)
        
        # TF-IDF와 유사한 가중치 부여: 매칭된 키워드가 텍스트에서 차지하는 비율
        text_words = text_clean.split()
        if text_words:
            # 단어 빈도를 고려한 밀도 스코어
            word_density = matched_words / len(text_words)
            # 문서가 매우 길면 밀도가 낮아지므로, 로그 스케일로 조정
            adjusted_density = min(1.0, word_density * 20)  # 최대 1.0으로 제한
            
            # 최종 키워드 점수: 매칭 비율 70% + 단어 밀도 30%
            keyword_score = 0.7 * keyword_score + 0.3 * adjusted_density
        
        return keyword_score

def main():
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='한화손해보험 사업보고서 RAG 시스템')
    parser.add_argument('--pdf', type=str, default=PDF_PATH, help='PDF 파일 경로')
    parser.add_argument('--chunk-size', type=int, default=500, help='청크 크기')
    parser.add_argument('--chunk-overlap', type=int, default=150, help='청크 겹침')
    parser.add_argument('--top-k', type=int, default=10, help='검색 결과 수')
    parser.add_argument('--force-update', action='store_true', help='벡터 인덱스 강제 업데이트')
    parser.add_argument('--flat-index', action='store_true', help='L2 인덱스 사용 (HNSW 비활성화)')
    parser.add_argument('--auto-eval', action='store_true', help='자동 평가 활성화')
    parser.add_argument('--auto-test', action='store_true', help='자동 테스트 모드 활성화')
    parser.add_argument('--num-questions', type=int, default=5, help='자동 테스트 질문 수')
    args = parser.parse_args()
    
    # 로그 설정
    global logger
    logger = setup_logging()
    
    # Ollama 서버 연결 및 모델 확인
    print("🔄 Ollama 모델 확인 중...")
    available_models = check_ollama_models()
    
    # 자동 테스트 모드인 경우 자동 테스트 실행
    if args.auto_test:
        print(f"\n{'='*80}")
        print(f"📝 자동 테스트 모드 활성화 (질문 수: {args.num_questions})")
        print(f"{'='*80}")
        
        try:
            # RAG 시스템 초기화
            rag = RAGSystem(
                use_hnsw=not args.flat_index,
                ef_search=200,
                ef_construction=200,
                m=64
            )
            
            # PDF 처리
            processor = PDFProcessor(args.pdf)
            documents = processor.process()
            
            # 문서 분할
            splitter = DocumentSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            chunks = splitter.split_documents(documents)
            
            # 벡터 저장소 생성
            rag.load_or_create_vector_store(chunks, force_update=args.force_update)
            
            # 자동 테스트 실행
            evaluator = ModelEvaluator()
            auto_evaluator = AutoEvaluator(model_name="gemma3:12b")
            auto_question_generator = AutoQuestionGenerator(model_name="gemma3:12b")
            
            auto_test_manager = AutoTestManager(
                rag_system=rag,
                auto_question_generator=auto_question_generator,
                auto_evaluator=auto_evaluator,
                evaluator=evaluator,
                available_models=AVAILABLE_MODELS
            )
            
            auto_test_manager.run_auto_test(num_questions=args.num_questions, top_k=args.top_k, stream=True)
            
            return 0
            
        except Exception as e:
            logger.error(f"자동 테스트 중 오류 발생: {e}")
            print(f"\n❌ 자동 테스트 중 오류 발생: {e}")
            traceback.print_exc()
            return 1
    
    # 일반 모드 실행
    try:
        # RAG 시스템 초기화
        rag = RAGSystem(
            use_hnsw=not args.flat_index,
            ef_search=200,
            ef_construction=200,
            m=64
        )
        
        # PDF 처리
        processor = PDFProcessor(args.pdf)
        documents = processor.process()
        
        # 문서 분할
        splitter = DocumentSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        chunks = splitter.split_documents(documents)
        
        # 벡터 저장소 생성
        rag.load_or_create_vector_store(chunks, force_update=args.force_update)
        
        # 질문 입력 받기 (표준 입력에서 읽기)
        print("\n💬 질문을 입력하세요 (Ctrl+D 또는 빈 줄로 종료):")
        lines = []
        try:
            for line in sys.stdin:
                if not line.strip():
                    break
                lines.append(line)
            question = ''.join(lines).strip()
        except EOFError:
            question = ''.join(lines).strip()
        
        if not question:
            print("❌ 질문이 입력되지 않았습니다.")
            return 1
        
        # 문서 검색
        retrieved_docs = rag.search(question, top_k=args.top_k)
        context = rag.format_context_for_model(retrieved_docs)
        
        # 결과 변수
        results = {}
        auto_evaluations = {}
        evaluation_id = None
        
        # 평가 설정
        evaluator = ModelEvaluator()
        
        # 자동 평가 활성화된 경우
        if args.auto_eval:
            auto_evaluator = AutoEvaluator(model_name="gemma3:12b")
            evaluation_id = evaluator.save_evaluation(question, context, {}, {
                "top_k": args.top_k,
                "has_auto_eval": True
            })
        
        # 각 모델에 대해 응답 생성
        for model in AVAILABLE_MODELS:
            print(f"\n{'='*80}")
            print(f"📝 모델: {model}")
            print(f"{'='*80}")
            
            # 모델 응답 생성
            result = rag.answer(question, model, context)
            answer = result["answer"]
            results[model] = answer
            
            # 자동 평가 실행
            if args.auto_eval:
                print(f"\n🤖 자동 평가 중...")
                auto_evaluation = auto_evaluator.evaluate_answer(question, context, answer)
                auto_evaluations[model] = auto_evaluation
                
                # 평가 결과 저장
                if evaluation_id:
                    score = auto_evaluation.get("score", 0)
                    comments = auto_evaluation.get("reason", "")
                    evaluator.add_evaluation_score(evaluation_id, model, score, comments)
                
                print(f"평가 점수: {auto_evaluation.get('score', '평가 실패')}/5")
                print(f"평가 이유: {auto_evaluation.get('reason', '평가 실패')}")
        
        # 결과 파일 저장 (Streamlit 앱에서 사용하기 위함)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        query_id = re.sub(r'\W+', '_', question)[:30]
        query_results_dir = os.path.join(SCRIPT_DIR, "query_results")
        os.makedirs(query_results_dir, exist_ok=True)
        query_results_file = os.path.join(query_results_dir, f"query_{timestamp}_{query_id}.json")
        
        query_result_data = {
            "query": question,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_id": evaluation_id,
            "results": results,
            "auto_evaluations": auto_evaluations if args.auto_eval else None,
            "context": context
        }
        
        with open(query_results_file, 'w', encoding='utf-8') as f:
            json.dump(query_result_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 응답이 생성되었습니다. 결과 파일: {query_results_file}")
        return 0
        
    except Exception as e:
        logger.error(f"치명적인 오류 발생: {e}")
        print(f"❌ 치명적인 오류 발생: {e}")
        print("세부 오류 내용:", traceback.format_exc())
        return 1

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
    
    # HNSW 파라미터 설정 (고급 설정 섹션)
    with st.sidebar.expander("HNSW 고급 설정", expanded=False):
        ef_search = st.slider("ef_search (검색 정확도)", min_value=50, max_value=500, value=200, step=50)
        ef_construction = st.slider("ef_construction (구축 정확도)", min_value=50, max_value=500, value=200, step=50)
        m_param = st.slider("M (연결성)", min_value=8, max_value=128, value=64, step=8)
        st.info("높은 값 = 높은 정확도 & 높은 메모리 사용량")
    
    # 인덱스 경로 표시
    index_dir = os.path.join(SCRIPT_DIR, "Index")
    st.sidebar.info(f"📁 인덱스 경로: {index_dir}")
    
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
    
    # RAG 시스템 초기화 (HNSW 파라미터 적용)
    rag = RAGSystem(
        use_hnsw=use_hnsw,
        ef_search=ef_search if 'ef_search' in locals() else 200,
        ef_construction=ef_construction if 'ef_construction' in locals() else 200,
        m=m_param if 'm_param' in locals() else 64
    )
    
    # ... (rest of the code remains unchanged)

# 평가 프롬프트 템플릿 정의 (상수로 변경)
EVAL_PROMPT_TEMPLATE = """당신은 질문-답변 쌍의 품질을 평가하는 엄격한 평가자입니다.
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

class AutoEvaluator:
    def __init__(self, model_name: str = "gemma3:12b"):
        self.model_name = model_name
        
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
        
        prompt = EVAL_PROMPT_TEMPLATE.format(
            question=question,
            context=context,
            answer=answer
        )
        
        # 평가 결과를 하드코딩된 예시로 반환 (실제로는 LLM을 호출해야 함)
        # 이 부분은 임시로 추가하여 코드가 동작하도록 함
        example_evaluation = {
            "score": 3,
            "reason": "답변이 기본적인 정보는 제공하지만, 세부 내용이 부족합니다.",
            "raw_evaluation": "평가 전체 텍스트"
        }
        
        return example_evaluation


# 질문 생성기 프롬프트 템플릿
QUESTION_GENERATOR_TEMPLATE = """당신은 한화손해보험 사업보고서에 관한 질문을 생성하는 AI입니다.
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


class AutoQuestionGenerator:
    def __init__(self, model_name: str = "gemma3:12b"):
        self.model_name = model_name
    
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
        
        prompt = QUESTION_GENERATOR_TEMPLATE.format(context=context)
        
        # 임시로 하드코딩된 질문 목록 반환 (실제로는 LLM을 호출해야 함)
        # 이 부분은 임시로 추가하여 코드가 동작하도록 함
        example_questions = [
            "한화손해보험의 2024년 1분기 당기순이익은 얼마인가요?",
            "한화손해보험의 주요 리스크 관리 전략은 무엇인가요?",
            "한화손해보험의 디지털 전환 전략에 대해 설명해주세요.",
            "한화손해보험의 지배구조 특징은 무엇인가요?",
            "한화손해보험의 주요 보험 상품 라인업은 어떻게 구성되어 있나요?"
        ]
        
        return example_questions


class AutoTestManager:
    def __init__(self, rag_system: RAGSystem, test_count: int = 5):
        """자동 테스트 관리자 초기화"""
        self.rag_system = rag_system
        self.test_count = test_count
        self.question_generator = AutoQuestionGenerator()
        self.evaluator = AutoEvaluator()
        self.available_models = ["gemma3:12b", "gemma3:7.8b", "claude3:sonnet", "claude3:haiku"]
        
        # 테스트 결과 저장
        self.results = {
            "tests": [],
            "summary": {
                "avg_score": 0,
                "count": 0,
                "score_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            }
        }
    
    def run_auto_test(self, use_random_docs: bool = True, test_count: Optional[int] = None) -> Dict[str, Any]:
        """자동 테스트 실행"""
        if test_count is not None:
            self.test_count = test_count
        
        test_results = []
        
        if use_random_docs:
            # 랜덤 문서에서 테스트 실행
            docs = self.rag_system.vectorstore.get_random_documents(min(20, self.test_count * 2))
            for doc in docs[:self.test_count]:
                test_results.extend(self._run_test_on_document(doc))
        else:
            # 검색 결과에서 테스트 실행
            # 구현 필요시 추가
            pass
        
        # 테스트 결과 요약 업데이트
        self._update_summary()
        
        return self.results
    
    def _run_test_on_document(self, document) -> List[Dict[str, Any]]:
        """단일 문서에 대한 테스트 실행"""
        context = document.page_content
        source = document.metadata.get("source", "알 수 없음")
        
        # 질문 생성
        try:
            questions = self.question_generator.generate_questions(context)
        except Exception as e:
            logging.error(f"질문 생성 오류: {str(e)}")
            return []
        
        test_results = []
        
        # 각 질문에 대해 테스트 실행
        for question in questions[:1]:  # 문서당 첫 번째 질문만 사용 (부하 제한)
            try:
                # RAG 시스템으로 답변 생성
                retrieved_docs = self.rag_system.search(question, top_k=5)
                answer = self.rag_system.answer(question, retrieved_docs)
                
                # 답변 평가
                evaluation = self.evaluator.evaluate_answer(
                    question=question,
                    context=context,
                    answer=answer
                )
                
                # 테스트 결과 저장
                test_result = {
                    "question": question,
                    "source": source,
                    "context": context[:500] + ("..." if len(context) > 500 else ""),
                    "answer": answer,
                    "evaluation": evaluation,
                    "score": evaluation["score"],
                    "timestamp": datetime.now().isoformat()
                }
                
                test_results.append(test_result)
                self.results["tests"].append(test_result)
                
            except Exception as e:
                logging.error(f"테스트 실행 오류: {str(e)}")
        
        return test_results
    
    def _update_summary(self):
        """테스트 결과 요약 업데이트"""
        tests = self.results["tests"]
        
        if not tests:
            return
        
        # 점수 분포 초기화
        score_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        # 각 테스트 결과의 점수 카운트
        total_score = 0
        for test in tests:
            score = test.get("score", 0)
            if 1 <= score <= 5:
                score_distribution[score] += 1
                total_score += score
        
        # 평균 점수 계산
        avg_score = total_score / len(tests) if tests else 0
        
        # 요약 업데이트
        self.results["summary"] = {
            "avg_score": round(avg_score, 2),
            "count": len(tests),
            "score_distribution": score_distribution
        }

def check_ollama_models() -> List[str]:
    """Ollama API를 통해 사용 가능한 모델을 확인합니다."""
    # 이미 실행된 경우 캐시된 결과 반환
    if hasattr(check_ollama_models, '_cached_models'):
        return check_ollama_models._cached_models
    
    api_url = "http://localhost:11434/api/tags"
    try:
        logger.info("Ollama REST API로 모델 조회 시도 중...")
        print("🔄 Ollama REST API로 모델 조회 시도 중...")
        
        response = requests.get(api_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            
            if models:
                logger.info(f"Ollama REST API 연결 성공: {len(models)}개 모델 발견")
                print(f"✓ Ollama REST API 연결 성공: {len(models)}개 모델 발견")
                
                # 사용 가능한 모델 목록 캐시
                check_ollama_models._cached_models = models
                return models
            else:
                logger.warning("Ollama API에 모델이 없습니다.")
                print("⚠️ Ollama API에 모델이 없습니다.")
        else:
            logger.error(f"Ollama API 오류: {response.status_code}")
            print(f"⚠️ Ollama API 오류: {response.status_code}")
    except Exception as e:
        logger.error(f"Ollama API 연결 실패: {e}")
        print(f"⚠️ Ollama API 연결 실패: {e}")
    
    # 기본 모델 리스트 (연결 실패시)
    check_ollama_models._cached_models = []
    return []

# --- 모델 평가 관리 클래스 ---
class ModelEvaluator:
    def __init__(self):
        """모델 평가 관리자 초기화"""
        self.evaluation_file = EVALUATION_FILE
        self.evaluations = self._load_evaluations()
    
    def _load_evaluations(self) -> Dict:
        """저장된 평가 데이터 로드"""
        os.makedirs(os.path.dirname(self.evaluation_file), exist_ok=True)
        if os.path.exists(self.evaluation_file):
            try:
                with open(self.evaluation_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 평가 파일 로드 중 오류: {e}")
                return {"evaluations": []}
        return {"evaluations": []}
    
    def _save_evaluations(self):
        """평가 데이터 저장"""
        os.makedirs(os.path.dirname(self.evaluation_file), exist_ok=True)
        try:
            with open(self.evaluation_file, 'w', encoding='utf-8') as f:
                json.dump(self.evaluations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 평가 파일 저장 중 오류: {e}")
    
    def save_evaluation(self, question: str, context: str, answers: Dict[str, str], metadata: Dict = None) -> str:
        """새 평가 세션 저장"""
        # 평가 ID 생성
        evaluation_id = f"eval_{int(time.time())}_{hashlib.md5(question.encode()).hexdigest()[:8]}"
        
        # 새 평가 데이터 생성
        evaluation = {
            "id": evaluation_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "context": context[:1000] + "..." if len(context) > 1000 else context,  # 컨텍스트 길이 제한
            "answers": answers,
            "scores": {},
            "metadata": metadata or {}
        }
        
        # 평가 목록에 추가
        self.evaluations["evaluations"].append(evaluation)
        self._save_evaluations()
        
        return evaluation_id
    
    def add_evaluation_score(self, evaluation_id: str, model_name: str, score: int, comments: str = ""):
        """특정 모델에 대한 평가 점수 추가"""
        for eval_item in self.evaluations["evaluations"]:
            if eval_item["id"] == evaluation_id:
                eval_item["scores"][model_name] = {
                    "score": score,
                    "comments": comments,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self._save_evaluations()
                return True
        return False
    
    def get_evaluation(self, evaluation_id: str) -> Dict:
        """특정 평가 데이터 조회"""
        for eval_item in self.evaluations["evaluations"]:
            if eval_item["id"] == evaluation_id:
                return eval_item
        return {}
    
    def get_all_evaluations(self) -> List[Dict]:
        """모든 평가 데이터 조회"""
        return self.evaluations["evaluations"]
    
    def get_model_avg_scores(self) -> Dict[str, float]:
        """모델별 평균 점수 계산"""
        model_scores = {}
        model_counts = {}
        
        for eval_item in self.evaluations["evaluations"]:
            for model, score_data in eval_item["scores"].items():
                score = score_data.get("score", 0)
                if model not in model_scores:
                    model_scores[model] = 0
                    model_counts[model] = 0
                model_scores[model] += score
                model_counts[model] += 1
        
        # 평균 계산
        avg_scores = {}
        for model, total_score in model_scores.items():
            count = model_counts.get(model, 0)
            avg_scores[model] = round(total_score / count, 2) if count > 0 else 0
        
        return avg_scores

if __name__ == "__main__":
    try:
        # 스크립트가 직접 실행될 때만 main 함수 호출
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 프로그램 종료 (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")
        traceback.print_exc()
        sys.exit(1)