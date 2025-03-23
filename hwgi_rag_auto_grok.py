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
from typing import List, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv
import pypdf
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.vectorstores import FAISS

# 디버그 모드 설정
DEBUG_MODE = False

# OLLAMA_AVAILABLE 변수 정의
import ollama
OLLAMA_AVAILABLE = True

# 현재 스크립트 파일의 절대 경로 가져오기
SCRIPT_DIR = os.getcwd()
BASE_DIR = os.path.dirname(SCRIPT_DIR)
print(f"현재 스크립트 파일 경로: {SCRIPT_DIR}")
print(f"상위 디렉토리 경로: {BASE_DIR}")

# .env 파일에서 환경변수 로드
load_dotenv()

# OpenMP 스레드 수 제한 (FAISS와 Java 충돌 방지)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 환경 설정
PDF_PATH = os.path.join(SCRIPT_DIR, "[한화손해보험]사업보고서(2025.03.11).pdf")
INDEX_DIR = os.path.join(SCRIPT_DIR, "Index")
METADATA_FILE = os.path.join(SCRIPT_DIR, "Index/document_metadata_bge.json")
LOG_FILE = os.path.join(SCRIPT_DIR, "Log/hwgi_rag_streamlit.log")
CACHE_FILE = os.path.join(SCRIPT_DIR, "cache.json")

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

# ### BGE-M3 임베딩 클래스
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
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트의 임베딩을 반환"""
        try:
            batch_size = 64
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = [self._preprocess_text(text) for text in texts[i:i + batch_size]]
                with torch.inference_mode():
                    embeddings = self.model.encode(
                        batch,
                        convert_to_tensor=True,
                        device=self.device,
                        normalize_embeddings=True
                    )
                    if self.device.type == "mps":
                        embeddings = embeddings.to("cpu")
                    all_embeddings.extend(embeddings.cpu().numpy().tolist())
            return all_embeddings
        except Exception as e:
            print(f"❌ 문서 임베딩 생성 중 오류: {e}")
            raise e
    
    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 텍스트의 임베딩을 반환"""
        try:
            query_text = f"query: {text}"
            with torch.inference_mode():
                embedding = self.model.encode(
                    [query_text],
                    convert_to_tensor=True,
                    device=self.device,
                    normalize_embeddings=True
                )
                if self.device.type == "mps":
                    embedding = embedding.to("cpu")
                return embedding.cpu().numpy().tolist()[0]
        except Exception as e:
            print(f"❌ 쿼리 임베딩 생성 중 오류: {e}")
            raise e

# ### PDF 처리 클래스
class PDFProcessor:
    def __init__(self, pdf_path: str):
        if not os.path.isabs(pdf_path):
            self.pdf_path = os.path.join(SCRIPT_DIR, pdf_path)
        else:
            self.pdf_path = pdf_path
        self.text_content = []
        self.tables = []
        self.page_count = 0
        self.pdf_hash = self._calculate_pdf_hash()
        self.hash_file = os.path.join(SCRIPT_DIR, "pdf_hash.json")
        logger.info(f"PDFProcessor 초기화: '{self.pdf_path}' 파일 처리 준비")
    
    def _calculate_pdf_hash(self) -> str:
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_hash = hashlib.md5(file.read()).hexdigest()
            return pdf_hash
        except Exception as e:
            logger.error(f"PDF 해시 계산 중 오류: {e}")
            return ""
    
    def _load_previous_hash(self) -> str:
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
        previous_hash = self._load_previous_hash()
        needs_processing = previous_hash != self.pdf_hash
        if not needs_processing:
            logger.info("이전에 처리된 동일한 PDF 파일 감지")
            print("✓ 이미 처리된 PDF 파일입니다")
        return needs_processing
    
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
            return []
    
    def extract_tables(self) -> List[Document]:
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
                        table_content = []
                        for row in table:
                            if any(cell for cell in row):
                                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                                table_content.append(cleaned_row)
                        if not table_content:
                            continue
                        table_text = '\n'.join([','.join(row) for row in table_content])
                        table_hash = hashlib.md5(table_text.encode('utf-8')).hexdigest()
                        metadata = {
                            'table_id': f'table_p{page_num}_t{table_idx + 1}',
                            'page': page_num,
                            'source': 'table',
                            'hash': table_hash,
                            'row_count': len(table_content),
                            'col_count': len(table_content[0]) if table_content else 0
                        }
                        self.tables.append({
                            'table_id': metadata['table_id'],
                            'content': table_text,
                            'raw_data': table_content,
                            'hash': table_hash,
                            'metadata': metadata
                        })
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
            return []
    
    def process(self) -> List[Document]:
        print(f"\n{'─'*60}")
        print("📌 1단계: PDF 문서 처리")
        print(f"{'─'*60}")
        if not self.needs_processing():
            logger.info("이전에 처리된 동일한 PDF 파일 감지")
            print("✓ 이미 처리된 PDF 파일입니다")
            return []
        logger.info("===== PDF 처리 시작 =====")
        text_docs = self.extract_text()
        table_docs = self.extract_tables()
        all_docs = text_docs + table_docs
        if not all_docs:
            print("⚠️ PDF에서 문서를 추출하지 못했습니다")
            return []
        self._save_current_hash()
        logger.info(f"📚 {len(all_docs)}개의 문서 조각 생성됨")
        print(f"📚 PDF 처리 완료: {len(text_docs)}개 텍스트 문서, {len(table_docs)}개 표 문서 생성")
        return all_docs

# ### 문서 분할 클래스
class DocumentSplitter:
    def __init__(self, chunk_size=800, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", ",", " ", ""]
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

# ### RAG 시스템 클래스
class RAGSystem:
    def __init__(self, embedding_type: str = "bge-m3", use_hnsw: bool = True, ef_search: int = 200, ef_construction: int = 200, m: int = 64):
        print("🔧 RAG 시스템 초기화 중...")
        print(f"  - 임베딩 모델: {embedding_type}")
        
        if hasattr(RAGSystem, '_initialized'):
            logger.info("RAG 시스템이 이미 초기화되어 있습니다")
            return
        
        RAGSystem._initialized = True
        self.embedding_type = embedding_type
        self.embedding_name = None
        
        if embedding_type == "bge-m3":
            self.embeddings = BGEM3Embeddings(model_name="BAAI/bge-m3")
            self.embedding_name = "bge-m3"
        
        self._cache = self._load_cache()
        self.use_hnsw = use_hnsw
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.m = m
        self.vector_store = None
        self.index_dir = INDEX_DIR
        os.makedirs(self.index_dir, exist_ok=True)
        self.ollama_base_url = "http://localhost:11434/api"
        self.available_models = self._check_ollama_models()
        
        if self.available_models:
            print(f"✓ 사용할 모델: {self.available_models[0]}")
        print("✅ RAG 시스템 초기화 완료")
    
    def _load_cache(self) -> Dict[str, str]:
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
        cache_file = os.path.join(SCRIPT_DIR, "cache.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"캐시 파일 저장 실패: {e}")
    
    def _check_ollama_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.ollama_base_url}/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models if model["name"] in AVAILABLE_MODELS]
            return []
        except Exception as e:
            logger.error(f"Ollama 모델 확인 중 오류: {e}")
            return []
    
    def _generate_with_ollama(self, prompt: str, model: str, stream: bool = True) -> str:
        logger.debug(f"답변 생성 요청 - 쿼리: '{prompt[:30]}...', 모델: {model}")
        api_json = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": 0.7, "top_p": 0.9}
        }
        response_text = ""
        try:
            if stream:
                print(f"\n================================================================================")
                print(f"📝 모델: {model}")
                print(f"================================================================================")
                with requests.post(f"{self.ollama_base_url}/generate", json=api_json, stream=True) as response:
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            response_text += chunk
                            print(chunk, end="", flush=True)
                            if data.get("done", False):
                                print()
            else:
                response = requests.post(f"{self.ollama_base_url}/generate", json=api_json)
                if response.status_code == 200:
                    response_text = response.json().get("response", "")
                    print(response_text)
            return response_text
        except Exception as e:
            error_msg = f"텍스트 생성 중 오류 발생: {e}"
            logger.error(error_msg)
            print(f"\n❌ {error_msg}")
            return f"⚠️ {error_msg}"
    
    def answer(self, query: str, model: str, context: str) -> Dict[str, Any]:
        cache_key = f"{model}:{hashlib.md5((query + context[:200]).encode()).hexdigest()}"
        cached_answer = self._cache.get(cache_key)
        if cached_answer and not os.environ.get('DISABLE_CACHE'):
            logger.info(f"캐시된 응답 사용: 모델={model}")
            print(f"💾 캐시된 응답 사용: {model}")
            return {"answer": cached_answer, "model": model, "cached": True}
        
        from datetime import datetime
        today = datetime.now().strftime("%Y년 %m월 %d일")
        qa_template = f"""당신은 현재 시간 {today} 기준으로 한화손해보험 사업보고서의 내용을 기반으로 정확하고 철저한 답변을 제공하는 전문가 AI 비서입니다.

다음 지침을 철저히 따라 답변해 주세요:
1. 제공된 문서 내용에만 기반하여 답변하세요.
2. 숫자, 날짜, 금액 등 사실적 정보는 문서 그대로 정확히 인용하세요.
3. 답변은 한국어로 작성하고, 명확하고 구조적으로 작성하세요.

사용자 질문: {query}

참고 문서 내용:
{context}

위 내용을 바탕으로 사용자 질문에 대한 답변:"""
        try:
            result = self._generate_with_ollama(qa_template, model, stream=True)
            self._cache[cache_key] = result
            self._save_cache()
            return {"answer": result, "model": model, "cached": False}
        except Exception as e:
            logger.error(f"❌ 답변 생성 중 오류: {e}")
            return {"answer": f"답변을 생성하는 중 오류가 발생했습니다: {str(e)}", "model": model, "error": True}
    
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        logger.info(f"검색 요청: '{query}' (top_k={top_k})")
        print(f"\n🔎 검색 중: '{query}'\n")
        if self.vector_store is None:
            print("❌ 벡터 스토어가 초기화되지 않았습니다")
            return []
        try:
            docs = self.vector_store.similarity_search(query, k=top_k)
            logger.info(f"검색 결과: {len(docs)}개 문서")
            print(f"✓ 검색 결과: {len(docs)}개 문서")
            return docs
        except Exception as e:
            logger.error(f"검색 중 오류 발생: {e}")
            print(f"❌ 검색 중 오류 발생: {e}")
            return []
    
    def load_or_create_vector_store(self, documents: List[Document], force_update: bool = False) -> bool:
        index_folder = os.path.join(INDEX_DIR, "faiss_index_bge-m3")
        metadata_file = os.path.join(INDEX_DIR, "document_metadata_bge-m3.json")
        
        if not force_update and os.path.exists(os.path.join(index_folder, "index.faiss")) and os.path.exists(os.path.join(index_folder, "index.pkl")):
            try:
                self.vector_store = FAISS.load_local(
                    index_folder,
                    self.embeddings,
                    normalize_L2=True,
                    allow_dangerous_deserialization=True
                )
                if self.use_hnsw and hasattr(self.vector_store.index, 'hnsw'):
                    self.vector_store.index.hnsw.efSearch = self.ef_search
                logger.info(f"인덱스 로드 성공: {index_folder}")
                print(f"✓ 인덱스 로드 성공: {index_folder}")
                return True
            except Exception as e:
                logger.error(f"벡터 스토어 로드 실패: {e}")
                print(f"❌ 기존 인덱스 로드 실패: {e}")
        
        return self._create_new_vector_store(documents, index_folder, metadata_file)
    
    def _create_new_vector_store(self, documents: List[Document], index_folder: str, metadata_file: str) -> bool:
        logger.info("벡터 저장소 생성 시작")
        print("🔄 벡터 저장소 생성 중...")
        if not documents:
            logger.error("유효한 문서가 없습니다")
            print("❌ 유효한 문서가 없습니다")
            return False
        
        for i, doc in enumerate(documents):
            doc.metadata["doc_id"] = f"doc_{i}"
        
        document_metadata = [{"content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content, **doc.metadata} for doc in documents]
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(document_metadata, f, ensure_ascii=False, indent=2)
        
        try:
            vector_store = FAISS.from_documents(documents, self.embeddings, normalize_L2=True, distance_strategy="COSINE")
            if self.use_hnsw and hasattr(vector_store.index, 'hnsw'):
                vector_store.index.hnsw.efConstruction = self.ef_construction
                vector_store.index.hnsw.efSearch = self.ef_search
                vector_store.index.hnsw.M = self.m
            vector_store.save_local(index_folder)
            self.vector_store = vector_store
            logger.info(f"✅ 벡터 저장소 생성 완료: {len(documents)}개 문서")
            print(f"✅ 벡터 저장소 생성 완료: {len(documents)}개 문서")
            return True
        except Exception as e:
            logger.error(f"벡터 저장소 생성 중 오류: {e}")
            print(f"❌ 벡터 저장소 생성 중 오류: {e}")
            return False
    
    def format_context_for_model(self, documents: List[Document]) -> str:
        if not documents:
            return "관련 문서를 찾을 수 없습니다"
        formatted_docs = [f"[출처: {doc.metadata.get('source', '알 수 없음')}, 페이지: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}" for doc in documents]
        return "다음은 사용자 질문과 관련된 한화손해보험 사업보고서 내용입니다:\n\n" + "\n\n---\n\n".join(formatted_docs)

# ### 메인 함수
def main():
    parser = argparse.ArgumentParser(description='한화손해보험 사업보고서 RAG 시스템')
    parser.add_argument('--pdf', type=str, default=PDF_PATH, help='PDF 파일 경로')
    parser.add_argument('--chunk-size', type=int, default=800, help='청크 크기')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='청크 겹침')
    parser.add_argument('--top-k', type=int, default=5, help='검색 결과 수')
    parser.add_argument('--force-update', action='store_true', help='벡터 인덱스 강제 업데이트')
    args = parser.parse_args()
    
    rag = RAGSystem(use_hnsw=True, ef_search=200, ef_construction=200, m=64)
    processor = PDFProcessor(args.pdf)
    documents = processor.process()
    splitter = DocumentSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(documents)
    rag.load_or_create_vector_store(chunks, force_update=args.force_update)
    
    print("\n💬 질문을 입력하세요 (종료하려면 Ctrl+D):")
    import sys
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
        print("❌ 질문이 입력되지 않았습니다")
        return 1
    
    retrieved_docs = rag.search(question, top_k=args.top_k)
    context = rag.format_context_for_model(retrieved_docs)
    for model in AVAILABLE_MODELS:
        print(f"\n{'='*80}")
        print(f"📝 모델: {model}")
        print(f"{'='*80}")
        result = rag.answer(question, model, context)
        print(result["answer"])
    
    return 0

if __name__ == "__main__":
    main()