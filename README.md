# 한화손해보험 사업보고서 RAG 시스템

사업보고서와 같은 금융 문서를 분석하고 질의응답을 수행하는 RAG(Retrieval Augmented Generation) 시스템입니다. 이 프로젝트는 BGE-M3 임베딩 모델과 Ollama를 통해 제공되는 로컬 LLM을 활용하여 구축되었습니다.

## 주요 기능

- **PDF 문서 처리**: 표, 텍스트 추출 및 청킹
- **벡터 검색**: FAISS를 이용한 효율적인 유사도 검색
- **쿼리 확장**: 원본 쿼리를 의미적으로 확장하여 검색 품질 향상
- **다중 모델 비교**: Gemma3, Llama3.1 등 여러 모델의 응답 비교
- **자동 평가**: 생성된 응답의 품질 자동 평가
- **CLI & Web UI**: 명령줄 인터페이스와 Streamlit 기반 웹 인터페이스 제공

## 시스템 요구사항

- Python 3.8 이상
- [Ollama](https://ollama.com/) - 로컬 LLM 실행을 위한 도구
- 최소 8GB RAM (16GB 이상 권장)
- GPU 권장 (MPS 또는 CUDA 지원)

## 설치 방법

1. 저장소 클론
   ```bash
   git clone https://github.com/yourusername/hwgi-rag-system.git
   cd hwgi-rag-system
   ```

2. 필요 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

3. Ollama 설치 및 모델 다운로드
   ```bash
   # Ollama 설치 (https://ollama.com/download)
   
   # 필요한 모델 다운로드
   ollama pull gemma3:4b
   ollama pull llama3.1:8b
   ollama pull gemma3:12b
   ```

## 사용 방법

### 1. CLI 모드

```bash
# 기본 실행
python hwgi_rag_auto.py

# 다양한 옵션으로 실행
python hwgi_rag_auto.py --pdf 경로/사업보고서.pdf --chunk-size 500 --chunk-overlap 150 --top-k 10 --auto-eval
```

### 2. Streamlit 웹 인터페이스

```bash
streamlit run hwgi_streamlit_app.py
```

## 파라미터 설명

- `--pdf`: 처리할 PDF 파일 경로 (기본값: './[한화손해보험]사업보고서(2025.03.11).pdf')
- `--chunk-size`: 청크 크기 (기본값: 500)
- `--chunk-overlap`: 청크 겹침 크기 (기본값: 150)
- `--top-k`: 검색 결과 수 (기본값: 10)
- `--force-update`: 벡터 인덱스 강제 업데이트 (기본값: False)
- `--auto-eval`: 자동 평가 활성화 (기본값: False)
- `--auto-test`: 자동 테스트 모드 실행 (기본값: False)
- `--num-questions`: 자동 테스트에서 생성할 질문 수 (기본값: 5)

## 참고 사항

- 처음 실행 시 임베딩 모델 다운로드 및 인덱스 생성에 시간이 소요될 수 있습니다.
- RAG 시스템 사용 시 최상의 결과를 위해 구체적인 질문을 권장합니다.
- 자동 평가 기능은 gemma3:12b 모델이 필요합니다.

## 라이선스

MIT 