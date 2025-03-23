import streamlit as st

# 기본 설정 - 반드시 다른 st 명령어보다 먼저 호출해야 함
st.set_page_config(
    page_title="한화손해보험 사업보고서 RAG 시스템",
    page_icon="📊",
    layout="wide"
)

import subprocess
import os
import sys
import glob
import json
import time
import threading
import pandas as pd
from datetime import datetime
import tempfile
import re
import langsmith
from langsmith import Client
from langsmith.run_helpers import traceable
import uuid
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# LangSmith 환경 변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "hwgi_rag_streamlit"  # 프로젝트 이름 설정

# LangSmith 클라이언트 설정 - st.sidebar 호출을 나중에 수행
try:
    langsmith_client = Client()
    langsmith_enabled = True
    # 여기서 st.sidebar 호출 제거
except Exception as e:
    langsmith_enabled = False
    print(f"LangSmith 연결 실패: {e}")

# 세션 상태 초기화
if 'auto_test_results' not in st.session_state:
    st.session_state.auto_test_results = None
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = {}
if 'backend_initialized' not in st.session_state:
    st.session_state.backend_initialized = False
if 'backend_process' not in st.session_state:
    st.session_state.backend_process = None
if 'multi_queries' not in st.session_state:
    st.session_state.multi_queries = []
if 'doc_summaries' not in st.session_state:
    st.session_state.doc_summaries = []
if 'original_query' not in st.session_state:
    st.session_state.original_query = ""
if 'raw_output' not in st.session_state:
    st.session_state.raw_output = ""
if 'auto_test_log' not in st.session_state:
    st.session_state.auto_test_log = ""
if 'run_ids' not in st.session_state:
    st.session_state.run_ids = {}

# 더미 traceable 데코레이터 함수 정의 (LangSmith 비활성화 시 사용)
def dummy_traceable(*args, **kwargs):
    # 함수를 그대로 반환하는 데코레이터
    def decorator(func):
        return func
    # run_type 인자가 직접 전달된 경우
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator

# 세션 상태 초기화 (API 키 유지를 위해)
if 'langsmith_api_key' not in st.session_state:
    st.session_state.langsmith_api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    
# LangSmith 활성화 여부 (기본값: 비활성화)
if 'enable_langsmith' not in st.session_state:
    st.session_state.enable_langsmith = False
    
# langsmith_error_occurred 플래그 초기화
if 'langsmith_error_occurred' not in st.session_state:
    st.session_state.langsmith_error_occurred = False

# LangSmith 환경 변수 확인
langsmith_api_key = st.session_state.langsmith_api_key
traceable = dummy_traceable  # 기본값으로 더미 함수 사용

# PDF 파일 목록 가져오기
def get_pdf_files():
    pdf_files = glob.glob("*.pdf")
    return pdf_files

# 쿼리 결과 파일 목록 가져오기
def get_query_results_files():
    if os.path.exists("query_results"):
        files = glob.glob("query_results/query_*.json")
        return sorted(files, key=os.path.getmtime, reverse=True)  # 최신 파일 순으로 정렬
    return []

# 제목 및 설명
st.title("📊 한화손해보험 RAG 시스템")
# st.subheader("(Local 모델 비교: Llama3.1:8b vs Gemma3:4b)")

# 사이드바 설정
st.sidebar.header("⚙️ 환경 설정")

# LangSmith 연결 상태를 사이드바에 표시 (이곳으로 이동)
if langsmith_enabled:
    st.sidebar.success("✅ LangSmith 연결됨")
else:
    st.sidebar.warning("⚠️ LangSmith 연결 실패")

# PDF 파일 업로드
uploaded_file = st.sidebar.file_uploader(
    "PDF 파일 업로드",
    type=["pdf"],
    help="분석할 PDF 파일을 업로드하세요. 파일은 서버에 임시로 저장됩니다."
)

# 업로드된 파일 처리
selected_pdf = None
if uploaded_file is not None:
    # 파일 확장자 검증
    if not uploaded_file.name.lower().endswith('.pdf'):
        st.sidebar.error("❌ PDF 파일만 업로드 가능합니다.")
    else:
        # 임시 파일로 저장
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 파일명 중복 방지를 위해 타임스탬프 추가
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_pdf_path = os.path.join(temp_dir, f"{timestamp}_{uploaded_file.name}")
        
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        selected_pdf = temp_pdf_path
        st.sidebar.success(f"✅ '{uploaded_file.name}' 파일이 업로드되었습니다.")
else:
    # 기존 PDF 파일 목록 (업로드가 없을 경우 선택 가능)
    pdf_files = get_pdf_files()
    if not pdf_files:
        st.sidebar.warning("⚠️ PDF 파일을 업로드하거나 프로그램 디렉토리에 PDF를, 저장해 주세요.")
        selected_pdf = None
    else:
        default_pdf = st.sidebar.selectbox(
            "또는 기존 PDF 파일 선택",
            options=pdf_files,
            index=0
        )
        selected_pdf = default_pdf
        st.sidebar.info(f"📄 선택된 PDF 파일: {os.path.basename(selected_pdf)}")

# 청크 크기 및 겹침 설정
chunk_size = st.sidebar.slider("청크 크기", min_value=100, max_value=1000, value=500, step=50)
chunk_overlap = st.sidebar.slider("청크 겹침", min_value=50, max_value=300, value=150, step=25)

# 검색 결과 수 설정
top_k = st.sidebar.slider("검색 결과 수 (Top-K)", min_value=3, max_value=20, value=10)

# 인덱스 강제 업데이트 옵션 (체크박스는 제거하고 기본값 설정)
force_update = st.sidebar.checkbox("벡터 인덱스 강제 업데이트", value=False)
# force_update = False

# HNSW 인덱스 사용 옵션 (체크박스는 제거하고 기본값 설정)
# use_hnsw = st.sidebar.checkbox("HNSW 인덱스 사용 (정확도 향상)", value=True)
use_hnsw = True

# 자동 평가 옵션 (체크박스는 제거하고 기본값 설정)
# auto_eval = st.sidebar.checkbox("자동 평가 활성화 (gemma3:12b 필요)", value=True)
auto_eval = True

# 메인 영역
tab1, tab2, tab3 = st.tabs(["💬 질문 응답", "🔄 자동 테스트", "🔍 디버그"])

# LangSmith로 추적하는 함수
@traceable(name="extract_info")
def extract_multi_queries_and_docs(output):
    multi_queries = []
    doc_summaries = []
    original_query = ""
    
    # 원본 쿼리 추출
    original_query_pattern = r"(?:💬 질문:|원본 질문:)\s*(.*?)(?:\n|$)"
    original_query_match = re.search(original_query_pattern, output)
    if original_query_match:
        original_query = original_query_match.group(1).strip()
    
    # 멀티쿼리 추출 (패턴 강화)
    multi_query_sections = [
        r"🔄 확장 질의:\s*\n(.*?)(?=\n\n|\n🔍|\n💡)",
        r"멀티쿼리 생성 결과:\s*\n(.*?)(?=\n\n|\n🔍|\n💡)",
        r"멀티쿼리 확장 결과:\s*\n(.*?)(?=\n\n|\n🔍|\n💡)",
        r"생성된 쿼리:\s*\n(.*?)(?=\n\n|\n🔍|\n💡)",
    ]
    
    # 여러 패턴 시도
    for pattern in multi_query_sections:
        multi_query_match = re.search(pattern, output, re.DOTALL)
        if multi_query_match:
            query_text = multi_query_match.group(1).strip()
            for line in query_text.split('\n'):
                clean_line = line.strip()
                if clean_line.startswith('- '):
                    multi_queries.append(clean_line[2:])  # '- ' 제거
                elif clean_line and not clean_line.startswith('#') and not clean_line.startswith('=='):  
                    # '#'이나 '=='으로 시작하지 않는 비어있지 않은 라인
                    multi_queries.append(clean_line)
    
    # 문서 요약 정보 추출
    docs_pattern = r"🔍 문서 검색 중\.\.\.[\s\S]*?(페이지.*?)(?:\n\n|\n💡)"
    docs_match = re.search(docs_pattern, output, re.DOTALL)
    if docs_match:
        doc_text = docs_match.group(1)
        doc_lines = [line.strip() for line in doc_text.split('\n') if line.strip()]
        doc_summaries = doc_lines
    
    return original_query, multi_queries, doc_summaries

# LangSmith로 추적하는 백그라운드 질문 처리 함수
@traceable(name="process_question", run_type="chain")
def process_question_with_file(question, pdf_path, chunk_size, chunk_overlap, top_k, force_update, use_hnsw, auto_eval):
    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as temp_file:
        temp_file.write(question)
        temp_file_path = temp_file.name
    
    try:
        # 프로세스 실행 중 플래그 설정
        st.session_state.process_running = True
        st.session_state.raw_output = "쿼리 처리 시작 중...\n"
        
        # 명령어 구성
        cmd = [
            "python3", "hwgi_rag_auto.py",
            "--pdf", pdf_path,
            "--chunk-size", str(chunk_size),
            "--chunk-overlap", str(chunk_overlap),
            "--top-k", str(top_k)
        ]
        
        # 추가 옵션
        if force_update:
            cmd.append("--force-update")
        if not use_hnsw:
            cmd.append("--flat-index")
        if auto_eval:
            cmd.append("--auto-eval")
        
        # 디버깅 정보 추가
        cmd_str = ' '.join(cmd)
        print(f"실행 명령어: {cmd_str}")
        st.session_state.raw_output += f"실행 명령어: {cmd_str}\n\n"
        st.session_state.raw_output += "쿼리 실행 중...\n"
        
        # 파일에서 질문 읽기 위한 입력 리디렉션 방식으로 변경
        # 이렇게 하면 명령행 인수로 질문을 전달하지 않아 공백 문제를 피할 수 있음
        with open(temp_file_path, 'r') as input_file:
            result = subprocess.run(
                cmd,
                stdin=input_file,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # 디버깅을 위한 출력 (stderr가 있는 경우만)
            if result.stderr:
                print(f"하위 프로세스 오류 출력: {result.stderr}")
                st.session_state.raw_output += f"\n오류 출력:\n{result.stderr}\n"
            
            # 표준 출력 추가
            if result.stdout:
                st.session_state.raw_output += f"\n명령 출력:\n{result.stdout}\n"
            
            # 실행 완료 메시지 추가
            if result.returncode == 0:
                st.session_state.raw_output += "\n✅ 프로세스 실행 완료\n"
            else:
                st.session_state.raw_output += f"\n❌ 프로세스 실행 실패 (종료 코드: {result.returncode})\n"
                
        return result
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        # 프로세스 종료 플래그 설정
        st.session_state.process_running = False

# 질문 응답 탭
with tab1:
    st.header("질문 응답")
    
    # 저장된 쿼리 결과 확인 옵션 추가
    query_results_files = get_query_results_files()
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
                
                # 멀티쿼리 표시 (있는 경우)
                if 'context' in query_data:
                    with st.expander("🔍 검색된 문서 컨텍스트", expanded=False):
                        st.markdown(query_data['context'])
                
                # 각 모델의 응답 표시
                if 'results' in query_data:
                    for model, answer in query_data['results'].items():
                        with st.expander(f"📝 모델: {model}", expanded=True):
                            st.markdown(answer)
                            
                            # 자동 평가 결과 표시 (있는 경우)
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
    
    # 새 쿼리 입력이 선택되었거나 저장된 결과가 없는 경우에만 입력 폼 표시
    if not query_results_files or selected_result_file == "새 쿼리 입력":
        # 질문 입력
        question = st.text_area("질문을 입력하세요:", height=100, value=st.session_state.last_question)
        
        if st.button("질문 제출", type="primary", disabled=(not selected_pdf)):
            if question:
                st.session_state.last_question = question
                
                with st.spinner("질문에 답변 생성 중..."):
                    try:
                        # LangSmith 추적을 위한 고유 ID 생성
                        run_id = None
                        if langsmith_enabled:
                            try:
                                run_id = str(uuid.uuid4())
                                st.session_state.run_ids['last_query'] = run_id
                                
                                # LangSmith 메타데이터 설정
                                langsmith_client.create_run(
                                    name="질문 응답 시작",
                                    run_type="chain",
                                    inputs={"query": question, "pdf": selected_pdf},
                                    run_id=run_id
                                )
                            except Exception as e:
                                st.warning(f"LangSmith 추적 활성화 실패 (영향 없음): {str(e)}")
                                print(f"LangSmith 추적 오류: {e}")
                                langsmith_enabled = False
                        
                        # 파일을 사용하여 질문 처리
                        result = process_question_with_file(
                            question,
                            selected_pdf,
                            chunk_size,
                            chunk_overlap,
                            top_k,
                            force_update,
                            use_hnsw,
                            auto_eval
                        )
                        
                        # 결과 처리
                        if result.returncode == 0:
                            output = result.stdout
                            # 디버깅을 위해 원본 출력 저장
                            st.session_state.raw_output = output
                            
                            # 원본 쿼리, 멀티쿼리와 문서 요약 정보 추출
                            original_query, multi_queries, doc_summaries = extract_multi_queries_and_docs(output)
                            st.session_state.original_query = original_query
                            st.session_state.multi_queries = multi_queries
                            st.session_state.doc_summaries = doc_summaries
                            
                            # 응답 파싱 (각 모델별 응답 섹션 추출)
                            model_answers = {}
                            current_model = None
                            answer_text = ""
                            
                            for line in output.split('\n'):
                                if line.startswith("📝 모델:"):
                                    if current_model and answer_text:
                                        model_answers[current_model] = answer_text.strip()
                                    current_model = line.replace("📝 모델:", "").strip()
                                    answer_text = ""
                                elif current_model and not line.startswith("─") and not line.startswith("🤖"):
                                    answer_text += line + "\n"
                            
                            # 마지막 모델 응답 저장
                            if current_model and answer_text:
                                model_answers[current_model] = answer_text.strip()
                            
                            st.session_state.last_answer = model_answers
                            
                            # LangSmith에 결과 기록
                            if langsmith_enabled:
                                langsmith_client.update_run(
                                    run_id=run_id,
                                    outputs={
                                        "model_answers": model_answers,
                                        "multi_queries": multi_queries,
                                        "doc_summaries": doc_summaries
                                    },
                                    end_time=datetime.utcnow(),
                                    error=None
                                )
                        else:
                            st.error("명령 실행 실패:")
                            st.code(result.stderr)
                            st.code(result.stdout)
                            
                            # LangSmith에 오류 기록
                            if langsmith_enabled:
                                langsmith_client.update_run(
                                    run_id=run_id,
                                    outputs=None,
                                    end_time=datetime.utcnow(),
                                    error=result.stderr
                                )
                        
                        # 로그 표시
                        # if result.stderr:
                        #     st.error("오류 발생:")
                        #     st.code(result.stderr)
                    
                    except subprocess.TimeoutExpired as e:
                        st.warning("실행 시간이 너무 깁니다. 프로세스가 계속 실행 중일 수 있습니다.")
                        # LangSmith에 오류 기록
                        if langsmith_enabled and 'run_id' in locals():
                            langsmith_client.update_run(
                                run_id=run_id,
                                outputs=None,
                                end_time=datetime.utcnow(),
                                error=str(e)
                            )
                    except Exception as e:
                        st.error(f"오류 발생: {str(e)}")
                        # LangSmith에 오류 기록
                        if langsmith_enabled and 'run_id' in locals():
                            langsmith_client.update_run(
                                run_id=run_id,
                                outputs=None,
                                end_time=datetime.utcnow(),
                                error=str(e)
                            )
    
    # 원본 쿼리와 멀티쿼리를 하나의 섹션에 표시
    if st.session_state.multi_queries:
        with st.expander("🔄 검색 쿼리 (Query Generation)", expanded=True):
            # 원본 쿼리 표시
            if st.session_state.original_query:
                st.markdown(f"**원본 질문**: {st.session_state.original_query}")
            
            # 구분선 추가
            st.markdown("---")
            
            # 멀티쿼리 표시
            st.markdown("**생성된 검색 쿼리**:")
            
            # 테이블 형식으로 표시
            query_data = []
            for i, query in enumerate(st.session_state.multi_queries):
                query_data.append({
                    "번호": i+1,
                    "검색 쿼리": query
                })
            
            if query_data:
                st.table(pd.DataFrame(query_data))
    
    # 검색된 문서 개요 표시
    if st.session_state.doc_summaries:
        with st.expander("🔍 검색된 문서 개요", expanded=False):
            for doc in st.session_state.doc_summaries:
                st.markdown(f"- {doc}")
    
    # 모델 응답 표시
    if st.session_state.last_answer:
        st.subheader(f"질문: {st.session_state.last_question}")
        
        for model, answer in st.session_state.last_answer.items():
            with st.expander(f"📝 모델: {model}", expanded=True):
                st.markdown(answer)
    
    # 로그 표시 영역 추가
    if st.session_state.raw_output:
        with st.expander("🔄 실행 로그", expanded=True):
            st.code(st.session_state.raw_output)
            
    # 실행 중인 프로세스 확인
    status_container = st.empty()
    if 'process_running' not in st.session_state:
        st.session_state.process_running = False
    
    if st.session_state.process_running:
        status_container.info("💬 쿼리 처리 중... 위 로그를 확인하세요.")
    else:
        status_container.empty()

# 자동 테스트 탭
with tab2:
    st.header("자동 테스트")
    
    col1, col2 = st.columns(2)
    with col1:
        num_questions = st.number_input("생성할 질문 수", min_value=1, max_value=20, value=5)
    
    if st.button("자동 테스트 실행", disabled=(not selected_pdf)):
        # LangSmith 추적을 위한 고유 ID 생성
        auto_test_run_id = str(uuid.uuid4())
        st.session_state.run_ids['auto_test'] = auto_test_run_id
        
        # LangSmith 메타데이터 설정
        if langsmith_enabled:
            langsmith_client.create_run(
                name="자동 테스트 실행",
                run_type="chain",
                inputs={"pdf": selected_pdf, "num_questions": num_questions},
                run_id=auto_test_run_id
            )
            
        # 명령어 구성
        cmd = [
            "python3", "hwgi_rag_auto.py",
            "--pdf", selected_pdf,
            "--chunk-size", str(chunk_size),
            "--chunk-overlap", str(chunk_overlap),
            "--top-k", str(top_k),
            "--auto-test"
        ]
        
        # 자동 평가 옵션 추가 
        if auto_eval:
            cmd.append("--auto-eval")
        
        # 추가 옵션
        cmd.extend(["--num-questions", str(num_questions)])
        if force_update:
            cmd.append("--force-update")
        if not use_hnsw:
            cmd.append("--flat-index")
        
        with st.spinner(f"{num_questions}개의 자동 테스트 질문 생성 및 평가 중..."):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 자동 테스트는 더 긴 타임아웃 허용
                )
                
                # 로그 저장
                st.session_state.auto_test_log = result.stdout
                
                # 가장 최근 자동 테스트 결과 파일 찾기
                auto_test_files = glob.glob("auto_test_results_*.json")
                if auto_test_files:
                    latest_file = max(auto_test_files, key=os.path.getctime)
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        st.session_state.auto_test_results = json.load(f)
                    
                    # LangSmith에 결과 기록
                    if langsmith_enabled:
                        langsmith_client.update_run(
                            run_id=auto_test_run_id,
                            outputs={"results": st.session_state.auto_test_results},
                            end_time=datetime.utcnow(),
                            error=None
                        )
                
                # 로그 표시
                if result.stderr:
                    st.error("오류 발생:")
                    st.code(result.stderr)
                    
                    # LangSmith에 오류 기록
                    if langsmith_enabled:
                        langsmith_client.update_run(
                            run_id=auto_test_run_id,
                            outputs=None,
                            end_time=datetime.utcnow(),
                            error=result.stderr
                        )
            
            except subprocess.TimeoutExpired as e:
                st.warning("실행 시간이 너무 깁니다. 프로세스가 계속 실행 중일 수 있습니다.")
                # LangSmith에 오류 기록
                if langsmith_enabled:
                    langsmith_client.update_run(
                        run_id=auto_test_run_id,
                        outputs=None,
                        end_time=datetime.utcnow(),
                        error=str(e)
                    )
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
                # LangSmith에 오류 기록
                if langsmith_enabled:
                    langsmith_client.update_run(
                        run_id=auto_test_run_id,
                        outputs=None,
                        end_time=datetime.utcnow(),
                        error=str(e)
                    )
    
    # 자동 테스트 결과 표시
    if st.session_state.auto_test_results:
        st.subheader("자동 테스트 결과")
        
        # 결과 데이터 구성
        results = st.session_state.auto_test_results
        
        # 질문 목록 표시
        if "tests" in results:
            for i, test in enumerate(results.get("tests", [])):
                with st.expander(f"질문 {i+1}: {test.get('question')}", expanded=i==0):
                    # context가 None인 경우 오류 방지
                    context = test.get('context')
                    if context:
                        st.markdown(f"**컨텍스트**: {context[:500]}...")
                    else:
                        st.markdown("**컨텍스트**: 컨텍스트 정보가 없습니다.")
                    
                    # 모델별 응답 표시
                    if "responses" in test:
                        for model, response in test.get("responses", {}).items():
                            st.markdown(f"**모델**: {model}")
                            st.markdown(response.get("answer", "응답 없음"))
                            
                            # 평가 결과가 있고 자동 평가가 활성화된 경우에만 표시
                            if auto_eval and "evaluation" in response:
                                score = response.get("evaluation", {}).get("score", "평가 없음")
                                reason = response.get("evaluation", {}).get("reason", "이유 없음")
                                st.markdown(f"**평가 점수**: {score}/5")
                                st.markdown(f"**평가 이유**: {reason}")
                    else:
                        st.markdown("모델 응답 정보가 없습니다.")
                    
                    st.markdown("---")
        else:
            st.warning("테스트 결과가 올바른 형식이 아닙니다.")
        
        # 종합 결과 표시 (자동 평가가 활성화된 경우에만)
        if auto_eval and "summary" in results:
            st.subheader("모델 평가 종합")
            summary_data = []
            
            for model, stats in results.get("summary", {}).items():
                summary_data.append({
                    "모델": model,
                    "평균 점수": round(stats.get("avg_score", 0), 2),
                    "총 평가 수": stats.get("total_evaluations", 0)
                })
            
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data))

# 디버그 탭 - 멀티쿼리 디버깅을 위한 탭
with tab3:
    st.header("디버그 정보")
    
    # LangSmith 디버깅 섹션 추가
    if langsmith_enabled:
        with st.expander("🔍 LangSmith 실행 추적", expanded=True):
            st.markdown("### LangSmith 추적 정보")
            
            if st.session_state.run_ids:
                for run_type, run_id in st.session_state.run_ids.items():
                    st.markdown(f"**{run_type}**: `{run_id}`")
                    
                # LangSmith 대시보드 링크 제공
                st.markdown("LangSmith 대시보드에서 추적 정보 확인하기:")
                langsmith_url = "https://smith.langchain.com"
                st.markdown(f"[대시보드 열기]({langsmith_url})")
            else:
                st.info("아직 추적된 실행 정보가 없습니다. 질문을 제출하거나 자동 테스트를 실행하세요.")
    else:
        st.warning("LangSmith 연결이 설정되지 않았습니다. API 키를 확인하세요.")
    
    # 로그 영역 추가 - 터미널 로그를 가장 먼저 표시
    if st.session_state.raw_output:
        with st.expander("📋 실시간 로그 출력", expanded=True):
            st.markdown("### 터미널 출력 로그")
            st.code(st.session_state.raw_output)
    
    # 멀티쿼리 디버깅
    if st.session_state.raw_output:
        with st.expander("🔄 멀티쿼리 디버그", expanded=False):
            st.markdown("### 멀티쿼리 관련 부분")
            
            # 멀티쿼리 관련 부분 추출
            multi_query_debug = ""
            lines = st.session_state.raw_output.split('\n')
            in_multi_query_section = False
            
            for line in lines:
                if "확장 질의" in line or "멀티쿼리" in line or "생성된 쿼리" in line:
                    in_multi_query_section = True
                    multi_query_debug += f"**{line}**\n"
                elif in_multi_query_section and (line.strip() == "" or "문서 검색" in line):
                    in_multi_query_section = False
                    multi_query_debug += "---\n"
                elif in_multi_query_section:
                    multi_query_debug += f"{line}\n"
            
            st.markdown(multi_query_debug)
    
    # 자동 테스트 로그 표시 (자동 테스트가 실행된 경우)
    # 세션 상태에 자동 테스트 로그 추가
    if 'auto_test_log' not in st.session_state:
        st.session_state.auto_test_log = ""
    
    if st.session_state.auto_test_log:
        with st.expander("📋 자동 테스트 로그", expanded=False):
            st.markdown("### 자동 테스트 실행 로그")
            st.code(st.session_state.auto_test_log)

# 페이지 하단 정보
st.markdown("---")
st.markdown("""
## 사용 방법
1. 왼쪽 사이드바에서 PDF 파일과 설정을 선택하세요.
2. '질문 응답' 탭에서 직접 질문을 입력하거나 '자동 테스트' 탭에서 자동 테스트를 실행하세요.
3. 자동 테스트는 최근 테스트 결과를 표시합니다.
4. LangSmith 대시보드에서 실행 과정과 결과를 추적하고 분석할 수 있습니다.

## 문제 해결
- Streamlit 페이지가 표시되지 않는 경우, 콘솔에서 `streamlit run hwgi_streamlit_app.py` 명령어로 실행하세요.
- Ollama 서버가 실행 중인지 확인하세요. (`http://localhost:11434`에 접속 가능해야 함)
- 로컬 환경에 Python 라이브러리가 모두 설치되어 있는지 확인하세요.
- 자동 평가를 사용하려면 gemma3:12b 모델이 설치되어 있어야 합니다. 설치하려면 `ollama pull gemma3:12b` 명령어를 사용하세요.
- LangSmith 추적을 위해서는 환경 변수 `LANGCHAIN_API_KEY`가 설정되어 있어야 합니다.
""")

# 기본 실행 정보를 표시
st.sidebar.markdown("---")
st.sidebar.markdown("### 시스템 정보")
st.sidebar.markdown("- Ollama 모델 자동 감지 및 사용")
st.sidebar.markdown("- 사업보고서 RAG 시스템")
st.sidebar.markdown("- 질문은 표준 입력으로 전달됩니다")
if langsmith_enabled:
    st.sidebar.markdown("- LangSmith 추적 활성화됨")
if auto_eval:
    st.sidebar.markdown("- 자동 평가 모드 활성화됨")
else:
    st.sidebar.markdown("- 자동 평가 모드 비활성화됨") 