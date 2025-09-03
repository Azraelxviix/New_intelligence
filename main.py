#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py - Version 17.0.0 (Resilient, Observable, and Optimized)
import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from threading import Lock

import functions_framework
import numpy as np
import psycopg
from google.api_core import exceptions as api_exceptions
from google.api_core.client_options import ClientOptions
from google.cloud import aiplatform, documentai as docai_v1
from google.cloud import firestore
from google.cloud import secretmanager
from google.cloud import storage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg import sql
from cloudevents.http import CloudEvent

import google.generativeai as genai
from google.genai import types
from vertexai.language_models import TextEmbeddingModel

SCRIPT_VERSION = "17.0.0"
LOCK_LEASE_MINUTES = 20

# --- Environment Configuration ---
os.environ["PROJECT_ID"] = "thebestever"
os.environ["VERTEX_AI_LOCATION"] = "us-central1"
os.environ["DOCAI_LOCATION"] = "us"
os.environ["INPUT_BUCKET"] = "knowledge-base-docs-thebestever"
os.environ["LOCK_BUCKET"] = "kblock"
os.environ["JSON_BUCKET"] = "kbjson"
os.environ["LOG_BUCKET"] = "kblogs"
os.environ["INSTRUCTIONS_BUCKET"] = "kbinfo"
os.environ["DOCAI_PROCESSOR_ID"] = "6e8f23fa5796a22b"
os.environ["INDEX_ENDPOINT_ID"] = "556724518584844288"
os.environ["DEPLOYED_INDEX_ID"] = "analysis_1756251260790"
os.environ["MASTER_INSTRUCTIONS_FILE"] = "master_instructions.txt"
os.environ["DB_USER"] = "retrieval-service"
os.environ["DB_NAME"] = "postgres"
os.environ["INSTANCE_CONNECTION_NAME"] = "thebestever:us-central1:genai-rag-db-6bdb68ec"
os.environ["ANALYSIS_MODEL"] = "gemini-1.5-pro"  # Use stable alias
os.environ["CLEANING_MODEL"] = "gemini-1.5-flash" # Use stable alias
os.environ["EMBEDDING_MODEL_NAME"] = "text-embedding-gecko@001" # Vertex AI Embedding Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_buffer = []

# --- Global, Thread-Safe Clients for Reuse ---
db_conn = None
db_lock = Lock()
genai_client = None
genai_client_lock = Lock()

def get_firestore_client(project_id: str, database: str = "(default)") -> firestore.Client:
    return firestore.Client(project=project_id, database=database)

def get_storage_client() -> storage.Client:
    return storage.Client()

def get_docai_client(location: str) -> docai_v1.DocumentProcessorServiceClient:
    client_options = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    return docai_v1.DocumentProcessorServiceClient(client_options=client_options)

def initialize_genai_client():
    """Initializes and returns a reusable, thread-safe genai.Client."""
    global genai_client
    with genai_client_lock:
        if genai_client is None:
            logging.info("Initializing new Google GenAI client...")
            try:
                client = secretmanager.SecretManagerServiceClient()
                project_id = os.environ["PROJECT_ID"]
                secret_name = "gemini_api"
                name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
                response = client.access_secret_version(name=name)
                api_key = response.payload.data.decode("UTF-8")
                genai.configure(api_key=api_key)
                logging.info("genai configured with API key.")
            except api_exceptions.NotFound:
                logging.warning("Gemini API key not found. Using Application Default Credentials (ADC).")
            
            # Pin to the stable 'v1' API for production reliability
            genai_client = genai.Client(http_options=types.HttpOptions(api_version='v1'))
        else:
            logging.info("Reusing existing Google GenAI client.")
    return genai_client

def get_db_connection():
    """Initializes and returns a reusable, thread-safe database connection."""
    global db_conn
    with db_lock:
        if db_conn is None or db_conn.closed:
            logging.info("No active DB connection. Initializing...")
            try:
                client = secretmanager.SecretManagerServiceClient()
                project_id = os.environ["PROJECT_ID"]
                secret_name = "db_retrieval_pass"
                name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
                response = client.access_secret_version(name=name)
                password = response.payload.data.decode("UTF-8")
                
                conn_str = (
                    f"dbname={os.environ['DB_NAME']} "
                    f"user={os.environ['DB_USER']} "
                    f"password={password} "
                    f"host=127.0.0.1 port=5432"
                )
                db_conn = psycopg.connect(conn_str)
                logging.info(f"DB connection established. ID: {id(db_conn)}")
            except Exception as e:
                logging.error(f"DB connection failed: {e}", exc_info=True)
                db_conn = None
                raise
        else:
            logging.info(f"Reusing existing DB connection. ID: {id(db_conn)}")
    return db_conn

def _log(step: int, total_steps: int, message: str, is_error: bool = False, is_final: bool = False, level: str = "INFO") -> None:
    global log_buffer
    if is_error:
        emoji_prefix = "🚨"
    elif is_final:
        emoji_prefix = "🏁"
    elif step > 0:
        emoji_prefix = "✅"
    else:
        emoji_prefix = "⏳"

    prefix_text = (f"Step {step}/{total_steps}: FAILED" if is_error else
                   "Pipeline Complete!" if is_final else
                   f"Step {step}/{total_steps}: SUCCESS")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_line = f"[{timestamp}] {emoji_prefix} {prefix_text} - {message}"
    log_buffer.append(log_line)
    
    log_func = getattr(logging, level.lower(), logging.info)
    log_func(log_line)
    print(log_line)

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[^a-zA-Z0-9.-]', '_', filename)

EMBEDDING_MODEL = None

def _get_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        _log(0, 0, f"LAZY LOADING: Initializing Vertex AI TextEmbeddingModel '{os.environ['EMBEDDING_MODEL_NAME']}'...")
        EMBEDDING_MODEL = TextEmbeddingModel.from_pretrained(os.environ["EMBEDDING_MODEL_NAME"])
        test_embedding_response = EMBEDDING_MODEL.get_embeddings(["test sentence"])
        test_embedding = test_embedding_response[0].values
        _log(0, 0, f"LAZY LOADING: Vertex AI TextEmbeddingModel loaded successfully with {len(test_embedding)} dimensions.")
        # text-embedding-gecko@001 has 768 dimensions
        if len(test_embedding) != 768:
            raise ValueError(f"Expected 768 dimensions for '{os.environ['EMBEDDING_MODEL_NAME']}', got {len(test_embedding)}")
    return EMBEDDING_MODEL

def check_and_reserve_document(case_id: int, document_name: str, db_conn) -> Optional[int]:
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (case_id, original_filename, processed_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (original_filename) DO NOTHING
            RETURNING id;
            """,
            (case_id, document_name)
        )
        result = cur.fetchone()
        if result is None:
            logging.warning(f"Document '{document_name}' already exists. Skipping reservation.")
            return None
        db_conn.commit()
        return result[0]

def update_document_with_content(document_id: int, document_text: str, chunk_embeddings: Optional[np.ndarray], db_conn) -> None:
    with db_conn.cursor() as cur:
        embedding: Optional[List[float]] = None
        if chunk_embeddings is not None and len(chunk_embeddings) > 0:
            embedding = np.mean(chunk_embeddings, axis=0).tolist()
        cur.execute(
            """
            UPDATE documents
            SET full_text = %s, full_text_embedding = %s, processed_at = NOW()
            WHERE id = %s;
            """,
            (document_text, embedding, document_id)
        )
        db_conn.commit()

def index_document_chunks(document_id: int, chunks: List[str], embeddings: np.ndarray, db_conn) -> None:
    _log(0, 0, f"Beginning to index {len(chunks)} chunks for document_id {document_id}...")
    with db_conn.cursor() as cur:
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (document_id, chunk_index) DO UPDATE
                SET chunk_text = EXCLUDED.chunk_text, embedding = EXCLUDED.embedding, updated_at = NOW();
                """,
                (document_id, i, chunk, embedding.tolist())
            )
        db_conn.commit()
    _log(4, 7, f"Successfully indexed {len(chunks)} chunks into Cloud SQL.")

def _call_generative_model(
    task_name: str,
    client: genai.Client,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    is_json: bool = False,
    response_schema: Optional[Dict] = None,
) -> Optional[str]:
    logging.info(f"Calling generative model for: {task_name}")
    try:
        generation_config_params = {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_output_tokens": 8192,
        }
        if is_json and response_schema:
            generation_config_params["response_mime_type"] = "application/json"
            if response_schema:
                generation_config_params["response_schema"] = response_schema
        
        model = client.generative_model(
            model_name=model_name,
            generation_config=generation_config_params
        )
        contents = [system_prompt, user_prompt]
        response = model.generate_content(contents=contents)

        return response.text
    except api_exceptions.NotFound as e:
        logging.error(f"Model '{model_name}' not found. Details: {e}", exc_info=True)
        return None
    except api_exceptions.InvalidArgument as e:
        logging.error(f"Invalid argument in request to model '{model_name}'. Details: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during the API call: {e}", exc_info=True)
        return None

def _get_context_from_rag(query_embedding: Optional[np.ndarray], db_conn, project_id: str, vertex_ai_location: str, index_endpoint_id: str, deployed_index_id: str) -> str:
    if query_embedding is None or query_embedding.size == 0:
        return "No relevant historical context found (Embedding missing)."
    _log(0, 0, "Retrieving context from RAG (Vertex AI Vector Search)...")
    try:
        query_vec = np.mean(query_embedding, axis=0).tolist() if query_embedding.ndim > 1 else query_embedding.tolist()
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(f"projects/{project_id}/locations/{vertex_ai_location}/indexEndpoints/{index_endpoint_id}")
        response = index_endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[query_vec],
            num_neighbors=5
        )
        if not response or not response[0]:
            return "No relevant historical context found in Vector Search."
        with db_conn.cursor() as cur:
            neighbor_ids = [int(n.id) for n in response[0] if n.id and n.id.isdigit()]
            if not neighbor_ids:
                return "No relevant historical context found in Vector Search (no valid IDs)."
            query = sql.SQL("SELECT chunk_text, (embedding <=> %s) AS distance FROM document_chunks WHERE id = ANY(%s) ORDER BY distance ASC")
            cur.execute(query, (query_vec, neighbor_ids))
            results = cur.fetchall()
            context = [f"- (Distance: {row[1]:.4f}): {row[0]}" for row in results]
            return "\n".join(context) if context else "No relevant historical context found."
    except Exception as e:
        logging.error(f"Error during RAG retrieval: {e}", exc_info=True)
        return "Error retrieving historical context."

def get_clean_document_text(input_file: str, mime_type: str, processor_id: str, temp_bucket: str, client: genai.Client) -> Optional[str]:
    _log(0, 0, "Extracting text with Document AI OCR...")
    try:
        docai_client = get_docai_client(os.environ["DOCAI_LOCATION"])
        processor_name = docai_client.processor_path(os.environ["PROJECT_ID"], os.environ["DOCAI_LOCATION"], processor_id)
        output_gcs_uri = f"gs://{temp_bucket}/ocr/{sanitize_filename(input_file.split('/')[-1])}/"
        operation = docai_client.batch_process_documents(
            request=docai_v1.BatchProcessRequest(
                name=processor_name,
                input_documents=docai_v1.BatchDocumentsInputConfig(
                    gcs_documents=docai_v1.GcsDocuments(documents=[docai_v1.GcsDocument(gcs_uri=input_file, mime_type=mime_type)]),
                ),
                document_output_config=docai_v1.DocumentOutputConfig(
                    gcs_output_config=docai_v1.DocumentOutputConfig.GcsOutputConfig(gcs_uri=output_gcs_uri, field_mask="text"),
                ),
            ),
        )
        operation.result(timeout=1800)
        _log(0, 0, "Document AI processing complete. Cleaning up output files.")
        storage_client = get_storage_client()
        metadata = docai_v1.BatchProcessMetadata(operation.metadata)
        if not metadata.individual_process_statuses:
            logging.error("No individual process statuses returned from Document AI.")
            return None
        output_gcs_path = metadata.individual_process_statuses[0].output_gcs_destination
        bucket_name, prefix = output_gcs_path.removeprefix("gs://").split("/", 1)
        output_blobs = list(storage_client.list_blobs(bucket_name, prefix=prefix))
        raw_text = "".join(
            docai_v1.Document.from_json(blob.download_as_bytes(), ignore_unknown_fields=True).text
            for blob in output_blobs if blob.name.endswith(".json")
        )
        for blob in output_blobs:
            blob.delete()
        
        if raw_text.strip():
            _log(0, 0, "OCR text extracted, now cleaning with GenAI...")
            return _ai_text_restoration(raw_text, client)
        else:
            _log(0, 0, "No text content found after OCR.", level="WARNING")
            return None
    except Exception as e:
        logging.error(f"Generic OCR failure: {e}", exc_info=True)
        return None

def _ai_text_restoration(text: str, client: genai.Client) -> str:
    return _call_generative_model(
        "AI Text Restoration",
        client=client,
        model_name=os.environ["CLEANING_MODEL"],
        system_prompt=(
            "The following text was extracted via OCR and may contain errors."
            "Your task is to clean and restore it. Do not add or remove information, only correct errors. "
            "Preserve paragraph structure."
        ),
        user_prompt=f"--- ORIGINAL TEXT ---\n{text}\n--- RESTORED TEXT ---",
        is_json=False
    ) or text

def generate_legal_analysis(
    document_text: str,
    db_conn,
    chunk_embeddings: Optional[np.ndarray],
    storage_client,
    instructions_bucket_name,
    master_instructions_file,
    analysis_model_name,
    client: genai.Client,
    project_id,
    vertex_ai_location,
    index_endpoint_id,
    deployed_index_id
) -> Optional[dict]:
    _log(5, 7, "Generating legal analysis...")
    try:
        instructions_blob = storage_client.bucket(instructions_bucket_name).blob(master_instructions_file)
        master_instructions = instructions_blob.download_as_text()
        logging.info("Successfully loaded master instructions from GCS.")
    except Exception as e:
        logging.warning(f"Could not load master instructions from GCS: {e}. Using default.")
        master_instructions = "You are an expert legal analyst."
    
    historical_context = _get_context_from_rag(chunk_embeddings, db_conn, project_id, vertex_ai_location, index_endpoint_id, deployed_index_id)
    analysis_schema = {"type":"object","properties":{"filing_details":{"type":"object","properties":{"case_number":{"type":"string"},"case_micro_id":{"type":"string","enum":["AS","AM","DG","AU","LA","MH","HK","MF","UNKNOWN"]},"court":{"type":"string"},"parties":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"party_codified_id":{"type":"string","enum":["AGH","CLD","LMH","PAIS","KIT","TEC","MBA","FPMG","AKD","DSM","MDM","CRM","SWM","BGM","LHM","KTX","TTA","JBA","CRA","DWA","SCC","BBP","BRD","KJM","TLN","DJC","DAD","CDM","ACM","SRM","DBM","JDP","HSC","VHC","SCD","BCD","MCD","USA","DVA","DHS","UCR","UCV","UCPD","ACI","SRD","DRS","HBT","CHP","FEMA","JCD","UNKNOWN"]},"role":{"type":"string","enum":["Plaintiff","Defendant","Other"]}},"required":["name","role","party_codified_id"]}}},"required":["case_number","court","parties","case_micro_id"]},"key_events":{"type":"array","items":{"type":"object","properties":{"event_date":{"type":"string","format":"date"},"event_description":{"type":"string"},"page_reference":{"type":"integer"}},"required":["event_date","event_description","page_reference"]}},"legal_criticism":{"type":"array","items":{"type":"string"}}},"required":["filing_details","key_events","legal_criticism"]}

    system_prompt = (
        f"# PERSONA\n{master_instructions}\n\n"
        "# TASK\nCritically analyze the legal document. Identify key details, dates, and infractions. "
        "Structure your response as a single, valid JSON object.\n\n"
        "# FORMAT\nReturn JSON matching the required schema. "
        "CRITICAL: Use the Master ID Lists for 'case_micro_id' and 'party_codified_id'. "
        "If an ID is not determinable, use 'UNKNOWN'."
    )
    user_prompt = (
        f"# CONTEXT\n--- RELEVANT CASE HISTORY ---\n{historical_context}\n--- END HISTORY ---\n\n"
        f"--- NEW DOCUMENT TO ANALYZE ---\n{document_text}\n--- END DOCUMENT ---"
    )

    analysis_text = _call_generative_model(
        "Legal Analysis", client, analysis_model_name, system_prompt, user_prompt, is_json=True, response_schema=analysis_schema
    )
    if not analysis_text:
        return None
    try:
        match = re.search(r"```json(.*?)```", analysis_text, re.DOTALL)
        json_text = match.group(1).strip() if match else analysis_text.strip()
        analysis_json = json.loads(json_text)
        return analysis_json if "legal_criticism" in analysis_json else None
    except json.JSONDecodeError:
        logging.error(f"Failed to decode model output into JSON: {analysis_text[:500]}...")
        return None

def generate_criticism_timeline(analysis: dict) -> Optional[str]:
    criticisms = analysis.get("legal_criticism", [])
    events = analysis.get("key_events", [])
    if not criticisms and not events:
        return "No criticisms or key events identified."
    timeline = ["# Criticism and Event Timeline\n"]
    if events:
        timeline.append("## Key Events\n")
        sorted_events = sorted(events, key=lambda x: x.get("event_date", "0000-00-00"))
        for event in sorted_events:
            date = event.get("event_date", "N/A")
            desc = event.get("event_description", "N/A")
            page = event.get("page_reference", "N/A")
            timeline.append(f"- {date} (Page {page}): {desc}")
    if criticisms:
        timeline.append("\n## Legal Criticisms\n")
        for criticism in criticisms:
            timeline.append(f"- {criticism}")
    return "\n".join(timeline)

def _persist_analysis_hierarchical(
    filename: str, clean_text: Optional[str], analysis: Optional[dict], timeline: Optional[str],
    db_client, db_conn, chunk_embeddings: Optional[np.ndarray], analysis_model_name: str,
    cleaning_model_name: str, case_micro_id: str, case_number: str, document_id: Optional[int]
) -> None:
    doc_id = sanitize_filename(filename)
    if clean_text or analysis or timeline:
        _log(6, 7, f"Persisting data to Firestore for {doc_id}...")
        doc_ref = db_client.collection("documents").document(doc_id)
        batch = db_client.batch()
        processing_status = "COMPLETE" if analysis else "PARTIAL"
        doc_data = {
            "source_document": filename, "processing_status": processing_status,
            "last_processed_utc": firestore.SERVER_TIMESTAMP,
            "ai_metadata": {"analysis_model": analysis_model_name, "cleaning_model": cleaning_model_name, "script_version": SCRIPT_VERSION}
        }
        if analysis:
            doc_data["filing_details"] = analysis.get("filing_details", {})
        batch.set(doc_ref, doc_data, merge=True)
        if clean_text:
            batch.set(doc_ref.collection("content").document("clean_text"), {"text": clean_text})
        if timeline:
            batch.set(doc_ref.collection("syntheses").document("criticism_timeline"), {"markdown_text": timeline})
        batch.commit()
        logging.info(f"Firestore persist complete for {doc_id}. Status: {processing_status}")
    else:
        logging.warning(f"Skipping Firestore for {filename}: no data.")

    if clean_text and clean_text.strip() and document_id:
        try:
            _log(0, 0, f"Persisting text and embeddings to Cloud SQL for {doc_id}...")
            update_document_with_content(document_id, clean_text, chunk_embeddings, db_conn)
            if chunk_embeddings is not None and len(chunk_embeddings) > 0:
                chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, length_function=len).split_text(clean_text)
                index_document_chunks(document_id, chunks, chunk_embeddings, db_conn)
            logging.info(f"Cloud SQL persist complete for {doc_id}.")
        except Exception as e:
            logging.error(f"CRITICAL: Failed to persist data to Cloud SQL for '{doc_id}': {e}", exc_info=True)
            if db_conn:
                db_conn.rollback()
            raise
    else:
        logging.info(f"Skipping Cloud SQL persistence for '{filename}' as clean text or document_id is missing.")

def _handle_processing_with_lease_lock(filename: str, mime_type: str):
    storage_client = get_storage_client()
    lock_bucket_name = os.environ.get("LOCK_BUCKET")
    if not lock_bucket_name:
        logging.error("FATAL: LOCK_BUCKET environment variable not set.")
        return
    lock_file_name = f"{sanitize_filename(filename)}.lock"
    bucket = storage_client.bucket(lock_bucket_name)
    blob = bucket.blob(lock_file_name)
    lease_acquired = False
    try:
        blob.reload()
        lock_content = blob.download_as_text()
        lock_timestamp = datetime.fromisoformat(lock_content)
        current_time = datetime.now(timezone.utc)
        elapsed_time = current_time - lock_timestamp
        if elapsed_time < timedelta(minutes=LOCK_LEASE_MINUTES):
            logging.warning(f"Lease for '{filename}' is active ({elapsed_time.total_seconds() / 60:.2f} mins ago). Skipping.")
            return
        else:
            logging.info(f"Stale lease for '{filename}' found. Attempting to acquire.")
            stale_generation = blob.generation
            new_lease_time = datetime.now(timezone.utc).isoformat()
            blob.upload_from_string(new_lease_time, if_generation_match=stale_generation)
            lease_acquired = True
    except api_exceptions.NotFound:
        logging.info(f"No active lease for '{filename}'. Attempting to acquire.")
        new_lease_time = datetime.now(timezone.utc).isoformat()
        blob.upload_from_string(new_lease_time, if_generation_match=0)
        lease_acquired = True
    except api_exceptions.PreconditionFailed:
        logging.warning(f"Failed to acquire lease for '{filename}'; another process won the race. Skipping.")
        return
    if lease_acquired:
        logging.info(f"Lease acquired for '{filename}'. Starting processing.")
        try:
            process_document_pipeline(filename=filename, mime_type=mime_type)
        finally:
            try:
                blob.reload()
                blob.delete()
                logging.info(f"Lease for '{filename}' released.")
            except Exception as e:
                logging.error(f"CRITICAL: Failed to release lease for '{filename}': {e}", exc_info=True)

@functions_framework.cloud_event
def on_cloud_event(event: CloudEvent) -> None:
    try:
        filename = event.data["name"]
        mime_type = event.data["contentType"]
        logging.info(f"Received CloudEvent for file: {filename} ({mime_type})")
        _handle_processing_with_lease_lock(filename=filename, mime_type=mime_type)
    except Exception as e:
        logging.critical(f"CRITICAL: Failed to process CloudEvent: {e}", exc_info=True)
        return

def process_document_pipeline(filename: str, mime_type: str):
    global log_buffer
    log_buffer = []
    TOTAL_STEPS = 7
    db_conn = None
    try:
        # --- Initialization Phase ---
        project_id = os.environ["PROJECT_ID"]
        input_bucket_name = os.environ["INPUT_BUCKET"]
        json_bucket_name = os.environ["JSON_BUCKET"]
        
        firestore_db = get_firestore_client(project_id)
        storage_client = get_storage_client()
        client = initialize_genai_client()
        aiplatform.init(project=project_id, location=os.environ["VERTEX_AI_LOCATION"])
        
        db_conn = get_db_connection()
        if not db_conn:
            raise ConnectionError("Failed to acquire database connection.")

        # --- Step 0: Pre-flight Checks & Reservation ---
        _log(0, TOTAL_STEPS, f"Starting pipeline for {filename}")
        
        known_ids = ['AS', 'AM', 'DG', 'AU', 'LA', 'MH', 'HK', 'MF']
        extracted_case_micro_id = "UNK"
        try:
            potential_id = filename.split('_')[1]
            if potential_id.upper() in known_ids:
                extracted_case_micro_id = potential_id.upper()
        except IndexError:
            id_pattern = r'\b(' + '|'.join(known_ids) + r')\b'
            match = re.search(id_pattern, filename, re.IGNORECASE)
            if match:
                extracted_case_micro_id = match.group(1).upper()
        
        if extracted_case_micro_id == "UNK":
            logging.warning(f"Could not determine a known case_micro_id from filename: '{filename}'. Defaulting to UNK.")

        case_id = None
        with db_conn.cursor() as cur:
            cur.execute("SELECT id FROM cases WHERE case_micro_id = %s", (extracted_case_micro_id,))
            result = cur.fetchone()
            if not result:
                logging.error(f"Case with micro_id '{extracted_case_micro_id}' not found in cases table.")
            else:
                case_id = result[0]
        
        if case_id is None:
            raise ValueError(f"Processing stopped because case_id for micro_id '{extracted_case_micro_id}' could not be resolved.")

        document_id = check_and_reserve_document(case_id, filename, db_conn)
        if document_id is None:
            _log(0, TOTAL_STEPS, f"Document already processed. Skipping.")
            return

        # --- Step 1 & 2: Document Processing & Cleaning ---
        input_gcs_uri = f"gs://{input_bucket_name}/{filename}"
        _log(1, TOTAL_STEPS, "Extracting and cleaning text with OCR and AI...")
        clean_text = get_clean_document_text(input_gcs_uri, mime_type, os.environ["DOCAI_PROCESSOR_ID"], json_bucket_name, client)
        if not clean_text:
            raise ValueError("OCR process resulted in no clean text.")
        _log(2, TOTAL_STEPS, "Text extraction and cleaning complete.")

        # --- Step 3 & 4: Embedding and Indexing ---
        _log(3, TOTAL_STEPS, "Generating text embeddings...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(clean_text)
        embeddings = None
        if chunks:
            model = _get_embedding_model()
            # Vertex AI embedding models have a limit of 5 chunks per request
            all_embeddings = []
            for i in range(0, len(chunks), 5):
                batch_chunks = chunks[i:i+5]
                response = model.get_embeddings(batch_chunks)
                all_embeddings.extend([embedding.values for embedding in response])
            embeddings = np.array(all_embeddings)
        _log(3, TOTAL_STEPS, f"Generated {len(embeddings) if embeddings is not None else 0} embeddings.")
        
        # Persist to SQL early
        _log(4, TOTAL_STEPS, "Persisting text and embeddings to Cloud SQL...")
        update_document_with_content(document_id, clean_text, embeddings, db_conn)
        if embeddings is not None and len(embeddings) > 0:
            index_document_chunks(document_id, chunks, embeddings, db_conn)
        
        # --- Step 5 & 6: Legal Analysis and Persistence ---
        analysis = generate_legal_analysis(
            clean_text, db_conn, embeddings, storage_client, os.environ["INSTRUCTIONS_BUCKET"],
            os.environ["MASTER_INSTRUCTIONS_FILE"], os.environ["ANALYSIS_MODEL"], client, project_id,
            os.environ["VERTEX_AI_LOCATION"], os.environ["INDEX_ENDPOINT_ID"], os.environ["DEPLOYED_INDEX_ID"]
        )
        
        found_case_number = "UNK"
        if analysis:
            found_case_number = analysis.get("filing_details", {}).get("case_number", "UNK")
            _log(0, 0, f"Analysis found case number: {found_case_number}. Updating record.")
            with db_conn.cursor() as cur:
                cur.execute("UPDATE documents SET case_number = %s WHERE id = %s", (found_case_number, document_id))
                db_conn.commit()

        timeline = generate_criticism_timeline(analysis) if analysis else None
        _persist_analysis_hierarchical(
            filename, clean_text, analysis, timeline, firestore_db, db_conn,
            embeddings, os.environ["ANALYSIS_MODEL"], os.environ["CLEANING_MODEL"],
            extracted_case_micro_id, found_case_number, document_id
        )
        _log(6, TOTAL_STEPS, "Firestore persistence complete.")
        
        # --- Step 7: Completion ---
        _log(7, TOTAL_STEPS, "Pipeline finished successfully.", is_final=True)
    except Exception as e:
        if db_conn:
            db_conn.rollback()
        _log(0, TOTAL_STEPS, f"Pipeline failed: {e}", is_error=True, level="CRITICAL")
        raise
    finally:
        # Global connection is intentionally not closed to be reused by the warm instance.
        pass
