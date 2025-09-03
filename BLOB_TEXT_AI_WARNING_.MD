PASSWORD-LESS_IAM.MD



This is a comprehensive guide to achieving fully passwordless authentication for Google Cloud SQL (PostgreSQL), Firestore, and Vertex AI within your Python application. This approach adheres to Google Cloud best practices, utilizing Application Default Credentials (ADC) and IAM for secure, optimized, and entirely passwordless connectivity.

### The Foundation: ADC and IAM

When your application runs on Google Cloud (e.g., Cloud Run, Cloud Functions), it assumes the identity of the attached Service Account. Google client libraries use **Application Default Credentials (ADC)** to automatically detect this identity and authenticate.

  * **Firestore, GCS, and Vertex AI:** The respective Google Cloud libraries automatically use ADC. Your existing code for these services is already passwordless.
  * **Cloud SQL:** This requires migrating to **IAM database authentication** using the **Cloud SQL Python Connector** and **SQLAlchemy** for connection pooling.

### 1\. Service Account Selection and IAM Roles

We recommend using a dedicated service account for this application. Based on the permissions provided, the most suitable account is:

**`genai-rag-run-sa-6bdb68ec@thebestever.iam.gserviceaccount.com`**

Ensure your Cloud Function or Cloud Run service is configured to execute as this identity.

#### Required Roles

This service account needs specific roles. You must add the roles marked as **Action Needed**.

| Service | Required Role(s) | Status |
| :--- | :--- | :--- |
| Cloud SQL (Connection) | `Cloud SQL Client` | ✅ Granted |
| Cloud SQL (IAM Auth) | `Cloud SQL Instance User` | ✅ Granted |
| Vertex AI (Models/Embeddings) | `Vertex AI User` | ✅ Granted |
| Firestore (Data Storage) | `Cloud Datastore User` | ❌ **Action Needed** |
| Cloud Storage (Files/Locks) | `Storage Object Admin` | ❌ **Action Needed** |
| Document AI (OCR) | `Document AI API User` | ❌ **Action Needed** |

### 2\. Cloud SQL Configuration for IAM Authentication

You must configure your Cloud SQL instance (`genai-rag-db-6bdb68ec`) before the code will work.

#### A. Enable the IAM Flag

Enable the required database flag:

```bash
gcloud sql instances patch genai-rag-db-6bdb68ec --database-flags=cloudsql.iam_authentication=on
```

#### B. Add the IAM User to the Database

Add the service account as a database user. Note that PostgreSQL typically uses the email address *without* the `.gserviceaccount.com` suffix as the username.

```bash
gcloud sql users create genai-rag-run-sa-6bdb68ec@thebestever.iam \
    --instance=genai-rag-db-6bdb68ec \
    --type=CLOUD_IAM_SERVICE_ACCOUNT
```

#### C. Grant Database Privileges

Connect to your PostgreSQL database using an admin account (e.g., `postgres`) and grant the necessary privileges:

```sql
-- Grant the specific role required for IAM login in Cloud SQL Postgres
GRANT pg_cloudsql_iam_users TO "genai-rag-run-sa-6bdb68ec@thebestever.iam";

-- Grant necessary privileges on your application database (adjust as needed)
GRANT CONNECT ON DATABASE postgres TO "genai-rag-run-sa-6bdb68ec@thebestever.iam";
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "genai-rag-run-sa-6bdb68ec@thebestever.iam";
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "genai-rag-run-sa-6bdb68ec@thebestever.iam";
```

### 3\. Dependencies

We will use `SQLAlchemy` for pooling, the `Cloud SQL Python Connector`, and `psycopg` (psycopg3) as the underlying driver, which integrates well with `pgvector`.

Update your `requirements.txt` (remove `psycopg` if listed separately):

```text
functions-framework
numpy
langchain-text-splitters
google-cloud-aiplatform
google-cloud-documentai
google-cloud-firestore
google-cloud-storage
cloudevents
google-generativeai
vertexai
# New dependencies for passwordless, pooled SQL with pgvector support
SQLAlchemy>=2.0
google-cloud-sql-python-connector[psycopg]
pgvector
```

### 4\. Revised Code (`main.py`)

The following is the complete, optimized, and passwordless implementation.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py - Version 18.0.0 (Passwordless, Pooled, Optimized)
import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any
from threading import Lock

import functions_framework
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.api_core import exceptions as api_exceptions
from google.api_core.client_options import ClientOptions
import google.cloud.aiplatform as aiplatform # Used for Vector Search Endpoint
# Import the modern Vertex AI SDK for optimized embeddings
import vertexai
from vertexai.language_models import TextEmbeddingModel

from google.cloud import documentai as docai_v1
from google.cloud import firestore
from google.cloud import storage
from cloudevents.http import CloudEvent
import google.generativeai as genai # Modernized GenAI import

# New Imports for Passwordless SQL (IAM Auth via Connector + SQLAlchemy + psycopg)
try:
    from sqlalchemy import create_engine, text, exc as sqlalchemy_exc, event
    from sqlalchemy.engine import Engine
    from google.cloud.sql.connector import Connector, IPTypes
    import psycopg
    # Import pgvector for registration
    from pgvector.sqlalchemy import Vector
    DB_LIBS_AVAILABLE = True
except ImportError as e:
    DB_LIBS_AVAILABLE = False
    Engine = Any # Type hint fallback
    Vector = None
    logging.warning(f"Database libraries missing ({e}). DB functionality will be disabled.")


SCRIPT_VERSION = "18.0.0"
LOCK_LEASE_MINUTES = 20

# --- Environment Configuration ---
os.environ.setdefault("PROJECT_ID", "thebestever")
os.environ.setdefault("VERTEX_AI_LOCATION", "us-central1")
# ... (Other environment variables remain the same) ...
os.environ.setdefault("DOCAI_PROCESSOR_ID", "6e8f23fa5796a22b")
os.environ.setdefault("INDEX_ENDPOINT_ID", "556724518584844288")
os.environ.setdefault("DEPLOYED_INDEX_ID", "analysis_1756251260790")
os.environ.setdefault("MASTER_INSTRUCTIONS_FILE", "master_instructions.txt")

# Optional DB environment
# DB_USER/DB_PASSWORD removed.
# NEW: The IAM Principal (SA email) is required by the Connector/psycopg for IAM auth.
os.environ.setdefault("DB_IAM_PRINCIPAL", "genai-rag-run-sa-6bdb68ec@thebestever.iam.gserviceaccount.com")
os.environ.setdefault("DB_NAME", "postgres")
os.environ.setdefault("INSTANCE_CONNECTION_NAME", "thebestever:us-central1:genai-rag-db-6bdb68ec")
os.environ.setdefault("ENABLE_DB", "0")
os.environ.setdefault("USE_PRIVATE_IP", "0") # Set to 1 if using VPC connector

# Model defaults
os.environ.setdefault("ANALYSIS_MODEL", "gemini-1.5-pro")
os.environ.setdefault("CLEANING_MODEL", "gemini-1.5-flash")
os.environ.setdefault("EMBEDDING_MODEL", "textembedding-gecko@003")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_buffer = []

# --- Global, Thread-Safe Clients and Connection Pools for Reuse ---
db_engine: Optional[Engine] = None
db_lock = Lock()
genai_client: Optional[genai.GenerativeModel] = None
genai_client_lock = Lock()
connector: Optional[Connector] = None
connector_lock = Lock()

# --- Client Initialization (All Passwordless via ADC) ---

def get_firestore_client(project_id: str, database: str = "(default)") -> firestore.Client:
    return firestore.Client(project=project_id, database=database)

def get_storage_client() -> storage.Client:
    return storage.Client()

def get_docai_client(location: str) -> docai_v1.DocumentProcessorServiceClient:
    client_options = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    return docai_v1.DocumentProcessorServiceClient(client_options=client_options)

def initialize_genai_client() -> genai.GenerativeModel:
    """Initializes reusable GenerativeModel using Vertex AI (Passwordless)."""
    global genai_client
    with genai_client_lock:
        if genai_client is None:
            project = os.environ.get("PROJECT_ID")
            location = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
            try:
                # Initialize the library for Vertex usage (enforces IAM)
                logging.info("Client using IAM (ADC) auth via Vertex endpoint.")
                genai.configure(vertex_ai=True, project=project, location=location)
                # Create a model instance which acts as the client interface
                genai_client = genai.GenerativeModel(model_name=os.environ["ANALYSIS_MODEL"])
            except Exception as e:
                logging.error(f"Failed to initialize AI client: {e}")
                raise
        else:
            logging.info("Reusing existing AI client.")
    return genai_client

def get_connector() -> Connector:
    """Initializes and returns a reusable Cloud SQL Connector."""
    global connector
    with connector_lock:
        if connector is None:
            if not DB_LIBS_AVAILABLE:
                raise RuntimeError("DB libraries not available; cannot initialize Connector.")
            logging.info("Initializing Cloud SQL Connector.")
            connector = Connector()
    return connector

def get_db_engine() -> Optional[Engine]:
    """Initializes and returns a SQLAlchemy connection pool using the Cloud SQL Connector for IAM Auth."""
    if os.environ.get("ENABLE_DB") != "1" or not DB_LIBS_AVAILABLE:
        return None

    global db_engine
    with db_lock:
        if db_engine is not None:
            return db_engine

        logging.info("Initializing new DB connection pool with IAM Auth...")
        
        instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]
        db_name = os.environ["DB_NAME"]
        # When using psycopg with IAM Auth, the user MUST be the IAM Principal (SA email)
        db_user = os.environ["DB_IAM_PRINCIPAL"]

        sql_connector = get_connector()
        ip_type = IPTypes.PRIVATE if os.environ.get("USE_PRIVATE_IP") == "1" else IPTypes.PUBLIC

        def getconn():
            # enable_iam_auth=True is the key to passwordless connection
            conn = sql_connector.connect(
                instance_connection_name,
                "psycopg",
                user=db_user,
                db=db_name,
                ip_type=ip_type,
                enable_iam_auth=True,
            )
            return conn

        # Create SQLAlchemy Engine Pool
        db_engine = create_engine(
            "postgresql+psycopg://",
            creator=getconn,
            pool_size=5,
            max_overflow=2,
            pool_timeout=30,
            pool_recycle=1800, # Recycle connections to refresh tokens
        )

        # CRITICAL: Register pgvector type handler for psycopg/SQLAlchemy
        if Vector:
            @event.listens_for(db_engine, "connect")
            def register_vector_type(dbapi_connection, connection_record):
                logging.debug("Registering pgvector adapter on new connection.")
                # Ensure the extension exists (optional, but good practice)
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    dbapi_connection.commit()
                except Exception:
                    dbapi_connection.rollback()
                finally:
                    cursor.close()
                # Register the adapter
                psycopg.adapt.register_adapter(Vector, Vector.to_db)

        logging.info(f"DB connection pool established. ID: {id(db_engine)}")
        return db_engine

# --- Utility Functions ---
# (_log and sanitize_filename implementations remain the same as input)
def _log(step: int, total_steps: int, message: str, is_error: bool = False, is_final: bool = False, level: str = "INFO") -> None:
    global log_buffer
    # (Logging implementation omitted for brevity)
    print(f"LOG: {message}")

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[^a-zA-Z0-9.-]', '_', filename)

# --- Database Interaction Functions (Rewritten for SQLAlchemy) ---

def check_and_reserve_document(case_id: int, document_name: str, db_engine: Engine) -> Optional[int]:
    """Checks if a document exists and reserves it if not, using SQLAlchemy transaction."""
    try:
        # Use engine.begin() for automatic commit/rollback transaction management
        with db_engine.begin() as conn:
            stmt = text("""
                INSERT INTO documents (case_id, original_filename, processed_at)
                VALUES (:case_id, :filename, NOW())
                ON CONFLICT (original_filename) DO NOTHING
                RETURNING id;
            """)
            # Use named parameters (:name)
            result = conn.execute(stmt, {"case_id": case_id, "filename": document_name})
            document_id = result.scalar()

            if document_id is None:
                logging.warning(f"Document '{document_name}' already exists. Skipping reservation.")
                return None
            
            # Commits automatically upon exiting the 'with' block
            return document_id
            
    except sqlalchemy_exc.SQLAlchemyError as e:
        logging.error(f"DB error during document reservation: {e}")
        # Rolls back automatically
        return None

def update_document_with_content(document_id: Optional[int], document_text: str, chunk_embeddings: Optional[np.ndarray], db_engine: Optional[Engine]) -> None:
    """Persist full text + mean embedding if DB available."""
    if not db_engine or document_id is None:
        return
    
    try:
        embedding = None
        if chunk_embeddings is not None and len(chunk_embeddings) > 0:
            # Calculate mean embedding (pgvector/psycopg handles list conversion)
            embedding = np.mean(chunk_embeddings, axis=0).tolist()

        stmt = text("""
            UPDATE documents
            SET full_text = :text, full_text_embedding = :embedding, processed_at = NOW()
            WHERE id = :doc_id;
        """)
        
        with db_engine.begin() as conn:
            conn.execute(stmt, {"text": document_text, "embedding": embedding, "doc_id": document_id})
            # Commits automatically

    except sqlalchemy_exc.SQLAlchemyError as e:
        logging.error(f"Failed to update document content (id={document_id}): {e}")
        # Rolls back automatically
        raise


# --- AI Model Interaction (Updated for newer SDK patterns) ---

def _call_ai_model(
    task_name: str,
    client: genai.GenerativeModel,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    is_json: bool = False,
    response_schema: Optional[Dict] = None,
) -> Optional[str]:
    """Wrapper for model.generate_content using IAM auth."""
    logging.info(f"Calling AI model for: {task_name} (Model: {model_name})")
    
    # Determine the correct model instance to use, applying system instructions
    # Note: client._model_name includes 'models/' prefix.
    if f"models/{model_name}" != client._model_name:
        logging.info(f"Switching model context to {model_name} for this task.")
        model = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)
    else:
        model = client
        # Ensure system instruction is set if reusing the client (if different)
        if system_prompt and model._system_instruction != system_prompt:
             model = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)

    try:
        # Define generation configuration
        generation_config = {}
        if is_json:
            # Use response_mime_type for reliable JSON output
            generation_config["response_mime_type"] = "application/json"
            if response_schema:
                 generation_config["response_schema"] = response_schema

        response = model.generate_content(
            user_prompt,
            generation_config=generation_config if generation_config else None
        )
        
        if hasattr(response, 'text') and response.text:
            return response.text
        return None
    except Exception as e:
        logging.error(f"Error calling AI model for {task_name}: {e}", exc_info=True)
        return None

def _get_context_from_rag(query_embedding: Optional[np.ndarray], db_engine: Optional[Engine], project_id: str, vertex_ai_location: str, index_endpoint_id: str, deployed_index_id: str) -> str:
    if db_engine is None:
        return "Historical context disabled (DB off)."
    if query_embedding is None or query_embedding.size == 0:
        return "No relevant historical context found (Embedding missing)."
    
    _log(0, 0, "Retrieving context from RAG (Vertex AI Vector Search)...")
    try:
        query_vec = np.mean(query_embedding, axis=0).tolist() if query_embedding.ndim > 1 else query_embedding.tolist()
        
        # Vertex AI Vector Search interaction (Passwordless/ADC)
        # We use the aiplatform SDK for MatchingEngineIndexEndpoint
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            f"projects/{project_id}/locations/{vertex_ai_location}/indexEndpoints/{index_endpoint_id}"
        )
        response = index_endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[query_vec],
            num_neighbors=5
        )
        
        if not response or not response[0]:
            return "No relevant historical context found in Vector Search."

        neighbor_ids = [int(n.id) for n in response[0] if n.id and n.id.isdigit()]
        if not neighbor_ids:
            return "No relevant historical context found (no valid IDs)."

        # Retrieve text from DB using SQLAlchemy
        # NOTE: Assuming 'embedding' column uses pgvector and '<=>' (cosine distance) operator is available
        stmt = text("""
            SELECT chunk_text, (embedding <=> :query_vec) AS distance 
            FROM document_chunks 
            WHERE id = ANY(:neighbor_ids) 
            ORDER BY distance ASC
        """)
        
        with db_engine.connect() as conn:
            # The registered pgvector adapter handles the conversion
            results = conn.execute(stmt, {"query_vec": query_vec, "neighbor_ids": neighbor_ids}).fetchall()

        # Access results using index
        context = [f"- (Distance: {row[1]:.4f}): {row[0]}" for row in results]
        return "\n".join(context) if context else "No relevant historical context found."

    except Exception as e:
        logging.error(f"Error during RAG retrieval: {e}", exc_info=True)
        return "Error retrieving historical context."

# --- Optimized Embedding / Chunking Utilities ---
def _generate_and_store_embeddings(document_id: Optional[int], text: str, db_engine: Optional[Engine], embedding_model_name: str) -> Optional[np.ndarray]:
    """Optimized: Split text, embed with Gecko (vertexai SDK), and perform a single bulk insert."""
    try:
        embedding_model_name = os.environ.get("TEXT_EMBED_MODEL") or embedding_model_name
        # Gecko models support up to 250 texts per request
        API_BATCH_SIZE = 250 

        # 1. Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        chunks = splitter.split_text(text)
        if not chunks:
            return None

        # 2. Load Model (Uses the modern vertexai SDK - Passwordless/ADC)
        try:
            # vertexai.init() must be called beforehand (handled in main pipeline)
            model = TextEmbeddingModel.from_pretrained(embedding_model_name)
        except Exception as e:
            logging.error(f"Failed to load embedding model '{embedding_model_name}': {e}")
            return None

        # 3. Generate Embeddings (Batched API Calls)
        embeddings_list = []
        chunk_data_to_insert = []
        can_persist_chunks = bool(db_engine) and document_id is not None

        logging.info(f"Generating embeddings for {len(chunks)} chunks (Batch size: {API_BATCH_SIZE})...")

        for i in range(0, len(chunks), API_BATCH_SIZE):
            batch_texts = chunks[i:i+API_BATCH_SIZE]
            try:
                # Optimized, batched API call
                responses = model.get_embeddings(batch_texts)
            except Exception as e:
                logging.error(f"Embedding batch failed (indexes {i}-{i+len(batch_texts)-1}): {e}")
                continue

            for j, response in enumerate(responses):
                vec = response.values
                if not vec:
                    continue
                
                embeddings_list.append(vec)
                
                if can_persist_chunks:
                    chunk_data_to_insert.append({
                        "document_id": document_id,
                        "chunk_index": i + j,
                        "chunk_text": batch_texts[j],
                        "embedding": vec # pgvector/psycopg handles the conversion
                    })

        if not embeddings_list:
            return None
            
        # 4. Optimized: Bulk insert using SQLAlchemy in a single transaction
        if can_persist_chunks and chunk_data_to_insert:
            logging.info(f"Bulk inserting {len(chunk_data_to_insert)} chunks into Cloud SQL...")
            try:
                stmt = text("""
                    INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding)
                    VALUES (:document_id, :chunk_index, :chunk_text, :embedding)
                    ON CONFLICT DO NOTHING;
                """)
                # Use a transaction block for the bulk insert
                with db_engine.begin() as conn:
                    # SQLAlchemy executes this efficiently (executemany)
                    conn.execute(stmt, chunk_data_to_insert)
                # Commits automatically
            except sqlalchemy_exc.SQLAlchemyError as db_err:
                logging.error(f"Failed to batch insert chunks: {db_err}")
                # Rolls back automatically
                # Continue to return embeddings even if DB persistence fails

        arr = np.array(embeddings_list, dtype=float)
        logging.info(f"Generated {arr.shape[0]} embeddings (model={embedding_model_name}).")
        return arr
        
    except Exception as e:
        logging.error(f"Unexpected error during embedding generation: {e}", exc_info=True)
        return None

# --- Document Processing and Analysis Functions ---
# (These remain largely the same, relying on the passwordless clients/connections established above.
# Full implementations omitted for brevity but assumed present based on the input script.)

def get_clean_document_text(input_file: str, mime_type: str, processor_id: str, temp_bucket: str, client: genai.GenerativeModel) -> Optional[str]:
    # Implementation uses DocAI and GCS clients via ADC.
    # Refer to the original script for the full implementation.
    pass 

def _ai_text_restoration(text: str, client: genai.GenerativeModel) -> str:
    # Implementation uses GenAI client via ADC.
    pass

def generate_legal_analysis(
    document_text: str,
    db_engine: Optional[Engine], # Updated parameter type
    chunk_embeddings: Optional[np.ndarray],
    # ... other parameters
) -> Optional[dict]:
    # Implementation uses GCS, RAG (SQL+Vector Search), and GenAI via ADC.
    # Ensure _get_context_from_rag is called with db_engine.
    pass

def generate_criticism_timeline(analysis: dict) -> Optional[str]:
    # Implementation is pure logic.
    pass

def _persist_analysis_hierarchical(
    filename: str, clean_text: Optional[str], analysis: Optional[dict], timeline: Optional[str],
    db_client, db_engine: Optional[Engine], chunk_embeddings: Optional[np.ndarray], 
    # ... other parameters
) -> None:
    # Implementation uses Firestore (ADC) and Cloud SQL (IAM Auth).
    # Ensure update_document_with_content is called with db_engine.
    pass

# --- Entry Points and Pipeline Logic ---

def _handle_processing_with_lease_lock(filename: str, mime_type: str):
    # Implementation uses GCS via ADC.
    pass

@functions_framework.cloud_event
def on_cloud_event(event: CloudEvent) -> None:
    # Main entry point.
    pass

def process_document_pipeline(filename: str, mime_type: str):
    """Main document processing pipeline."""
    global log_buffer
    log_buffer = []
    TOTAL_STEPS = 7
    enable_db = os.environ.get("ENABLE_DB") == "1"
    local_engine: Optional[Engine] = None
    document_id: Optional[int] = None

    try:
        # --- Initialization (All Passwordless) ---
        project_id = os.environ["PROJECT_ID"]
        # ...
        
        firestore_db = get_firestore_client(project_id)
        storage_client = get_storage_client()
        client = initialize_genai_client()
        
        # Initialize both SDKs for Vertex AI access
        aiplatform.init(project=project_id, location=os.environ["VERTEX_AI_LOCATION"])
        vertexai.init(project=project_id, location=os.environ["VERTEX_AI_LOCATION"])

        if enable_db:
            # Get the connection pool engine
            local_engine = get_db_engine()
            if local_engine is None:
                logging.error("DB initialization failed; proceeding statelessly.")
                enable_db = False

        _log(0, TOTAL_STEPS, f"Starting pipeline for {filename} (db={enable_db})")

        # --- Case ID Extraction ---
        # ... (Logic remains the same)

        # --- DB Reservation (Updated for SQLAlchemy) ---
        case_id = None
        if enable_db and local_engine:
            try:
                # Case Lookup using SQLAlchemy connection
                stmt = text("SELECT id FROM cases WHERE case_micro_id = :micro_id")
                with local_engine.connect() as conn:
                    result = conn.execute(stmt, {"micro_id": extracted_case_micro_id})
                    case_id = result.scalar()
            except Exception as e:
                logging.error(f"Case lookup failed; disabling DB for this run: {e}")
                enable_db = False
                local_engine = None

        if enable_db and local_engine and case_id is not None:
            # Pass the engine
            document_id = check_and_reserve_document(case_id, filename, local_engine)
            if document_id is None:
                _log(0, TOTAL_STEPS, "Document already processed or reservation failed. Skipping.")
                return
        else:
            logging.info("Skipping DB reservation (stateless or lookup failure).")

        # --- Steps 1-7: Processing (OCR, Embeddings, Analysis, Persistence) ---
        # ... (Pipeline steps orchestrated here, passing local_engine where db_conn was previously used)
        
        # Example: Step 3 (Embeddings)
        _log(3, TOTAL_STEPS, "Generating embeddings and storing chunks...")
        embeddings = _generate_and_store_embeddings(
            document_id if enable_db else None, 
            clean_text, 
            local_engine if enable_db else None, 
            os.environ["EMBEDDING_MODEL"]
        )
        # ...

        _log(7, TOTAL_STEPS, "Pipeline finished successfully.", is_final=True)

    except Exception as e:
        # Errors are logged. Rollbacks are handled automatically by SQLAlchemy transaction blocks (engine.begin()).
        _log(0, TOTAL_STEPS, f"Pipeline failed: {e}", is_error=True, level="CRITICAL")
        raise
    finally:
        # The engine remains open globally for reuse in subsequent invocations.
        pass

# NOTE: You must manually reintegrate the full code for functions marked with 'pass' or '...' 
# from your original script into this structure, ensuring the structural changes (imports, initialization, 
# SQLAlchemy usage, modern AI SDK calls) are maintained.
```





# Guide to Secure, Passwordless IAM Authentication for Google Cloud SQL, Firestore, and Vertex AI (Gecko Embeddings)

---

In the modern Google Cloud landscape, passwordless authentication—anchored on fine-grained, auditable IAM (Identity and Access Management) controls—is the gold standard for secure service integration. By delegating authentication and access control to IAM and leveraging service accounts, organizations reduce operational risk, eliminate password management overhead, and gain clearer audit trails across application layers. This technical guide provides a detailed, up-to-date blueprint for configuring passwordless (IAM-based) authentication in three widely-used Google Cloud Platform (GCP) services: Cloud SQL, Firestore, and Vertex AI (focusing on the Gecko embeddings model). Each section covers service account selection, optimal role/permission assignment, and fully annotated Python configuration/integration examples designed to minimize code complexity.

**Assumptions**:  
- Reader has multiple service accounts available and is open to using separate accounts per service for principle-of-least-privilege isolation.
- Integration code is primarily in Python.
- The highest priorities are minimizing password/code complexity, clear role separation, and strict avoidance of retry/idempotency pitfalls.
- The current stack is already partially configured (referenced in `main.py`).

For each service, tables summarize IAM roles and permissions, immediately followed by in-depth prose explaining best practices, common pitfalls, and optimized integration patterns.

---

## Cloud SQL: Passwordless Authentication with IAM

### Overview and Rationale

Traditional database access models—rooted in static usernames and passwords—create persistent, hard-to-rotate credential risks and blind spots in your security posture. With Cloud SQL (PostgreSQL, MySQL), enabling IAM Database Authentication ensures that only callers with dynamically minted, short-lived credentials can access databases, which vastly reduces the attack surface, strengthens the identity chain for auditability, and simplifies service lifecycle management. This is especially beneficial when multiple environments or microservices (each with distinct service accounts) need distinct access scopes, as you eliminate the sprawl of hardcoded secrets and enable granular, revocable access control.

---

### IAM Roles and Permissions

| Service Account Scope             | Required IAM Roles                  | Essential Permissions                                                     | Notes                                                                                     |
|-----------------------------------|-------------------------------------|----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| For Cloud SQL connection/auth     | `roles/cloudsql.client`             | `cloudsql.instances.connect`, `cloudsql.instances.get`                      | Enables network-level connectivity via Auth Proxy or connector                            |
| To login as a DB user (IAM auth)  | `roles/cloudsql.instanceUser`       | `cloudsql.instances.login`, `cloudsql.instances.get`                        | Also requires adding the principal as an IAM database user in the DB instance             |
| For API/instance management       | `roles/cloudsql.admin` (optional)   | Full admin privileges on Cloud SQL resources                                | Only grant to operational/automation accounts, not for application/service logic          |

Cloud SQL Service Account Role Table (synthesized from GCP documentation and recent releases):

| GCP IAM Role                   | Use Case                                                  | Granularity       |
|-------------------------------|-----------------------------------------------------------|-------------------|
| `roles/cloudsql.client`        | Application authenticating via Proxy or Connector         | Least privilege   |
| `roles/cloudsql.instanceUser`  | Account acting as database user with IAM DB authentication| Database identity |
| `roles/cloudsql.admin`         | Lifecycle/admin management (e.g., backup/restore, users)  | Admin-only        |

#### Elaboration

**Assigning multiple service accounts**:  
For maximum separation—e.g., if a single application talks to more than one database or Cloud SQL instance, or you want to restrict SQL write/read capabilities on a per-service level—create separate service accounts per logical function and assign only the minimal roles needed for each.  
**Never re-use higher-privileged service accounts across untrusted code paths**:  
Service accounts with `cloudsql.admin` on production must never be used by test/dev or transient scripts.

---

### Enabling Cloud SQL IAM Database Authentication

**Step 1: Enable IAM Database Authentication on the Instance**

- Via gcloud CLI:
  ```
  gcloud sql instances patch INSTANCE_NAME --enable-iam-authentication
  ```
- Or via GCP Console:
  - Navigate to your SQL instance → Edit → Enable "IAM database authentication" → Save.

**Step 2: Add IAM-Based Database User**

- For every service account needing DB login, add the user:
  ```
  gcloud sql users create USER_EMAIL \
    --instance=[INSTANCE_NAME] \
    --type=CLOUD_IAM_SERVICE_ACCOUNT
  ```
  - For code/Proxy, the database username is the service account's email **minus** the `.gserviceaccount.com` suffix.  
  - For example, `svc-mlapp@yourproj.iam.gserviceaccount.com` → `svc-mlapp@yourproj.iam`.

- For users (optional):
  ```
  gcloud sql users create USER_EMAIL \
    --instance=[INSTANCE_NAME] \
    --type=CLOUD_IAM_USER
  ```

**Step 3: Grant Required IAM Roles**

- E.g., to give a service account network connectivity and DB login:
  ```
  gcloud projects add-iam-policy-binding PROJECT_ID \
      --member="serviceAccount:svc-mlapp@yourproj.iam.gserviceaccount.com" \
      --role="roles/cloudsql.client"

  gcloud projects add-iam-policy-binding PROJECT_ID \
      --member="serviceAccount:svc-mlapp@yourproj.iam.gserviceaccount.com" \
      --role="roles/cloudsql.instanceUser"
  ```

  Alternatively, use resource-level policies if you want finer scoping.

**Step 4: Grant Database Privileges**  
By default, newly added IAM users have no DB privileges. Connect as a DB admin and run e.g., (for PostgreSQL):

```sql
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "svc-mlapp@yourproj.iam";
```
Change role and schema as needed.

---

### Integration Syntax: Passwordless Python Connection

**Best-practice: Use Google-Managed Cloud SQL Python Connector**  
As of mid-2025, [cloud-sql-python-connector](https://pypi.org/project/cloud-sql-python-connector/) provides the cleanest IAM-auth integration, fully compatible with native SQLAlchemy and async drivers. It handles underlying IAM credential wiring and rotates short-lived DB passwords (actually OAuth tokens) automatically, eliminating password code and retry/idempotency issues.  
**The older Cloud SQL Auth Proxy (v2.x+) is robust for non-Python stacks or environments where code modification isn't practical, e.g., legacy apps or as a sidecar/container in Kubernetes.** Both methods are shown.

#### Python Example with Cloud SQL Python Connector (recommended):

Install:
```sh
pip install "cloud-sql-python-connector[pg8000,sqlalchemy]"
```

Sample code for PostgreSQL with IAM DB Auth (replace placeholders as needed):

```python
from google.cloud.sql.connector import Connector
import sqlalchemy

# Initialize Connector (auto-picks up credentials from environment/service account)
with Connector() as connector:
    def getconn():
        return connector.connect(
            "project:region:instance",
            "pg8000",
            user="svc-mlapp@yourproj.iam",         # exact user as in DB
            db="YOUR_DB_NAME",
            enable_iam_auth=True
        )

    engine = sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn)

    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("SELECT now()"))
        print(result.fetchall())
```
**No passwords are managed.**  
**No connection retry/backoff logic is needed—the connector handles short-lived token refresh.**  
Multiple connections can be pooled as needed.

##### Key Optimizations
- **No need for manual password/token rotation.**
- **No need to build explicit retry or refresh-credentials logic—Connector handles token delegation.**
- If using async, Connector's async mode is supported (e.g., with asyncpg).

#### Alternative: Cloud SQL Auth Proxy (Container/Sidecar/CLI)

1. Download Proxy and run with IAM automatic auth:
   ```
   ./cloud-sql-proxy --auto-iam-authn project:region:instance
   ```
2. In application, connect to DB via localhost, username as `svc-mlapp@yourproj.iam`, no password needed:
   ```sh
   psql -h 127.0.0.1 -U svc-mlapp@yourproj.iam yourdbname
   ```

**Python Example (with Proxy running):**
```python
import psycopg2

conn = psycopg2.connect(
    dbname="YOUR_DB",
    user="svc-mlapp@yourproj.iam",
    host="127.0.0.1",
    port=5432
)
# Use as usual.
```
**Proxy modes**: as a CLI, systemd unit, Kubernetes sidecar, or Docker container for maximum portability.

#### Avoiding Retrying/idempotency Concerns

- When using the Python Connector or Proxy, handling of credential expiry/retries is managed by the library; connection pooling handles refresh seamlessly.
- **Don't build custom code that tries to "auto-retry failed connects and replay queries" yourself.** This can cause duplicate DB operations. Use connection pooling and let the driver raise on unrecoverable errors—the library will transparently refresh IAM tokens as needed.

---

## Firestore: Secure, Passwordless Server-Side IAM Access

### Firestore IAM Integration Overview

Firestore is heavily integrated with IAM for all *server-to-server* access (Cloud Functions, server apps, etc.). Unlike frontend/mobile SDKs, which authenticate users (with Firebase Auth) and are filtered through Firestore Security Rules, **server-side Firestore SDKs (such as the Python Admin SDK) bypass security rules entirely** and rely solely on IAM permissions assigned to the executing service account.  
Thus, Firestore access control is determined by which GCP service account is running the server process, and what IAM roles have been granted to that account. This removes all password logic from your code—no Firebase API keys or user passwords required.

---

### Service Account and IAM Roles for Firestore

| Role Name                    | Typical Permissions                        | Use for...                               |
|------------------------------|--------------------------------------------|------------------------------------------|
| `roles/datastore.user`       | Read/write access to Firestore data        | Most application service accounts        |
| `roles/datastore.viewer`     | Read-only access                           | Monitoring/analytics jobs, dashboards    |
| `roles/datastore.owner`      | Full administrative control                | Rare, only for ops automation            |
| `roles/datastore.indexAdmin` | Index definition management                | Index/build automation                   |

For reference, Firestore (in Native mode) maps backend access to Datastore permissions and roles; roles grant sets of permissions detailed extensively in GCP docs.

#### Best practices:
- **Assign a unique GCP service account to each application/service with the minimum required role.**
- **Prefer `roles/datastore.user` for most trusted backend applications**; avoid over-granting `owner`/`admin` unless necessary.

---

### Granting Service Account Access

**Step 1: Grant IAM Role**

- For `roles/datastore.user`:
  ```
  gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:svc-app@YOUR_PROJECT.iam.gserviceaccount.com" \
    --role="roles/datastore.user"
  ```

**Step 2: Use Service Account for Authentication**

- Compute Engine/Cloud Run/Cloud Functions/GKE: Set the service account as the runtime identity for service deployment.
- Local development: Download a service account key JSON, set `GOOGLE_APPLICATION_CREDENTIALS` env var.

---

### Python Integration: Using the Admin SDK with IAM

**No code-level password logic, no retry/idempotency logic needed.**

```python
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.ApplicationDefault()  # auto-detects environment
firebase_admin.initialize_app(cred)

db = firestore.client()
# Read/write as needed
doc_ref = db.collection('users').document('alice')
doc_ref.set({"score": 100})

# Query example
for doc in db.collection('users').where('score', '>', 50).stream():
    print(doc.id, doc.to_dict())
```
- If running on GCP, credentials are resolved via ADC (Application Default Credentials).
- If running locally, set `GOOGLE_APPLICATION_CREDENTIALS` to the downloaded service account key.

#### Avoiding Custom Retry/idempotency Code

- The Firestore Admin SDK already implements retry/backoff for transient errors and is idempotent on item updates, provided your app logic avoids the "read-modify-write" race pattern.
- **Write small, simple logic; let the SDK handle retries where safe.**
- For critical cross-service operations (e.g., function triggers), consider Firestore transactions to enforce idempotency.

---

### Notes on Firestore Security

- **Google Cloud service account-based access can only be controlled through IAM roles, not Firestore security rules** (rules are for end-user Firebase access, not Admin SDK/server-to-server).
- **Auditing**: All access via GCP service accounts is centrally auditable in Cloud Audit Logs.

---

## Vertex AI Gecko Embeddings: IAM-Based Access for Embedding APIs

### Vertex AI Model and Embedding IAM Overview

Vertex AI supports a wide range of Foundation Models (text, embedding, generation) available with passwordless IAM authentication via service accounts and roles. The Gecko embedding family is optimized for high-quality text representation and is invoked through the Vertex AI API (Python SDK or REST).  
**Authentication is strictly handled through GCP's IAM system, so you need only set up a properly-scoped service account, with no passwords or API tokens in your codebase**.

---

### Vertex AI: IAM Roles/Permissions Matrix

| Role Name                      | IAM Role ID              | Notes                                         |
|--------------------------------|--------------------------|-----------------------------------------------|
| Vertex AI User                 | `roles/aiplatform.user`  | **Minimum required for inference/prediction** |
| Vertex AI Service Agent        | `roles/aiplatform.serviceAgent` | Used internally for resource ops          |
| Vertex AI Administrator        | `roles/aiplatform.admin` | For model mgmt and full platform control      |

**For ML API and embedding usage, `roles/aiplatform.user` is almost always sufficient**.

#### Assignment

```
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:svc-vai@YOUR_PROJECT.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

---

### Python SDK Integration: Gecko Embedding Model

**Primary library**: `google-cloud-aiplatform` (part of the Vertex AI SDK for Python)

#### Step 1: Install Vertex AI Python SDK
```sh
pip install google-cloud-aiplatform
```

#### Step 2: Grant Service Account as per above, and authenticate (by default, via ADC).

#### Step 3: Minimal, Optimized Example for Text Embeddings (Gecko)

```python
import vertexai
from vertexai.language_models import TextEmbeddingModel

vertexai.init(project="YOUR_PROJECT_ID", location="us-central1")

model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

# Get embeddings for a list of documents (no password, IAM handles auth)
embeddings = model.get_embeddings([
    "This is the first document.",
    "And this is a second one."
])
for emb in embeddings:
    print(emb.values)  # array of floats
```
- No credentials are passed in code unless for local dev (use `GOOGLE_APPLICATION_CREDENTIALS`).
- No password or API key logic anywhere.

**All credential handling is done by the Vertex AI Python SDK, which automatically uses GCP's built-in retry/idempotency and error handling logic**.

---

#### Model Versioning and Regionality

- Always use the latest suitable Gecko model: As of 2025, `textembedding-gecko@003` (and successors) are supported.
- Place your models/services in a supported location (e.g., `us-central1`).
- Model usage is via short-lived OAuth tokens issued by IAM, auto-handled in the SDK.

#### Avoid API retries/idempotency Pitfalls

- The Vertex AI Python SDK implements efficient retries and backoff.
- Your code should NOT auto-retry failed requests unless you know that the operation is idempotent (embedding generation is always idempotent).
- When running inside GCP (Cloud Run, GKE, etc.), always use the built-in service account identity.

#### Advanced Usage: REST API Example (for other languages or direct HTTP)

```sh
export ACCESS_TOKEN=$(gcloud auth print-access-token)
curl -X POST \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{
        "instances": [{"content": "text to embed"}]
      }' \
  https://us-central1-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT_ID/locations/us-central1/publishers/google/models/textembedding-gecko@003:predict
```

---

### Service Account Selection Best Practices

- **Use a dedicated service account for all Vertex AI model inference tasks** (e.g., one per microservice).
- **Assign only `roles/aiplatform.user` unless you also need to manage models (in which case, step up to `admin`).**
- **Avoid sharing the same service account with other workloads—this ensures clear audit trails and simpler least-privilege management.**
  
---

## Cross-Cutting Tips: Reducing Code and Logic Complexity

### Using Separate Service Accounts

- **Principle-of-least-privilege**: Each logical service gets its own service account with only those permissions required to interact with the specific GCP resource.
- **Auditability**: Different service accounts provide unambiguous audit trails for each service/component.
- **Zero passwords or API keys**: All authentication is via service account-based JWT/OAuth2, handled internally by GCP managed runtime or via ADC in local dev.

### Service Account Activation/Switching

- For CLI or development:
  ```sh
  gcloud auth application-default login
  # or
  gcloud auth activate-service-account --key-file=...
  ```
  Use ADC; set `GOOGLE_APPLICATION_CREDENTIALS` as needed.
- In production (GKE, Cloud Run): assign the correct service account at deployment time.

### Table: Summary of IAM Roles/Permissions

| Service / Task                  | Service Account     | IAM Role(s) Needed               | Scope / Best Practice Summary                   |
|---------------------------------|---------------------|----------------------------------|-------------------------------------------------|
| Cloud SQL: DB connection, IAM   | Separate per-app    | cloudsql.client, instanceUser    | Unique per microservice/app, no admin overkill  |
| Firestore: server access        | Separate per-app    | datastore.user / viewer          | Only writer if required; no admin for app jobs  |
| Vertex AI: embeddings/LLMs      | Separate per-app    | aiplatform.user                  | No model admin rights in app service accounts   |

---

## Avoiding Retrying and Idempotency Traps in GCP SDKs

- **All official Python SDKs for GCP (Firestore Admin SDK, Cloud SQL Connector, Vertex AI SDK) are built to handle automatic retries for transient errors (e.g., HTTP 5xx, connection drops) using exponential backoff**.
- **Never implement manual retry loops around write or database operation unless the operation is designed to be idempotent.**  
  Example:  
  - "Read-modify-write" patterns in Firestore should use transactions if you want retry-safety.
  - For Cloud SQL, commit logic should be atomic—avoid custom retry logic for inserts/updates.
- **Connection pooling**: Use pooling features in the relevant SQLAlchemy or SDK wrapper; do not manually reissue new connections for expired tokens—the connectors will re-authenticate as needed.

---

## Full Example: Minimal IAM-Driven, Passwordless Stack (Python)

Suppose your `main.py` uses a structure like:

```python
from google.cloud.sql.connector import Connector
from firebase_admin import credentials, firestore, initialize_app
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Cloud SQL setup (see above)

# Firestore setup
cred = credentials.ApplicationDefault()
initialize_app(cred)
db = firestore.client()
# Use Firestore as above

# Vertex AI init
vertexai.init(project="YOUR_PROJECT_ID", location="us-central1")
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
embeddings = embedding_model.get_embeddings(["text 1", "text 2"])
# Use `embeddings` as needed
```
In this pattern, **all authentication is driven by the environment/service account identity; no passwords or API keys are maintained in the codebase. All retry and refresh logic is handled by the SDKs**.

---

## Recommendations and Security Best Practices

- **Least privilege**: Assign only the roles strictly necessary for each application or component. Use custom scoped roles if none fit your exact needs.
- **Separation of duties**: Use separate service accounts for each app or even for different responsibilities within a single system (e.g., one for Cloud SQL reads, another for writes).
- **Audit logging**: With all access tied to IAM-controlled service accounts, all interactions are logged in Cloud Audit Logs, and can be filtered by principal for deep compliance auditing.
- **Credentials management**: Only store long-lived service account key files in highly secure locations or (preferably) rely on GCP-managed workload identity binding (Cloud Run, GKE, Functions). Never check credentials into source control.
- **Key rotation and decommissioning**: When replacing or decommissioning a service, simply remove the IAM bindings or service account; no need for password rotation.

---

## Conclusion

Implementing passwordless authentication for Cloud SQL, Firestore, and Vertex AI is not only the most secure and auditable way to run production services on GCP, but also the simplest for minimizing code complexity and operational risk. By leveraging IAM-based authentication, rigorous service-account separation, and Google’s robust SDKs and connectors, you can write secure, maintainable integration code with zero password logic and maximal clarity and maintainability.  
Always assign separate service accounts with only the minimal roles necessary for access; use official connectors and SDKs to manage authentication transparently; and avoid writing manual retry, idempotency, or token-refresh logic—leave this to professionally maintained SDKs.

**References to GCP documentation, recent blog posts, Stack Overflow answers, and open-source SDKs have been used throughout this guide to ensure practical, real-world instructions aligned with current cloud security best practices.**
Perfect — I’m now working on a comprehensive technical guide for passwordless IAM-based access to Cloud SQL, Firestore, and Vertex AI (Gecko text embedding). I’ll include service account recommendations, required permissions, and optimized integration code tailored to your setup. This will take several minutes, so feel free to step away — I’ll keep working in the background and save the report right here in our conversation.
















*****  grok on google passwordless 


Research suggests that passwordless (IAM-based) authentication is feasible for Cloud SQL (PostgreSQL), Firestore, and Vertex AI text embeddings in Google Cloud, provided the appropriate setup and permissions are in place.
It seems likely that using a single service account, such as your Cloud Run service account (genai-rag-run-sa-6bdb68ec@thebestever.iam.gserviceaccount.com), can handle all connections, though separate accounts could be used if isolation is preferred; this approach balances simplicity with security.
Evidence leans toward requiring minor code changes for Cloud SQL to switch to IAM authentication via the Cloud SQL Python Connector, while Firestore and Vertex AI already operate passwordlessly using Application Default Credentials (ADC).
Key considerations include enabling IAM database authentication on your Cloud SQL instance and ensuring the service account has the necessary roles, acknowledging that setup steps may vary slightly based on your environment.

Setup Overview
To achieve passwordless connections, leverage Google Cloud's Identity and Access Management (IAM) system. This eliminates the need for static passwords by using short-lived tokens or ADC. For Cloud SQL, you'll need to enable IAM authentication on the instance and use a connector library. Firestore and Vertex AI inherently support IAM via their client libraries when running in a Google Cloud environment like Cloud Run.
Recommended Service Account
Use your existing Cloud Run service account (genai-rag-run-sa-6bdb68ec@thebestever.iam.gserviceaccount.com) for all connections, as it already has relevant permissions like Cloud SQL Client, Cloud SQL Instance User, and Vertex AI Administrator. This minimizes management overhead. If you prefer separation (e.g., for least privilege), create dedicated service accounts:

One for Cloud SQL (with roles/cloudsql.instanceUser and roles/cloudsql.client).
One for Firestore (with roles/datastore.user).
One for Vertex AI (with roles/aiplatform.user).

Assign these via environment variables or ADC in your code.
Additional Permissions Needed
Your Cloud Run service account is mostly set, but add roles/datastore.user for Firestore access, as it's not currently listed in your permissions. No changes are needed for Vertex AI, but consider downgrading from Vertex AI Administrator to Vertex AI User for better security. For Cloud SQL, ensure the instance flag cloudsql.iam_authentication is set to On (via gcloud or console).
Integration Syntax
Update your code to remove password logic. For Cloud SQL, install cloud-sql-python-connector and psycopg (already in use). Here's an optimized version of get_db_connection (removes unnecessary checks, assumes ENABLE_DB=1):
pythonfrom google.cloud.sql.connector import Connector
import psycopg  # Your existing import

def get_db_connection():
    global db_conn
    with db_lock:
        if db_conn is None or db_conn.closed:
            connector = Connector()
            db_conn = connector.connect(
                os.environ['INSTANCE_CONNECTION_NAME'],
                "psycopg",  # Driver for psycopg3
                user=os.environ['DB_USER'],  # Set to SA email, e.g., "genai-rag-run-sa-6bdb68ec@thebestever.iam.gserviceaccount.com"
                db=os.environ['DB_NAME'],
                enable_iam_auth=True
            )
            logging.info(f"IAM DB connection established. ID: {id(db_conn)}")
        else:
            logging.info(f"Reusing IAM DB connection. ID: {id(db_conn)}")
    return db_conn
For Firestore and Vertex AI, no changes needed—they use ADC automatically. Set DB_USER to the service account email in your environment variables and remove DB_PASSWORD.

Implementing passwordless authentication in Google Cloud for services like Cloud SQL (PostgreSQL), Firestore, and Vertex AI involves shifting from traditional username/password credentials to IAM-based mechanisms. This approach enhances security by using ephemeral tokens and service account identities, reducing the risk of credential leaks. As of September 2025, these features are fully supported and generally available, with ongoing improvements in libraries like the Cloud SQL Python Connector (version 1.5+ supports psycopg3 natively). Below, we provide a comprehensive guide, including setup steps, code optimizations, permission details, and potential pitfalls, drawing from official Google Cloud documentation and best practices.
Enabling Passwordless Authentication: Step-by-Step

For Cloud SQL (PostgreSQL):

Enable IAM database authentication on your instance: Use the Google Cloud Console, gcloud CLI (gcloud sql instances patch [INSTANCE_NAME] --database-flags cloudsql.iam_authentication=on), or Terraform. This flag allows IAM principals (like service accounts) to authenticate without passwords.
Grant the service account the roles/cloudsql.instanceUser role (your SA already has this) and roles/cloudsql.client for connectivity.
In code, use the Cloud SQL Python Connector to handle token generation. This library automatically fetches an OAuth 2.0 token for the service account and uses it as a temporary password. No manual token management is needed.
Connection timeouts and refreshes are handled internally; no retries are required in your code, aligning with your preference for simplicity.


For Firestore:

Firestore clients authenticate via ADC, which defaults to the attached service account when running on Cloud Run or similar. No explicit credentials are needed in code—firestore.Client() suffices.
Ensure the service account has roles/datastore.user (or equivalent for Firestore mode). If not present, add it via IAM policies to avoid permission denied errors.
This is inherently passwordless and scales well for server-to-server interactions.


For Vertex AI Text Embeddings (Gecko Model):

The Vertex AI SDK uses ADC for authentication, so libraries like google.cloud.aiplatform and google.generativeai (with vertexai=True) operate without passwords.
Your code already loads the Gecko model (textembedding-gecko@003) via aiplatform.TextEmbeddingModel.from_pretrained, which authenticates implicitly.
Permissions: roles/aiplatform.user is sufficient for embedding generation; your SA's Vertex AI Administrator role covers this but could be refined for least privilege.
No changes needed, but for optimization, batch embeddings in larger chunks (up to 100 as in your code) to reduce API calls.



All services can run fully passwordless using the same service account, as IAM policies allow granular role assignments. If using separate accounts, impersonate them via GOOGLE_APPLICATION_CREDENTIALS environment variables pointing to JSON keys, but ADC with the runtime SA is more efficient and secure.
Optimized Code Integration
Your existing code is already lean, but we can streamline it further by removing password-related logic, lazy imports (for clarity), and unnecessary checks. Focus on the get_db_connection function, as other clients (Firestore, Vertex AI) are unchanged. Add cloud-sql-python-connector to your requirements.txt.
Updated environment setup (remove password defaults):
pythonos.environ.setdefault("DB_USER", "genai-rag-run-sa-6bdb68ec@thebestever.iam.gserviceaccount.com")  # Use runtime SA email
# Remove os.environ.setdefault("DB_PASSWORD", ...)
Full optimized get_db_connection (concise, no retries, thread-safe):
pythonfrom google.cloud.sql.connector import Connector
import psycopg

db_conn = None
db_lock = Lock()
connector = Connector()  # Global for reuse

def get_db_connection():
    if os.environ.get("ENABLE_DB") != "1":
        return None
    global db_conn
    with db_lock:
        if db_conn is None or db_conn.closed:
            db_conn = connector.connect(
                os.environ["INSTANCE_CONNECTION_NAME"],
                "psycopg",
                user=os.environ["DB_USER"],
                db=os.environ["DB_NAME"],
                enable_iam_auth=True
            )
            logging.info("IAM-authenticated DB connection established.")
        else:
            logging.info("Reusing IAM DB connection.")
    return db_conn
For Vertex AI embeddings, your _generate_and_store_embeddings is efficient; no further trimming needed without losing clarity. If space is a concern, inline the batch processing loop slightly:
python# In _generate_and_store_embeddings
embeddings_list = []
for i in range(0, len(chunks), 100):
    batch = chunks[i:i+100]
    try:
        embeddings = model.get_embeddings(batch)
        embeddings_list.extend(e.values for e in embeddings)  # Assuming 'values' attr
    except Exception as e:
        logging.error(f"Embedding batch {i} failed: {e}")
if not embeddings_list:
    return None
arr = np.array(embeddings_list, dtype=float)
# ... persist logic ...
This reduces lines while maintaining readability. For Firestore, the get_firestore_client remains as-is—it's already optimal.
Permissions Table
Here's a table summarizing required and recommended IAM roles for each service, based on your current setup. Your Cloud Run SA handles most, but additions are noted.





ServiceRequired RolesCurrent in Your SA?Recommended AdditionsNotesCloud SQLroles/cloudsql.instanceUser, roles/cloudsql.clientYesNoneEnables IAM auth; instance flag must be on.Firestoreroles/datastore.userNoAdd this roleFor read/write; Editor would also work but is broader.Vertex AI (Embeddings)roles/aiplatform.userYes (via Administrator)None (but downgrade from Administrator for security)Covers Gecko model access.
Potential Challenges and Guidance

Cloud SQL Instance Configuration: If IAM auth isn't enabled, connections will fail with "IAM service account authentication failed." Test via gcloud: gcloud sql instances describe [INSTANCE_NAME] --format="value(settings.databaseFlags)" and verify the flag.
Service Account Email Length: Postgres usernames are limited to 63 characters; your SA email is 61, so it's fine. If using a longer one, truncate or use a shorter-named SA.
Performance and Patience: IAM token fetches add negligible latency (seconds at most), fitting your tolerance for waiting. No retries needed—the connector handles refreshes.
Testing: Deploy to a staging environment first. Use gcloud auth application-default login locally for ADC simulation.
Optimizations Beyond Basics: For embeddings, consider upgrading to a newer Gecko version if available (e.g., @004) for better accuracy, but stick to @003 as in your code unless tested. Batch sizes of 100 are optimal for cost/latency balance.
Security Considerations: Passwordless reduces attack surface but monitor IAM policies regularly. Use workload identity federation if running outside GCP.

This setup ensures a robust, passwordless pipeline while keeping your code concise and functional.
Key Citations

IAM authentication | Cloud SQL for PostgreSQL
Log in using IAM database authentication | Cloud SQL for PostgreSQL
Connect PostgreSQL Clients to Cloud SQL for ... - Medium
Create a connection using the Cloud SQL Python Connector
cloud-sql-python-connector - PyPI
Connect using Cloud SQL Language Connectors
Authenticate to Firestore | Firestore in Native mode - Google Cloud
Text embeddings API | Generative AI on Vertex AI - Google Cloud
Architecting Passwordless Python Applications on Google Cloud: A Definitive Guide to Secure Integration with Cloud SQL, Firestore, and Vertex AI


I. Introduction: The Imperative of Passwordless Service-to-Service Communication


The Paradigm Shift from Secrets to Identity

In modern cloud-native application development, the methodology for securing communication between services has undergone a fundamental paradigm shift. The traditional approach, reliant on the management and distribution of static secrets such as database passwords and API keys, is increasingly recognized as a significant security liability. This model creates a vast and fragile "secret sprawl," where credentials must be stored in configuration files, environment variables, or dedicated secret management systems. The operational overhead of rotating these secrets, coupled with the catastrophic risk of accidental leakage, necessitates a more robust and scalable solution. The contemporary best practice is to move from a model of shared secrets to one of verifiable identity. In this identity-centric paradigm, applications are treated as first-class principals within the security framework, each with a unique, verifiable identity that can be granted specific, auditable permissions. This report provides a definitive technical blueprint for implementing such a passwordless architecture for Python applications on Google Cloud Platform (GCP), focusing on secure, identity-based integration with Cloud SQL, Firestore, and Vertex AI.

Clarifying "Passwordless": Service Identity vs. User Authentication

It is critical to establish a clear distinction at the outset. The term "passwordless" in this context refers specifically to service-to-service authentication. It addresses how an application workload—running on a server, in a container, or as a serverless function—authenticates itself to backend services and APIs without using long-lived, static passwords. This is fundamentally different from user-facing passwordless authentication technologies like passkeys, which are designed to allow human users to log into applications using biometrics or device-level security.1 The focus of this guide is the secure, automated authentication of the application itself, a foundational element of a zero-trust security posture in the cloud.

Benefits of the Passwordless Architecture

Adopting an identity-based, passwordless architecture for service-to-service communication yields substantial benefits across security, operations, and compliance domains.
Enhanced Security: The primary advantage is the elimination of static credentials from the application's environment. By removing passwords and API keys from code, configuration files, and environment variables, the most common vector for credential compromise is neutralized. Authentication instead relies on short-lived access tokens that are automatically generated and rotated by the cloud platform, drastically reducing the attack surface.2
Simplified Operations: The complex and error-prone process of manual secret rotation is rendered obsolete. Access control is centralized within Google Cloud's Identity and Access Management (IAM) service, providing a single, unified control plane for managing all application permissions. This simplifies security policy administration and reduces the operational burden on development and operations teams.
Improved Auditability and Compliance: When an application authenticates using a dedicated identity (a service account), every action it performs against a GCP service is meticulously logged and attributed to that specific identity in Cloud Audit Logs. This creates an unambiguous audit trail, which is invaluable for security forensics, incident response, and demonstrating compliance with regulatory standards.
Seamless Developer Experience: This architecture fosters a superior developer workflow. During local development, developers can use their own user credentials via the Google Cloud CLI to authenticate their application, allowing it to run with their permissions. When the same code is deployed to a production environment on GCP, it seamlessly and automatically switches to using a designated service account identity, with no code changes required.3 This consistency between development and production environments accelerates development cycles and reduces configuration-related errors.

II. Foundational Pillars: Service Accounts and Application Default Credentials (ADC)

Understanding the passwordless authentication model on Google Cloud requires a firm grasp of two foundational concepts: Service Accounts, which provide the "who," and Application Default Credentials (ADC), which provides the "how." Together, they form the backbone of secure, identity-driven access to GCP services.

Service Accounts as Secure Application Principals

A service account is a special type of non-human identity within Google Cloud, designed to represent an application, a virtual machine, or another automated workload that needs to authenticate to and be authorized to access GCP APIs.5 It serves as the application's distinct identity, analogous to a user account for a person.

Best Practice: User-Managed vs. Default Service Accounts

Google Cloud environments, such as a new Compute Engine instance, often come with a default service account. A critical security best practice is to avoid using these default service accounts for production applications. By default, these accounts are often granted the highly permissive basic "Editor" role for the entire project, which provides broad read and write access to most GCP resources.5 Using such an account directly violates the
principle of least privilege, a core security tenet stating that an identity should only be granted the minimum permissions necessary to perform its intended function. Instead, the mandated approach is to create a dedicated, user-managed service account for each application or microservice. This allows for the precise and granular assignment of permissions, ensuring that if the application's identity is ever compromised, the potential blast radius is strictly limited to the resources it absolutely needs to access.

The ADC Authentication Flow: A Deconstructed View

Application Default Credentials (ADC) is not a specific type of credential but rather a standardized strategy employed by Google Cloud client libraries to automatically locate and use credentials based on the environment in which the application is running.3 This mechanism creates a powerful abstraction that decouples the application's code from the specifics of its authentication environment.

The Chain of Discovery

ADC searches for credentials in a well-defined, sequential order. The first valid credential it finds is used for authentication. This deterministic process is key to understanding and troubleshooting authentication behavior.
GOOGLE_APPLICATION_CREDENTIALS Environment Variable: ADC first checks for the presence of this environment variable. If set, it is expected to contain the file path to a service account key file (a JSON file). While this method is useful for local testing or for applications running outside of GCP (e.g., on-premises or in another cloud), it is strongly discouraged for production workloads within GCP. It reintroduces a static secret (the key file) into the environment, undermining the primary goal of a passwordless architecture.
User Credentials from the gcloud CLI: If the environment variable is not set, ADC looks for user credentials that have been configured for application use. This is typically done by a developer running the command gcloud auth application-default login on their local machine.6 This command generates a credential file in a well-known location on the developer's filesystem. This is the preferred method for local development, as it allows the application to temporarily act with the developer's own permissions in a secure and auditable manner.
The Attached Service Account via the Metadata Server: If neither of the above methods yields a credential, and the code is running on a GCP compute resource (such as a Compute Engine VM, Cloud Run service, App Engine instance, or GKE node), ADC turns to the instance metadata server. This is the primary mechanism for production environments. When a GCP resource is launched, it can have a service account "attached" to it. The client library can then query a local, non-routable metadata server endpoint (http://metadata.google.internal) to obtain short-lived access tokens for that attached service account.5 The client library handles this entire interaction—including token caching and refreshing—transparently.
This elegant chain of discovery is what enables code portability. The same Python code, which simply instantiates a client library (e.g., firestore.Client()), will use the developer's gcloud credentials when run locally and automatically switch to using the attached service account's credentials via the metadata server when deployed to Cloud Run, all without a single line of conditional logic. This significantly simplifies the development-to-production pipeline and reduces environmental configuration drift.

III. Part 1: Passwordless Connectivity for Google Cloud SQL

Connecting to Cloud SQL in a passwordless manner is a more nuanced process than for other cloud-native services. This is because Cloud SQL manages standard database engines like PostgreSQL and MySQL, which have their own long-standing, built-in authentication systems that are distinct from GCP's IAM framework. The solution involves a sophisticated "bridge" that harmonizes these two systems, orchestrated by a specialized client library.

Mechanism Deep Dive: Bridging IAM and Traditional Database Authentication

The core challenge is to make a traditional database engine, which expects a username and password, accept an IAM identity for authentication. Google Cloud solves this with two key components: IAM Database Authentication and the Cloud SQL Python Connector.
IAM Database Authentication: This is a feature that must be explicitly enabled on a Cloud SQL instance. When active, it configures the database engine to recognize a short-lived OAuth 2.0 access token as a valid, temporary password for a specially designated IAM user.2 This effectively creates a new authentication path into the database that is governed by IAM policies.
The Cloud SQL Python Connector: This library is far more than a simple database driver. It is an intelligent client that automates the entire IAM authentication lifecycle. When an application requests a connection, the connector uses the ambient ADC credentials to securely call the Cloud SQL Admin API. It retrieves necessary connection details and, most importantly, a valid one-hour access token for the application's service account. The connector then injects this token as the password into the database connection string. For long-running applications that use connection pooling, the connector manages the token refresh process automatically in a background thread, ensuring uninterrupted connectivity.2 This provides a native, in-process alternative to running the standalone Cloud SQL Auth Proxy.11
This two-part system cleverly adapts a traditional authentication model to a modern, token-based identity framework without requiring modifications to the underlying database engine. The cloudsql.iam_authentication flag tells the database instance to accept these special token-based "passwords," while the connector acts as the trusted intermediary that generates and supplies them.

Step-by-Step Configuration and IAM Permissions

A successful implementation requires configuration at both the instance and IAM levels.
Enable Required APIs: The Cloud SQL Python Connector relies on the Cloud SQL Admin API to function. This API must be enabled in the GCP project.
gcloud services enable sqladmin.googleapis.com
Enable the cloudsql.iam_authentication Flag: This flag must be set on the Cloud SQL instance. This is a non-disruptive change and can be applied to an existing instance. It allows both traditional password-based logins and new IAM-based logins to coexist.2

gcloud sql instances patch INSTANCE_NAME --database-flags=cloudsql.iam_authentication=on
Create an IAM Database User: The application's service account must be added to the Cloud SQL instance as a database user. The user type must be specified as CLOUD_IAM_SERVICE_ACCOUNT.13

gcloud sql users create SERVICE_ACCOUNT_EMAIL --instance=INSTANCE_NAME --type=CLOUD_IAM_SERVICE_ACCOUNT
Grant Necessary IAM Roles: This is a critical two-part requirement that is often a source of confusion. The service account needs two distinct roles to complete the authentication flow successfully.
roles/cloudsql.client: This role governs access to the control plane. It contains permissions like cloudsql.instances.connect and cloudsql.instances.get, which are necessary for the connector to call the Cloud SQL Admin API, retrieve instance metadata, and generate the short-lived access token.15
roles/cloudsql.instanceUser: This role governs access to the data plane. It contains the cloudsql.instances.login permission, which authorizes the service account to actually use the token it just fetched to log in to the database instance.16
Without both roles, the process will fail. The service account might be able to fetch a token (cloudsql.client) but be denied entry at the database (cloudsql.instanceUser), or it might be authorized to log in but be unable to get a token in the first place.
IAM Role
Key Permission(s)
Purpose in Authentication Flow
roles/cloudsql.client
cloudsql.instances.connect, cloudsql.instances.get
Control Plane: Allows the application (via the connector) to connect to the Cloud SQL Admin API to retrieve instance metadata and a short-lived OAuth 2.0 access token.
roles/cloudsql.instanceUser
cloudsql.instances.login
Data Plane: Authorizes the IAM principal (the service account) to use the fetched access token to successfully log in to the database instance itself.


Python Implementation with the Cloud SQL Connector

For robust, production-grade applications, using the connector with an object-relational mapper (ORM) or connection pooling library like SQLAlchemy is highly recommended.11
First, install the necessary packages, specifying the database driver (e.g., pg8000 for PostgreSQL).

Bash


pip install "cloud-sql-python-connector[pg8000]" sqlalchemy


The following Python code demonstrates how to configure a SQLAlchemy connection pool to use the connector for passwordless IAM authentication to a PostgreSQL instance.

Python


import sqlalchemy
from google.cloud.sql.connector import Connector, IPTypes
import pg8000 # Driver for PostgreSQL

# Initialize the Cloud SQL Python Connector.
# This object manages the background refresh of credentials.
connector = Connector()

# Function to create a database connection.
# SQLAlchemy's connection pool will call this function to create new connections.
def getconn() -> pg8000.dbapi.Connection:
    """
    Creates a connection to a Cloud SQL instance using IAM authentication.
    """
    # Define connection parameters.
    instance_connection_name = "your-project:your-region:your-instance"
    db_name = "your-database-name"

    # For PostgreSQL, the IAM user must be the service account's email
    # address WITHOUT the ".gserviceaccount.com" suffix.
    # For MySQL, the full service account email is used.
    iam_user = "your-app-sa@your-project.iam"

    # The connector.connect method handles the entire authentication flow:
    # 1. Uses ADC to find the application's credentials.
    # 2. Calls the Cloud SQL Admin API to get an access token.
    # 3. Establishes a secure TLS connection to the instance.
    # 4. Returns a standard database driver connection object.
    conn: pg8000.dbapi.Connection = connector.connect(
        instance_connection_name,
        "pg8000",
        user=iam_user,
        db=db_name,
        enable_iam_auth=True, # This flag is essential for passwordless authentication.
        ip_type=IPTypes.PRIVATE # Optional: use IPTypes.PUBLIC for public IP.
    )
    return conn

# Create a SQLAlchemy engine that uses the getconn function as its creator.
# The connection URL is dialect-only, as the connector handles the host, port, etc.
pool = sqlalchemy.create_engine(
    "postgresql+pg8000://",
    creator=getconn,
)

# Example of using the connection pool to interact with the database.
with pool.connect() as db_conn:
    result = db_conn.execute(sqlalchemy.text("SELECT NOW()")).fetchone()
    print(f"Database current time: {result}")

# Clean up the connector's background resources when the application shuts down.
connector.close()


The most critical elements in this implementation are the creator=getconn argument passed to sqlalchemy.create_engine, which delegates connection creation to the connector, and the enable_iam_auth=True flag within the connector.connect call, which activates the passwordless authentication flow.11 The specific formatting of the
iam_user string is also a crucial detail that differs between database engines and must be configured correctly.11

IV. Part 2: Seamless, Secure Access to Firestore

In stark contrast to the bridged approach required for Cloud SQL, authentication with Firestore is a model of cloud-native simplicity. As a fully managed NoSQL database service designed from the ground up for the Google Cloud ecosystem, Firestore integrates natively and transparently with IAM.

Leveraging ADC for Native Firestore Authentication

Firestore does not possess a separate, legacy authentication system that needs to be accommodated. Its access control model is built directly upon GCP IAM for server-side access.6 Consequently, the
google-cloud-firestore client library is designed to use Application Default Credentials (ADC) out of the box. There are no special connector libraries to install, no instance flags to enable, and no special authentication parameters to pass in the code. The authentication process is handled entirely and transparently by the underlying Google Cloud authentication library, which automatically executes the ADC discovery chain.4
It is important to distinguish this server-side IAM mechanism from Firebase Security Rules. While both control access to Firestore data, they serve different purposes. IAM policies are evaluated for requests coming from server environments (like a Python application running on Cloud Run) that are authenticated with a service account or user credential. Firebase Security Rules, on the other hand, are designed to enforce granular, data-aware access control for requests originating from mobile and web clients that are typically authenticated via Firebase Authentication.23 This report is exclusively concerned with the server-side IAM model.

IAM Permissions for Firestore Operations

For a typical backend application that needs to perform standard Create, Read, Update, and Delete (CRUD) operations on Firestore documents, the principle of least privilege points to a single predefined role: roles/datastore.user. This role is specifically designed for application service accounts and provides the necessary permissions for data manipulation without granting administrative privileges.25
The roles/datastore.user role is a collection of more granular permissions that map directly to the Firestore API methods. Understanding these underlying permissions is useful for creating even more restrictive custom roles if needed. The key permissions included are:
datastore.entities.create: Allows the creation of new documents.
datastore.entities.get: Allows reading the data from a single document.
datastore.entities.list: Allows querying collections to retrieve multiple documents.
datastore.entities.update: Allows modifying the data within existing documents.
datastore.entities.delete: Allows the deletion of documents.
These permissions provide comprehensive coverage for all standard data access patterns.25
Application Action
Required IAM Permission
Provided by Role
Creating a document
datastore.entities.create
roles/datastore.user
Reading a single document
datastore.entities.get
roles/datastore.user
Querying a collection
datastore.entities.list
roles/datastore.user
Updating a document
datastore.entities.update
roles/datastore.user
Deleting a document
datastore.entities.delete
roles/datastore.user


Python Implementation with the Firestore Client Library

The implementation of passwordless Firestore access in Python highlights the elegance of the ADC model. The code is clean, concise, and entirely focused on business logic, with no explicit authentication code.
First, install the client library.

Bash


pip install google-cloud-firestore


The following code snippet demonstrates initializing the client and performing a simple write operation.

Python


from google.cloud import firestore

# Initialize the Firestore client.
# No credentials, API keys, or service account files are passed.
# The client library automatically uses Application Default Credentials (ADC)
# to find the ambient credentials from the environment.
db = firestore.Client(project="your-gcp-project-id")

# Create a reference to a document.
doc_ref = db.collection("programmers").document("alovelace")

# Write data to the document.
# The client handles authentication and authorization for this API call
# transparently using the ADC-provided credentials.
doc_ref.set({
    "first": "Ada",
    "last": "Lovelace",
    "born": 1815
})

print(f"Document {doc_ref.id} created successfully.")


The key takeaway from this code is its simplicity. The instantiation firestore.Client() triggers the ADC flow, and all subsequent API calls made with the db object are automatically authenticated. This declarative approach—where the developer declares their intent to use the service and the platform handles the mechanics of authentication based on the environment's identity—represents the ideal state for cloud service integration. It allows developers to remain focused on application logic, while security is managed declaratively through IAM and the configuration of the execution environment.

V. Part 3: Passwordless Operations with Vertex AI Text Embeddings

Authenticating against Google Cloud's advanced AI and Machine Learning services, such as Vertex AI, follows the same modern, standardized pattern as Firestore. These services expose robust APIs that are secured by IAM, and their corresponding client libraries are built to leverage ADC for seamless, passwordless authentication.

Authenticating to the Vertex AI API

Vertex AI provides its capabilities through a standard set of REST and gRPC APIs. Access to these APIs is governed by IAM permissions, making the authentication model consistent with other contemporary GCP services. The google-cloud-aiplatform Python SDK, which provides a high-level interface to these APIs, is designed to use ADC as its default authentication mechanism. As with Firestore, this means that no explicit credential management is required within the application code itself; the library automatically discovers and uses the credentials available in the environment.8

IAM Permissions for textembedding-gecko

To generate text embeddings using a model like textembedding-gecko, the application needs permission to make a prediction request against the model's serving endpoint. For an application that is only consuming pre-trained models and not managing or training them, the principle of least privilege leads to the roles/aiplatform.user predefined role.31
This role is specifically designed for principals that need to use Vertex AI services without having administrative control over them. The most critical permission contained within this role for the purpose of generating embeddings is aiplatform.endpoints.predict.32 This permission grants the ability to invoke a deployed model on a Vertex AI endpoint, which is the underlying API action performed when requesting an embedding from a publisher model like
textembedding-gecko.
It is crucial to avoid granting overly broad roles such as roles/aiplatform.admin or the basic Editor role. These roles include permissions to create, delete, or modify models, endpoints, and datasets, which are unnecessary for a simple inference workload and would represent a significant security risk if the application's service account were compromised.32
Application Goal
API Action
Required IAM Permission
Provided by Role
Generate text embeddings
Call a publisher model endpoint
aiplatform.endpoints.predict
roles/aiplatform.user


Python Implementation for Generating Text Embeddings

The Python code for generating embeddings from the textembedding-gecko model is straightforward and, from an authentication perspective, identical to the Firestore example.
First, install the necessary Vertex AI SDK.

Bash


pip install google-cloud-aiplatform


The following code initializes the client, specifies the model, and requests embeddings for a list of text strings.

Python


import vertexai
from vertexai.language_models import TextEmbeddingModel

# Initialize the Vertex AI SDK for your project and region.
# As with other Google Cloud client libraries, this initialization
# does not require explicit credentials. It relies on the ADC flow
# to authenticate subsequent API calls.
vertexai.init(project="your-gcp-project-id", location="us-central1")

# Instantiate a client for a specific text embedding model.
# The model name includes a version number, e.g., @003.
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

# The list of texts to be embedded.
texts_to_embed =

# Call the get_embeddings method. The SDK handles the authenticated
# API request to the Vertex AI endpoint in the background.
embeddings = model.get_embeddings(texts_to_embed)

# Process the results.
for i, embedding in enumerate(embeddings):
    vector = embedding.values
    print(f"--- Text: '{texts_to_embed[i]}' ---")
    print(f"Generated embedding vector of length: {len(vector)}")
    # print(f"Vector (first 5 elements): {vector[:5]}")


This code demonstrates the power of a standardized platform design. Accessing a sophisticated large language model for text embedding is, from a security and authentication standpoint, identical to accessing a NoSQL database. This consistency across vastly different service types is a significant advantage of the GCP ecosystem. Once developers understand the ADC and IAM pattern for one service, they can apply that knowledge to dozens of others, dramatically accelerating development and reducing the cognitive load associated with learning disparate security models for each new service they integrate.

VI. Unified IAM Strategy: The Optimal Service Account Configuration

Having analyzed the specific requirements for Cloud SQL, Firestore, and Vertex AI, the final step is to synthesize these findings into a single, cohesive IAM strategy. The optimal approach is to create one dedicated service account for the application and bind all the necessary, least-privilege roles to it. This centralizes the application's identity and provides a clear, auditable record of its permissions.

Designing the Least-Privilege Service Account

A new, user-managed service account should be created specifically for this Python application. Naming conventions are important for manageability; a name like python-app-prod-sa clearly indicates its purpose.
The following gcloud command creates the service account:

Bash


gcloud iam service-accounts create python-app-prod-sa \
    --display-name="Production Python Application Service Account" \
    --description="Service account for the primary Python backend application, providing access to Cloud SQL, Firestore, and Vertex AI."


This command, based on standard practices, establishes the unique identity for the application.5

Consolidated Role Bindings

With the service account created, the next step is to bind the four predefined roles identified in the previous sections to this single principal. This grants the service account the exact set of permissions it needs to interact with all three backend services, and nothing more.
The following gcloud commands attach each required role to the newly created service account. It is best practice to script these commands as part of an infrastructure-as-code (IaC) deployment.

Bash


# Set environment variables for clarity and reusability
export PROJECT_ID=$(gcloud config get-value project)
export SA_EMAIL="python-app-prod-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Grant Cloud SQL Permissions
# Allows the connector to fetch connection info and auth tokens
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/cloudsql.client"

# Authorizes the service account to log in to the database with the token
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/cloudsql.instanceUser"

# Grant Firestore Permissions
# Provides permissions for all standard document CRUD operations
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/datastore.user"

# Grant Vertex AI Permissions
# Provides permission to call the text embedding model endpoint
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/aiplatform.user"


Executing these commands results in a single service account principal that is precisely configured according to the principle of least privilege for the application's specific needs.5
The following table provides a consolidated summary of the final recommended IAM configuration for the application's service account. This serves as a definitive checklist for implementation.
Service
Recommended IAM Role
Reason for Inclusion
Cloud SQL
roles/cloudsql.client
Required by the Python Connector to fetch connection info and auth tokens from the Cloud SQL Admin API.
Cloud SQL
roles/cloudsql.instanceUser
Required to authorize the service account to log in to the database instance using the fetched OAuth 2.0 token.
Firestore
roles/datastore.user
Provides necessary permissions for all standard document read and write (CRUD) operations.
Vertex AI
roles/aiplatform.user
Provides the aiplatform.endpoints.predict permission needed to call the textembedding-gecko model.


VII. Conclusion: Consolidating Your Passwordless Architecture

This report has provided a comprehensive, step-by-step guide to implementing a secure, passwordless architecture for Python applications on Google Cloud. By moving away from vulnerable static credentials and embracing an identity-centric security model, developers can build applications that are more secure, operationally efficient, and easier to audit.
The core principles underpinning this modern architecture are clear and consistent:
Prioritize Identity Over Secrets: Treat applications as first-class principals with their own unique identities in the form of dedicated, user-managed service accounts.
Leverage Application Default Credentials (ADC): Utilize the ADC flow as a powerful abstraction layer that decouples application code from the authentication environment, enabling seamless portability from local development to production.
Apply the Principle of Least Privilege: Meticulously grant only the minimum necessary IAM permissions required for an application to perform its function, using predefined roles where possible and custom roles for more granular control.
The analysis demonstrated two primary patterns of integration. For cloud-native services like Firestore and Vertex AI, the passwordless integration is seamless and transparent, handled automatically by the client libraries. For services like Cloud SQL that bridge the gap between traditional and cloud-native systems, a more sophisticated approach using the Cloud SQL Python Connector and IAM Database Authentication provides a robust and secure solution.
The architectural patterns and IAM strategies detailed here are not isolated solutions but foundational components of a modern cloud security posture. This identity-based approach is extensible to virtually all other Google Cloud services. Furthermore, it serves as the basis for more advanced security configurations, such as integrating with CI/CD pipelines using Workload Identity Federation for providers like GitHub Actions, or securing containerized workloads in Google Kubernetes Engine with Workload Identity.
Ultimately, adopting this passwordless architecture is not merely an incremental improvement but a fundamental step toward building truly cloud-native systems. It represents the industry best practice and the definitive standard for developing secure, scalable, and maintainable applications on Google Cloud.
Works cited
Passkeys - Google for Developers, accessed September 2, 2025, https://developers.google.com/identity/passkeys
IAM authentication | Cloud SQL for PostgreSQL | Google Cloud, accessed September 2, 2025, https://cloud.google.com/sql/docs/postgres/iam-authentication
Set up Application Default Credentials | Authentication - Google Cloud, accessed September 2, 2025, https://cloud.google.com/docs/authentication/provide-credentials-adc
Authenticate for using client libraries | Authentication | Google Cloud, accessed September 2, 2025, https://cloud.google.com/docs/authentication/client-libraries
Set up ADC for a resource with an attached service account | Authentication - Google Cloud, accessed September 2, 2025, https://cloud.google.com/docs/authentication/set-up-adc-attached-service-account
Authenticate to Firestore | Firestore in Native mode - Google Cloud, accessed September 2, 2025, https://cloud.google.com/firestore/native/docs/authentication
Authenticate to Cloud Storage - Google Cloud, accessed September 2, 2025, https://cloud.google.com/storage/docs/authentication
Authenticate to Vertex AI | Google Cloud, accessed September 2, 2025, https://cloud.google.com/vertex-ai/docs/authentication
Connect using Cloud SQL Language Connectors | Cloud SQL for MySQL - Google Cloud, accessed September 2, 2025, https://cloud.google.com/sql/docs/mysql/connect-connectors
Enhancing Cloud Database Security in PostgreSQL with IAM and Just-in-Time Access, accessed September 2, 2025, https://nhimg.org/community/nhi-best-practices/enhancing-cloud-database-security-in-postgresql-with-iam-and-just-in-time-access/
GoogleCloudPlatform/cloud-sql-python-connector: A ... - GitHub, accessed September 2, 2025, https://github.com/GoogleCloudPlatform/cloud-sql-python-connector
IAM based authentication for CloudSql and how to connect to CloudSql with private ip from local machine. | by Jitendra Jaladi | Google Cloud - Medium, accessed September 2, 2025, https://medium.com/google-cloud/iam-based-authentication-for-cloudsql-and-how-to-connect-to-cloudsql-with-private-ip-from-local-bf2644a14cef
How To Connect To CloudSQL With IAM Authentication - Xebia, accessed September 2, 2025, https://xebia.com/blog/how-to-connect-to-a-cloudsql-with-iam-authentication/
Example of how to enable Cloud SQL IAM Database Authentication for Groups - GitHub, accessed September 2, 2025, https://github.com/GoogleCloudPlatform/cloud-sql-iam-db-authn-groups
Roles and permissions | Cloud SQL for MySQL - Google Cloud, accessed September 2, 2025, https://cloud.google.com/sql/docs/mysql/roles-and-permissions
Cloud SQL roles | Cloud SQL for MySQL - Google Cloud, accessed September 2, 2025, https://cloud.google.com/sql/docs/mysql/iam-roles
Database Access with Cloud SQL for MySQL - Teleport, accessed September 2, 2025, https://goteleport.com/docs/enroll-resources/database-access/enroll-google-cloud-databases/mysql-cloudsql/
Log in using IAM database authentication | Cloud SQL for PostgreSQL, accessed September 2, 2025, https://cloud.google.com/sql/docs/postgres/iam-logins
Create a connection using the Cloud SQL Python Connector, accessed September 2, 2025, https://cloud.google.com/sql/docs/postgres/samples/cloud-sql-postgres-sqlalchemy-connect-connector
Connect to Cloud SQL using the Cloud SQL Python Connector - Colab - Google, accessed September 2, 2025, https://colab.research.google.com/github/GoogleCloudPlatform/cloud-sql-python-connector/blob/main/samples/notebooks/postgres_python_connector.ipynb
Get started with Cloud Firestore - Firebase - Google, accessed September 2, 2025, https://firebase.google.com/docs/firestore/quickstart
Cloud Firestore API - Python client library | Google Cloud, accessed September 2, 2025, https://cloud.google.com/python/docs/reference/firestore/latest
Secure data access for users and groups | Firestore - Firebase, accessed September 2, 2025, https://firebase.google.com/docs/firestore/solutions/role-based-access
Get started with Cloud Firestore Security Rules - Firebase - Google, accessed September 2, 2025, https://firebase.google.com/docs/firestore/security/get-started
Security for server client libraries | Firestore in Native mode - Google Cloud, accessed September 2, 2025, https://cloud.google.com/firestore/native/docs/security/iam
Firestore roles and permissions | IAM Documentation - Google Cloud, accessed September 2, 2025, https://cloud.google.com/iam/docs/roles-permissions/firestore
Security for server client libraries | Firestore - Google Cloud, accessed September 2, 2025, https://cloud.google.com/firestore/docs/security/iam
Identity and Access Management (IAM) | Firestore - Firebase, accessed September 2, 2025, https://firebase.google.com/docs/firestore/enterprise/security/iam
Authenticate | Generative AI on Vertex AI - Google Cloud, accessed September 2, 2025, https://cloud.google.com/vertex-ai/generative-ai/docs/migrate/openai/auth-and-credentials
Embeddings for Text – Vertex AI - Google Cloud Console, accessed September 2, 2025, https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/textembedding-gecko
GCP Vertex AI - Cline Docs, accessed September 2, 2025, https://docs.cline.bot/provider-config/gcp-vertex-ai
Vertex AI access control with IAM | Google Cloud, accessed September 2, 2025, https://cloud.google.com/vertex-ai/docs/general/access-control
