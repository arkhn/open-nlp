# Doctor-Editor-Moderator Workflow Diagram

Based on the current code implementation in `pipeline.py` and agent classes.

## High-Level Flow

```
Document Pair → Doctor Agent → Editor Agent → Moderator Agent → Database
                                     ↑              ↓
                                     └── Retry Loop ──┘
```

## Detailed Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        CLINICAL CONFLICT GENERATION PIPELINE                    │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│   Data Source   │
│ MIMIC-III Data  │
│ (4,753 docs)    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Document Pair   │
│   Sampling      │
│ (ClinicalData   │
│     Loader)     │
└─────────┬───────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PROCESSING LOOP                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────┐
        │ Start Processing│
        │  Document Pair  │
        │   (pair_id)     │
        └─────────┬───────┘
                  │
                  ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                               STEP 1: DOCTOR AGENT                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        ┌─────────────────┐
        │   Doctor Agent  │
        │   Analysis      │
        │                 │
        │ • Load prompt   │ ← prompts/doctor_agent.txt
        │ • Analyze docs  │
        │ • Select type   │
        │ • Generate      │
        │   instructions  │
        └─────────┬───────┘
                  │
                  ▼
        ┌─────────────────┐
        │ Conflict Result │
        │                 │
        │ • conflict_type │
        │ • reasoning     │
        │ • modification  │
        │   instructions  │
        └─────────┬───────┘
                  │
                  ▼ Log to processing_history
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                          STEP 2: RETRY LOOP (Max 3 attempts)                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                  │
                  ▼
        ┌─────────────────┐
        │ Start Attempt   │
        │  (attempt ++)   │
        └─────────┬───────┘
                  │
                  ▼
╔═════════════════════════════════════════════════════════════════════════════════╗
║                           EDITOR AGENT PROCESSING                              ║
╚═════════════════════════════════════════════════════════════════════════════════╝
        ┌─────────────────┐
        │  Editor Agent   │
        │  Modification   │
        │                 │
        │ • Load prompt   │ ← prompts/editor_agent.txt
        │ • Get conflict  │
        │   type info     │
        │ • Modify docs   │
        │ • Create        │
        │   conflicts     │
        │ • Validate      │
        │   changes made  │
        └─────────┬───────┘
                  │
                  ▼
        ┌─────────────────┐
        │  Editor Result  │
        │                 │
        │ • modified_doc1 │
        │ • modified_doc2 │
        │ • changes_made  │
        └─────────┬───────┘
                  │
                  ▼ Log to processing_history
╔═════════════════════════════════════════════════════════════════════════════════╗
║                         MODERATOR AGENT VALIDATION                             ║
╚═════════════════════════════════════════════════════════════════════════════════╝
        ┌─────────────────┐
        │ Moderator Agent │
        │   Validation    │
        │                 │
        │ • Load prompt   │ ← prompts/moderator_agent.txt
        │ • Compare docs  │
        │ • Score quality │
        │ • Check realism │
        │ • Generate      │
        │   feedback      │
        └─────────┬───────┘
                  │
                  ▼
        ┌─────────────────┐
        │ValidationResult │
        │                 │
        │ • is_valid      │
        │ • validation_   │
        │   score (0-100) │
        │ • feedback      │
        │ • issues_found  │
        │ • reasoning     │
        └─────────┬───────┘
                  │
                  ▼ Log to processing_history
        ┌─────────────────┐
        │  Score Check    │
        │                 │
        │ is_valid AND    │
        │ score ≥         │
        │ threshold?      │
        └─────┬─────┬─────┘
              │     │
             YES    NO
              │     │
              │     ▼
              │   ┌─────────────────┐
              │   │ Attempt < Max?  │
              │   │                 │
              │   │ • Log failure   │
              │   │ • Sleep 1s      │
              │   └─────┬─────┬─────┘
              │         │     │
              │        YES    NO
              │         │     │
              │         │     ▼
              │         │   ┌─────────────────┐
              │         │   │  Final Failure  │
              │         │   │                 │
              │         │   │ • Mark failed   │
              │         │   │ • Log error     │
              │         │   │ • Return result │
              │         │   └─────────────────┘
              │         │
              │         │
              │         └─────────────────────────┐
              │                                   │
              │                                   ▼
              │                         ┌─────────────────┐
              │                         │   Retry Loop    │
              │                         │                 │
              │                         │ Back to Editor  │
              │                         │ Agent (same     │
              │                         │ conflict_result)│
              │                         └─────────────────┘
              │                                   │
              │                                   │
              │   ┌───────────────────────────────┘
              │   │
              │   ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                            STEP 3: DATABASE SAVE                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
              │
              ▼
        ┌─────────────────┐
        │ Database Save   │
        │                 │
        │ Table:          │
        │ validated_      │
        │ documents       │
        │                 │
        │ • original_docs │
        │ • modified_docs │
        │ • conflict_type │
        │ • validation_   │
        │   score         │
        │ • changes_made  │
        │ • timestamp     │
        └─────────┬───────┘
                  │
                  ▼
        ┌─────────────────┐
        │  Success!       │
        │                 │
        │ • Return        │
        │   database_id   │
        │ • Mark success  │
        │ • Log result    │
        └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PARALLEL LOGGING                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

Every step logs to processing_history table:
┌─────────────────┐
│ processing_     │
│ history         │
│                 │
│ • doc_pair_id   │
│ • agent_name    │
│ • result_data   │
│ • processing_   │
│   time          │
│ • created_at    │
└─────────────────┘
```

## Timing and Performance Metrics

```
┌─────────────────────────────────────────────────────────────────┐
│                      PERFORMANCE BREAKDOWN                     │
├─────────────────────────────────────────────────────────────────┤
│ Doctor Agent:    2-5 seconds   │ LLM Analysis + Reasoning      │
│ Editor Agent:    3-8 seconds   │ LLM Generation + Validation   │
│ Moderator Agent: 2-4 seconds   │ LLM Validation + Scoring      │
├─────────────────────────────────────────────────────────────────┤
│ Total per pair:  7-17 seconds  │ Success rate: 60-85%         │
│ Retry overhead:  +1s sleep     │ Max attempts: 3              │
│ Database save:   <1 second     │ SQLite operations            │
└─────────────────────────────────────────────────────────────────┘
```

## Error Handling Points

```
┌─────────────────────────────────────────────────────────────────┐
│                        ERROR HANDLING                          │
├─────────────────────────────────────────────────────────────────┤
│ 1. Data Loading:     │ Fallback to sample data generation      │
│ 2. Prompt Loading:   │ Clear error messages + available list   │
│ 3. API Failures:     │ Logged with retry mechanism             │
│ 4. JSON Parsing:     │ Graceful fallback with error logging    │
│ 5. Validation Fails: │ Retry up to max_retries attempts        │
│ 6. Database Issues:  │ Transaction rollback + error logging    │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Parameters

```
┌─────────────────────────────────────────────────────────────────┐
│                       CONFIGURABLE SETTINGS                    │
├─────────────────────────────────────────────────────────────────┤
│ max_retries:           │ 3 (default)    │ MAX_RETRY_ATTEMPTS   │
│ min_validation_score:  │ 70 (default)   │ Moderator threshold  │
│ temperature settings:  │ 0.3, 0.4, 0.2  │ Doctor, Editor, Mod  │
│ document truncation:   │ 2000-3000 chars│ Fit within prompts   │
│ batch_size:           │ 5 (default)    │ Documents per batch   │
│ retry_sleep:          │ 1 second       │ Delay between retry   │
└─────────────────────────────────────────────────────────────────┘
```
