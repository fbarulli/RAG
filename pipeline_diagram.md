```mermaid
flowchart TD
    subgraph Core[src/rag_pipeline]
    llm_client[llm_client.py]
    logging[logging.py]
    paths[paths.py]
    schemas[schemas.py]
    end
    subgraph Stages[production_pipeline/]
        direction LR
        subgraph p01_data_cleaning[p01_data_cleaning]
        p01_data_cleaning_p01_download[p01_download.py]
        p01_data_cleaning_p02_parse[p02_parse.py]
        p01_data_cleaning_p01_download --> p01_data_cleaning_p02_parse
        p01_data_cleaning_p03_dedup[p03_dedup.py]
        p01_data_cleaning_p02_parse --> p01_data_cleaning_p03_dedup
        end
        subgraph p02_eda[p02_eda]
        p02_eda_p01_load_and_inspect[p01_load_and_inspect.py]
        end
        Clean -->|Read| p02_eda_p01_load_and_inspect
        p02_eda_p01_load_and_inspect -->|Write| Summary[(eda_summary.json)]
        subgraph p03_generation[p03_generation]
        p03_generation_p01_sample_docs[p01_sample_docs.py]
        p03_generation_p02_generate_queries[p02_generate_queries.py]
        p03_generation_p01_sample_docs --> p03_generation_p02_generate_queries
        p03_generation_p03_add_ood[p03_add_ood.py]
        p03_generation_p02_generate_queries --> p03_generation_p03_add_ood
        end
        subgraph p04_ingestion[p04_ingestion]
        p04_ingestion_p01_ingest_es[p01_ingest_es.py]
        p04_ingestion_p01_ingest_qdrant[p01_ingest_qdrant.py]
        p04_ingestion_p01_ingest_es --> p04_ingestion_p01_ingest_qdrant
        end
    end
    subgraph Tests[tests/]
    inspect_comparisons[inspect_comparisons.py]
    test_cleaning[test_cleaning.py]
    end
    p01_data_cleaning_p02_parse -.->|Verify| test_cleaning
    parsed -.->|Verify| inspect_comparisons
```