---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:9
- loss:MultipleNegativesRankingLoss
base_model: cross-encoder/ms-marco-TinyBERT-L2-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-TinyBERT-L2-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-TinyBERT-L2-v2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-TinyBERT-L2-v2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2) <!-- at revision 81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Setup: No development environment', 'Error:\n\n```plaintext\nThis project does not have a development environment configured. Please create a development environment and configure your development credentials to use the dbt IDE.\n```\n\nThe error message provides guidance on resolving this issue. Follow the guide in the dbt cloud setup documentation. Additional instructions can be found in the video @1:42.'],
]
scores = model.predict(pairs)
print(scores)
# [6.6184]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Setup: No development environment',
    [
        'Error:\n\n```plaintext\nThis project does not have a development environment configured. Please create a development environment and configure your development credentials to use the dbt IDE.\n```\n\nThe error message provides guidance on resolving this issue. Follow the guide in the dbt cloud setup documentation. Additional instructions can be found in the video @1:42.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 9 training samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 9 samples:
  |          | query                                                                              | positive                                                                             | negative                                                                             |
  |:---------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                             | string                                                                               | string                                                                               |
  | modality | text                                                                               | text                                                                                 | text                                                                                 |
  | details  | <ul><li>min: 10 tokens</li><li>mean: 14.67 tokens</li><li>max: 18 tokens</li></ul> | <ul><li>min: 42 tokens</li><li>mean: 200.78 tokens</li><li>max: 403 tokens</li></ul> | <ul><li>min: 42 tokens</li><li>mean: 134.56 tokens</li><li>max: 403 tokens</li></ul> |
* Samples:
  | query                                                                         | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
  |:------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Terraform: Teardown of BigQuery Dataset</code>                          | <code>When running terraform destroy, the following error can occur:<br><br>```Do you really want to destroy all resources?<br><br>Terraform will destroy all your managed infrastructure, as shown above.<br><br>There is no undo. Only 'yes' will be accepted to confirm.<br><br>Enter a value: yes<br><br>google_bigquery_dataset.homework_dataset: Destroying... [id=projects/terraform-demo-449214/datasets/homework_dataset]<br><br>╷<br><br>│ Error: Error when reading or editing Dataset: googleapi: Error 400: Dataset terraform-demo-449214:homework_dataset is still in use, resourceInUse<br>```<br><br>This is because the dataset is still in use by a table. To delete the dataset, set the delete_contents_on_destroy property to true in the main.tf file.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>This issue arises from the way deduplication is handled in two staging files.<br><br>Solution:<br><br>- Add an ORDER BY clause in the PARTITION BY section of both staging files.<br>- Continue adding columns to the ORDER BY clause until the row count in the fact_trips table is consistent upon re-running the model.<br><br>Explanation:<br><br>We partition by vendor_id and pickup_datetime, selecting the first row (rn=1) from these partitions. These partitions lack an order, so every execution might yield a different first row. The inconsistency leads to different rows being processed, possibly with or without an unknown borough. Consequently, the fact_trips model discards a varying number of rows based on the presence of unknown boroughs.</code> |
  | <code>Python: Ingestion with Jupyter notebook - missing 100000 records</code> | <code>If you follow the video 1.2.2 - Ingesting NY Taxi Data to Postgres and execute the same steps, you will ingest all the data (~1.3 million rows) into the table yellow_taxi_data. However, running the whole script in the Jupyter notebook for a second time from top to bottom will result in missing the first chunk of 100,000 records. This occurs because a call to the iterator appears before the while loop, leading to the second chunk being ingested first.<br><br>- Remove the cell df=next(df_iter) located higher up in the notebook than the while loop.<br>- Ensure the first w(df_iter) call is within the while loop.<br><br>📔 Note: The notebook is used to test the code and is not intended to be run top to bottom. The logic is organized in a later step when inserted into a .py file for the pipeline.</code>                                                                                                                                                                                                                                                                                                                                                                              | <code>This issue arises from the way deduplication is handled in two staging files.<br><br>Solution:<br><br>- Add an ORDER BY clause in the PARTITION BY section of both staging files.<br>- Continue adding columns to the ORDER BY clause until the row count in the fact_trips table is consistent upon re-running the model.<br><br>Explanation:<br><br>We partition by vendor_id and pickup_datetime, selecting the first row (rn=1) from these partitions. These partitions lack an order, so every execution might yield a different first row. The inconsistency leads to different rows being processed, possibly with or without an unknown borough. Consequently, the fact_trips model discards a varying number of rows based on the presence of unknown boroughs.</code> |
  | <code>Docker-Compose: PgAdmin – no database in PgAdmin</code>                 | <code>When you log into PgAdmin and see an empty database, the following solution can help:<br><br>Run:<br>   <br>```bash<br>docker-compose up<br>```<br><br>And at the same time run:<br><br>```bash<br>docker build -t taxi_ingest:v001 .<br><br># NETWORK NAME IS THE SAME AS THAT CREATED BY DOCKER COMPOSE<br>docker run -it \<br>  --network=pg-network \<br>  taxi_ingest:v001 \<br>  --user=postgres \<br>  --password=postgres \<br>  --host=db \<br>  --port=5432 \<br>  --db=ny_taxi \<br>  --table_name=green_tripdata \<br>  --url=${URL}<br>```<br><br>It's important to use the same --network as stated in the docker-compose.yaml file.<br><br>The docker-compose.yaml file might not specify a network, as shown below:<br><br>```yaml<br>services:<br>  db:<br>    container_name: postgres<br>    image: postgres:17-alpine<br>    environment:<br>      ...<br>    ports:<br>      - '5433:5432'<br>    volumes:<br>      - ...<br>  pgadmin:<br>    container_name: pgadmin<br>    image: dpage/pgadmin4:latest<br>    environment:<br>      ...<br>    ports:<br>      - '8080:80'<br>    volumes:<br>      - ...<br><br>volumes:<br>  vol-pgdata:<br>    name: vol-pgdata<br>  vol-pgadm...</code> | <code>You might have installed Docker via snap. Run the following command to verify:<br><br>```bash<br>sudo snap status docker<br>```<br><br>If you receive the response:<br><br>```error: unknown command "status", see 'snap help'.<br>```<br><br>Then uninstall Docker and install it via the official website.<br><br>Error message: "Bind for 0.0.0.0:5432 failed: port is already allocated."</code>                                                                                                                                                                                                                                                                                                                                                                            |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 10.0,
      "num_negatives": 4,
      "activation_fn": "torch.nn.modules.activation.Sigmoid"
  }
  ```

### Evaluation Dataset

#### Unnamed Dataset

* Size: 1 evaluation samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1 samples:
  |          | query                                                                          | positive                                                                          | negative                           |
  |:---------|:-------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:-----------------------------------|
  | type     | string                                                                         | string                                                                            | list                               |
  | modality | text                                                                           | text                                                                              |                                    |
  | details  | <ul><li>min: 7 tokens</li><li>mean: 7.0 tokens</li><li>max: 7 tokens</li></ul> | <ul><li>min: 77 tokens</li><li>mean: 77.0 tokens</li><li>max: 77 tokens</li></ul> | <ul><li>size: 5 elements</li></ul> |
* Samples:
  | query                                          | positive                                                                                                                                                                                                                                                                                                                                                                                                      | negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
  |:-----------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Setup: No development environment</code> | <code>Error:<br><br>```plaintext<br>This project does not have a development environment configured. Please create a development environment and configure your development credentials to use the dbt IDE.<br>```<br><br>The error message provides guidance on resolving this issue. Follow the guide in the dbt cloud setup documentation. Additional instructions can be found in the video @1:42.</code> | <code>["When you log into PgAdmin and see an empty database, the following solution can help:\n\nRun:\n   \n```bash\ndocker-compose up\n```\n\nAnd at the same time run:\n\n```bash\ndocker build -t taxi_ingest:v001 .\n\n# NETWORK NAME IS THE SAME AS THAT CREATED BY DOCKER COMPOSE\ndocker run -it \\\n  --network=pg-network \\\n  taxi_ingest:v001 \\\n  --user=postgres \\\n  --password=postgres \\\n  --host=db \\\n  --port=5432 \\\n  --db=ny_taxi \\\n  --table_name=green_tripdata \\\n  --url=${URL}\n```\n\nIt's important to use the same --network as stated in the docker-compose.yaml file.\n\nThe docker-compose.yaml file might not specify a network, as shown below:\n\n```yaml\nservices:\n  db:\n    container_name: postgres\n    image: postgres:17-alpine\n    environment:\n      ...\n    ports:\n      - '5433:5432'\n    volumes:\n      - ...\n  pgadmin:\n    container_name: pgadmin\n    image: dpage/pgadmin4:latest\n    environment:\n      ...\n    ports:\n      - '8080:80'\n    volumes:\n      - ...\n\nvolumes:\n  vol-pgdata:\n    name: vol-pgdata\n  vol-pgadmin_data:\n    name: vol-pgadmin_data\n```\n\nIf the network name is not specified, it is generated automatically: The name of the directory containing the docker-compose.yaml file in lowercase + _default.\n\nYou can find the network’s name when running docker-compose up:\n\n```pg-database Pulling pg-database Pulled \nNetwork week_1_default  Creating\nNetwork week_1_default  Created\n```", "When running terraform destroy, the following error can occur:\n\n```Do you really want to destroy all resources?\n\nTerraform will destroy all your managed infrastructure, as shown above.\n\nThere is no undo. Only 'yes' will be accepted to confirm.\n\nEnter a value: yes\n\ngoogle_bigquery_dataset.homework_dataset: Destroying... [id=projects/terraform-demo-449214/datasets/homework_dataset]\n\n╷\n\n│ Error: Error when reading or editing Dataset: googleapi: Error 400: Dataset terraform-demo-449214:homework_dataset is still in use, resourceInUse\n```\n\nThis is because the dataset is still in use by a table. To delete the dataset, set the delete_contents_on_destroy property to true in the main.tf file.", 'To read from multiple topics in the same Spark session, follow these steps:\n\n1. Initiate a Spark Session:\n   \n   ```python\nspark = (SparkSession\n       .builder\n       .appName(app_name)\n       .master(master=master)\n       .getOrCreate())\n   \n   spark.streams.resetTerminated()\n```\n\n2. Read Streams from Multiple Topics:\n   \n   ```python\nquery1 = spark\n       .readStream\n       ...\n       ...\n       .load()\n   \n   query2 = spark\n       .readStream\n       ...\n       ...\n       .load()\n   \n   query3 = spark\n       .readStream\n       ...\n       ...\n       .load()\n```\n\n3. Start the Queries:\n   \n   ```python\nquery1.start()\n   query2.start()\n   query3.start()\n```\n\n4. Await Termination:\n   \n   ```python\nspark.streams.awaitAnyTermination()  # Waits for any one of the queries to receive a kill signal or error failure. This is asynchronous.\n```\n\n   Note: query3.start().awaitTermination() is a blocking call. It works well when we are reading only from one topic.', 'The issue comes down to how Unix processes produce output and how Kestra interprets it:\n\n1. Python\'s logging module writes to stderr by default. If you, for instance, call logging.basicConfig() without specifying a stream argument, the root handler sends basically everything to stderr.\n\n2. Kestra maps the two standard streams to its own log levels. Anything the container writes to stdout becomes a Kestra DEBUG entry and anything written on stderr becomes ERROR. There is no middle ground.\n\nThe fix\n\nGood news is that the fix is simple: redirect Python logging to stdout:\n\n```python\nimport sys\nimport logging\n\nlogging.basicConfig(\n    level=logging.INFO,\n    format="%(asctime)s %(levelname)s %(message)s",\n    stream=sys.stdout,\n)\n```\n\nThat single stream=sys.stdout argument is enough. After the change, your informational messages will show up as DEBUG in Kestra.', 'This issue arises from the way deduplication is handled in two staging files.\n\nSolution:\n\n- Add an ORDER BY clause in the PARTITION BY section of both staging files.\n- Continue adding columns to the ORDER BY clause until the row count in the fact_trips table is consistent upon re-running the model.\n\nExplanation:\n\nWe partition by vendor_id and pickup_datetime, selecting the first row (rn=1) from these partitions. These partitions lack an order, so every execution might yield a different first row. The inconsistency leads to different rows being processed, possibly with or without an unknown borough. Consequently, the fact_trips model discards a varying number of rows based on the presence of unknown boroughs.']</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 10.0,
      "num_negatives": 4,
      "activation_fn": "torch.nn.modules.activation.Sigmoid"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `overwrite_output_dir`: True
- `num_train_epochs`: 2
- `warmup_steps`: 30
- `fp16`: True
- `dataloader_num_workers`: 2
- `load_best_model_at_end`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: True
- `do_predict`: False
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 30
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 2
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: True
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Time
- **Training**: 4.0 seconds

### Framework Versions
- Python: 3.11.15
- Sentence Transformers: 5.5.1
- Transformers: 4.57.6
- PyTorch: 2.12.0+cu130
- Accelerate: 1.13.0
- Datasets: 4.8.5
- Tokenizers: 0.22.2

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->