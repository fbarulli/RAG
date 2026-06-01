---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:868
- loss:MultipleNegativesRankingLoss
base_model: cross-encoder/ms-marco-TinyBERT-L2-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- map
- mrr@10
- ndcg@10
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-TinyBERT-L2-v2
  results:
  - task:
      type: cross-encoder-reranking
      name: Cross Encoder Reranking
    dataset:
      name: reranker eval
      type: reranker_eval
    metrics:
    - type: map
      value: 0.7900343642611684
      name: Map
    - type: mrr@10
      value: 0.7900343642611684
      name: Mrr@10
    - type: ndcg@10
      value: 0.8417912018989373
      name: Ndcg@10
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
    ['Homework: What if my answer is not exactly the same as the choices presented?', 'Please choose the closest one to your answer. Also, do not post your answer in the course Slack channel.'],
    ['Size limit when uploading to GitHub', 'To manage size limits effectively when uploading to GitHub, add the mlruns and artifacts directories to your .gitignore, like this:\n\n```02-experiment-tracking/mlruns\n02-experiment-tracking/runnin-mflow-examples/mlruns\n02-experiment-tracking/homework/mlruns\n02-experiment-tracking/homework/artifacts\n```'],
    ['Grafana dashboard error after reset: db query error: pq: database “test” does not exist', 'Problem: You’ve already loaded your data, created a dashboard, and saved it. However, upon running docker-compose up after saving the dashboard, you encounter this error: \n\n```plaintext\ndb query error: pq: database “test” does not exist\n```\n\nSolution:\n\nThis error indicates you haven’t run the DB initialization code. If you did run it before and even saw results, the issue likely arises because you restarted the docker-compose services.\n\nThe default docker-compose.yml file doesn’t have a volume for the Postgres DB. This means every restart will delete the DB data.\n\nTo resolve this:\n\n1. If not planning to restart the services again: Simply rerun the DB initialization and filling code of your exercise.\n\n2. If you plan to restart services frequently:\n   - Add a volume to your PostgreSQL service in the docker-compose.yml file:\n\n     ```yaml\nvolumes:\n       - ./data/postgres:/var/lib/postgresql/data\n```\n\n   - Note: Ensure you create a ./data directory in your project.\n\n3. To attach the volume, run the following:\n\n   ```bash\ndocker-compose down\n   docker-compose up --build\n```'],
    ['How is my capstone project going to be evaluated?', 'Each submitted project will be evaluated by three randomly assigned students who have also submitted the project.\n\nYou will also be responsible for grading the projects of three fellow students yourself. Please be aware that not complying with this rule will result in failing to achieve the Certificate at the end of the course.\n\nThe final grade you receive will be the median score of the grades from the peer reviewers.\n\nThe peer review criteria for evaluation must follow the guidelines defined here (TBA for link).'],
    ['Multiline commands in Windows Powershell', 'To use multiline commands in Windows PowerShell, place a backtick () at the end of each line except the last. Note that multiline strings do not require a backtick.\n\n- Escape double quotes (") to "\\\n- Use $env:` to create environment variables (non-persistent). For example:\n\n```powershell\n$env:KINESIS_STREAM_INPUT="ride_events"\n\naws kinesis put-record --cli-binary-format raw-in-base64-out `\n\n--stream-name $env:KINESIS_STREAM_INPUT `\n\n--partition-key 1 `\n\n--data \'{\n\n\\"ride\\": {\n\n\\"PULocationID\\": 130,\n\n\\"DOLocationID\\": 205,\n\n\\"trip_distance\\": 3.66\n\n},\n\n\\"ride_id\\": 156\n\n}\'\n```'],
]
scores = model.predict(pairs)
print(scores)
# [-1.591   8.4191  6.8743  3.0805  8.0556]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Homework: What if my answer is not exactly the same as the choices presented?',
    [
        'Please choose the closest one to your answer. Also, do not post your answer in the course Slack channel.',
        'To manage size limits effectively when uploading to GitHub, add the mlruns and artifacts directories to your .gitignore, like this:\n\n```02-experiment-tracking/mlruns\n02-experiment-tracking/runnin-mflow-examples/mlruns\n02-experiment-tracking/homework/mlruns\n02-experiment-tracking/homework/artifacts\n```',
        'Problem: You’ve already loaded your data, created a dashboard, and saved it. However, upon running docker-compose up after saving the dashboard, you encounter this error: \n\n```plaintext\ndb query error: pq: database “test” does not exist\n```\n\nSolution:\n\nThis error indicates you haven’t run the DB initialization code. If you did run it before and even saw results, the issue likely arises because you restarted the docker-compose services.\n\nThe default docker-compose.yml file doesn’t have a volume for the Postgres DB. This means every restart will delete the DB data.\n\nTo resolve this:\n\n1. If not planning to restart the services again: Simply rerun the DB initialization and filling code of your exercise.\n\n2. If you plan to restart services frequently:\n   - Add a volume to your PostgreSQL service in the docker-compose.yml file:\n\n     ```yaml\nvolumes:\n       - ./data/postgres:/var/lib/postgresql/data\n```\n\n   - Note: Ensure you create a ./data directory in your project.\n\n3. To attach the volume, run the following:\n\n   ```bash\ndocker-compose down\n   docker-compose up --build\n```',
        'Each submitted project will be evaluated by three randomly assigned students who have also submitted the project.\n\nYou will also be responsible for grading the projects of three fellow students yourself. Please be aware that not complying with this rule will result in failing to achieve the Certificate at the end of the course.\n\nThe final grade you receive will be the median score of the grades from the peer reviewers.\n\nThe peer review criteria for evaluation must follow the guidelines defined here (TBA for link).',
        'To use multiline commands in Windows PowerShell, place a backtick () at the end of each line except the last. Note that multiline strings do not require a backtick.\n\n- Escape double quotes (") to "\\\n- Use $env:` to create environment variables (non-persistent). For example:\n\n```powershell\n$env:KINESIS_STREAM_INPUT="ride_events"\n\naws kinesis put-record --cli-binary-format raw-in-base64-out `\n\n--stream-name $env:KINESIS_STREAM_INPUT `\n\n--partition-key 1 `\n\n--data \'{\n\n\\"ride\\": {\n\n\\"PULocationID\\": 130,\n\n\\"DOLocationID\\": 205,\n\n\\"trip_distance\\": 3.66\n\n},\n\n\\"ride_id\\": 156\n\n}\'\n```',
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

## Evaluation

### Metrics

#### Cross Encoder Reranking

* Dataset: `reranker_eval`
* Evaluated with [<code>CERerankingEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CERerankingEvaluator) with these parameters:
  ```json
  {
      "at_k": 10
  }
  ```

| Metric      | Value      |
|:------------|:-----------|
| map         | 0.79       |
| mrr@10      | 0.79       |
| **ndcg@10** | **0.8418** |

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

* Size: 868 training samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 100 samples:
  |          | query                                                                             | positive                                                                            | negative                                                                             |
  |:---------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                            | string                                                                              | string                                                                               |
  | modality | text                                                                              | text                                                                                | text                                                                                 |
  | details  | <ul><li>min: 7 tokens</li><li>mean: 20.67 tokens</li><li>max: 65 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 165.11 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 13 tokens</li><li>mean: 173.58 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | query                                                                         | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
  |:------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Terraform: Teardown of BigQuery Dataset</code>                          | <code>When running terraform destroy, the following error can occur:<br><br>```Do you really want to destroy all resources?<br><br>Terraform will destroy all your managed infrastructure, as shown above.<br><br>There is no undo. Only 'yes' will be accepted to confirm.<br><br>Enter a value: yes<br><br>google_bigquery_dataset.homework_dataset: Destroying... [id=projects/terraform-demo-449214/datasets/homework_dataset]<br><br>╷<br><br>│ Error: Error when reading or editing Dataset: googleapi: Error 400: Dataset terraform-demo-449214:homework_dataset is still in use, resourceInUse<br>```<br><br>This is because the dataset is still in use by a table. To delete the dataset, set the delete_contents_on_destroy property to true in the main.tf file.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>The most commonly hit BigQuery limits in the course:<br><br>- Partition columns per table: 1. You cannot partition by multiple columns. (docs)<br>- Cluster columns per table: up to 4. You can cluster on a tuple of columns up to that limit. (docs)<br>- Partitions per table: 10,000 (older docs and the course playlist may say 4,000 — that limit was raised). (docs)<br>- Partitions modified by a single job: 4,000. A single load/query/copy/DML job can't touch more than 4,000 partitions at once.<br><br>Implications for time-based partitioning under the 10,000 partition limit:<br><br>- Daily partitions cover ~27 years.<br>- Hourly partitions cover ~416 days (just over a year).<br>- Monthly partitions cover over 800 years.<br><br>So daily partitioning is fine for almost any workload; hourly partitioning needs a retention strategy if your data goes back more than ~1 year.</code>                                                                                                                                                                                     |
  | <code>Python: Ingestion with Jupyter notebook - missing 100000 records</code> | <code>If you follow the video 1.2.2 - Ingesting NY Taxi Data to Postgres and execute the same steps, you will ingest all the data (~1.3 million rows) into the table yellow_taxi_data. However, running the whole script in the Jupyter notebook for a second time from top to bottom will result in missing the first chunk of 100,000 records. This occurs because a call to the iterator appears before the while loop, leading to the second chunk being ingested first.<br><br>- Remove the cell df=next(df_iter) located higher up in the notebook than the while loop.<br>- Ensure the first w(df_iter) call is within the while loop.<br><br>📔 Note: The notebook is used to test the code and is not intended to be run top to bottom. The logic is organized in a later step when inserted into a .py file for the pipeline.</code>                                                                                                                                                                                                                                                                                                                                                                              | <code>```python<br>os.system(f"curl -LO {url} -o {csv_name}")<br>```</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  | <code>Docker-Compose: PgAdmin – no database in PgAdmin</code>                 | <code>When you log into PgAdmin and see an empty database, the following solution can help:<br><br>Run:<br>   <br>```bash<br>docker-compose up<br>```<br><br>And at the same time run:<br><br>```bash<br>docker build -t taxi_ingest:v001 .<br><br># NETWORK NAME IS THE SAME AS THAT CREATED BY DOCKER COMPOSE<br>docker run -it \<br>  --network=pg-network \<br>  taxi_ingest:v001 \<br>  --user=postgres \<br>  --password=postgres \<br>  --host=db \<br>  --port=5432 \<br>  --db=ny_taxi \<br>  --table_name=green_tripdata \<br>  --url=${URL}<br>```<br><br>It's important to use the same --network as stated in the docker-compose.yaml file.<br><br>The docker-compose.yaml file might not specify a network, as shown below:<br><br>```yaml<br>services:<br>  db:<br>    container_name: postgres<br>    image: postgres:17-alpine<br>    environment:<br>      ...<br>    ports:<br>      - '5433:5432'<br>    volumes:<br>      - ...<br>  pgadmin:<br>    container_name: pgadmin<br>    image: dpage/pgadmin4:latest<br>    environment:<br>      ...<br>    ports:<br>      - '8080:80'<br>    volumes:<br>      - ...<br><br>volumes:<br>  vol-pgdata:<br>    name: vol-pgdata<br>  vol-pgadm...</code> | <code>This error means your container is looking for another service by name on a Docker network, but they aren't on the same network. Common variants:<br><br>```sqlalchemy.exc.OperationalError: could not translate host name "pgdatabase" to address: Name or service not known<br>Unable to connect to server: could not translate host name 'pg-database' to address: Name does not resolve<br>network <hash> not found<br>```<br><br>What's happening<br><br>Docker network DNS only resolves service names within the same network. Two reasons it might fail:<br><br>1. The ingestion container was started with --network  but  doesn't match the network compose actually created. By default, docker compose creates a network named after the project directory plus _default (e.g. 2docker_default).<br><br>2. Your ingestion script is hardcoded to use a host name like pgdatabase, but the compose service is actually called pgdatabase-1, or you're running the script outside Docker entirely.<br><br>1. List networks and confirm the actual name compose created:<br><br>  ...</code> |
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

* Size: 97 evaluation samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 97 samples:
  |          | query                                                                             | positive                                                                            | negative                                                                            |
  |:---------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type     | string                                                                            | string                                                                              | string                                                                              |
  | modality | text                                                                              | text                                                                                | text                                                                                |
  | details  | <ul><li>min: 6 tokens</li><li>mean: 17.85 tokens</li><li>max: 62 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 117.78 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 131.94 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | query                                                                                                | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
  |:-----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Homework: What if my answer is not exactly the same as the choices presented?</code>           | <code>Please choose the closest one to your answer. Also, do not post your answer in the course Slack channel.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | <code>Deadlines differ for participants in different time zones — the cutoff is midnight Berlin time on the published date, and whatever corresponds to that in your time zone. The exact deadline for each homework is shown on the homework submission page:<br><br>Homework Submission Page</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
  | <code>Size limit when uploading to GitHub</code>                                                     | <code>To manage size limits effectively when uploading to GitHub, add the mlruns and artifacts directories to your .gitignore, like this:<br><br>```02-experiment-tracking/mlruns<br>02-experiment-tracking/runnin-mflow-examples/mlruns<br>02-experiment-tracking/homework/mlruns<br>02-experiment-tracking/homework/artifacts<br>```</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | <code>In case you want to set up a GitHub repository (e.g., for homeworks) from a remote VM, you can follow these helpful tutorials:<br><br>- Setting up GitHub on AWS instance: Tutorial<br>- Setting up keys on AWS instance: GitHub Documentation<br><br>Once you complete these steps, you should be able to push to your repository successfully.<br><br>AWS Instance Note:<br><br>The selected AWS instance may not be covered under the free tier due to its size or other factors. Here is what the AWS free tier includes:<br><br>- Resizable compute capacity in the Cloud.<br>- 750 hours per month of Linux, RHEL, or SLES t2.micro or t3.micro instance, depending on the region.<br>- 750 hours per month of Windows t2.micro or t3.micro instance, depending on the region.<br>- 750 hours per month of public IPv4 address regardless of the instance type.<br><br>*Instances launch in Unlimited mode and may incur additional charges.</code> |
  | <code>Grafana dashboard error after reset: db query error: pq: database “test” does not exist</code> | <code>Problem: You’ve already loaded your data, created a dashboard, and saved it. However, upon running docker-compose up after saving the dashboard, you encounter this error: <br><br>```plaintext<br>db query error: pq: database “test” does not exist<br>```<br><br>Solution:<br><br>This error indicates you haven’t run the DB initialization code. If you did run it before and even saw results, the issue likely arises because you restarted the docker-compose services.<br><br>The default docker-compose.yml file doesn’t have a volume for the Postgres DB. This means every restart will delete the DB data.<br><br>To resolve this:<br><br>1. If not planning to restart the services again: Simply rerun the DB initialization and filling code of your exercise.<br><br>2. If you plan to restart services frequently:<br>   - Add a volume to your PostgreSQL service in the docker-compose.yml file:<br><br>     ```yaml<br>volumes:<br>       - ./data/postgres:/var/lib/postgresql/data<br>```<br><br>   - Note: Ensure you create a ./data directory in your project.<br><br>3. To attach the volum...</code> | <code>When trying to log in to Grafana with the standard credentials (admin/admin), an error occurs.<br><br>1. To reset the admin password, use the following command inside the Grafana container:<br><br>   ```bash<br>grafana cli admin reset-admin-password admin<br>```<br><br>   Note: The grafana-cli command is deprecated. Use grafana cli instead.<br><br>2. Enter the Docker container with Grafana:<br><br>   - Find the Container ID by running:<br>     <br>     ```bash<br>docker ps<br>```<br><br>   - Use the Container ID to reset the password. Replace  with the actual Container ID:<br><br>     ```bash<br>lpep_pickup_datetime<container_ID> grafana cli admin reset-admin-password admin<br>```<br><br>This should resolve the login issue.</code>                                                                                                                                                                                      |
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

### Training Logs
| Epoch      | Step    | Training Loss | Validation Loss | reranker_eval_ndcg@10 |
|:----------:|:-------:|:-------------:|:---------------:|:---------------------:|
| 0.0917     | 10      | 1.3311        | -               | -                     |
| 0.1835     | 20      | 0.8854        | -               | -                     |
| 0.2752     | 30      | 1.0085        | -               | -                     |
| 0.3670     | 40      | 0.9846        | -               | -                     |
| 0.4587     | 50      | 0.9537        | -               | -                     |
| 0.5505     | 60      | 1.1537        | -               | -                     |
| 0.6422     | 70      | 0.761         | -               | -                     |
| 0.7339     | 80      | 0.6916        | -               | -                     |
| 0.8257     | 90      | 0.9746        | -               | -                     |
| 0.9174     | 100     | 1.0063        | 0.8967          | 0.8428                |
| 1.0092     | 110     | 0.6202        | -               | -                     |
| 1.1009     | 120     | 0.8927        | -               | -                     |
| 1.1927     | 130     | 0.8488        | -               | -                     |
| 1.2844     | 140     | 1.1571        | -               | -                     |
| 1.3761     | 150     | 0.5108        | -               | -                     |
| 1.4679     | 160     | 0.5365        | -               | -                     |
| 1.5596     | 170     | 0.698         | -               | -                     |
| 1.6514     | 180     | 0.8295        | -               | -                     |
| 1.7431     | 190     | 0.9728        | -               | -                     |
| **1.8349** | **200** | **0.5471**    | **0.7838**      | **0.8418**            |
| 1.9266     | 210     | 0.6577        | -               | -                     |

* The bold row denotes the saved checkpoint.

### Training Time
- **Training**: 5.8 minutes
- **Evaluation**: 18.0 seconds
- **Total**: 6.1 minutes

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