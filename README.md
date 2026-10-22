# DraMaze

## Environment Setup

Use Python 3.10 or later. From the repository directory, create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Model API Configuration

Set your OpenAI-compatible API base URL, model ID, and API key:

```bash
export DRAMA_LLM_BASE_URL="https://your-api-provider.example/v1"
export DRAMA_LLM_MODEL="your-model-name"
export DRAMA_LLM_API_KEY="your-api-key"
```

## Run

The repository includes a sample microdrama idea in `example/idea.jsonl`. Run the complete pipeline:

```bash
bash example/run.sh
```

By default, this generates a 60-episode microdrama with a requested length of 700–1,000 words per episode. Each run creates a new directory under `output/`; the generated scripts are saved in its `drama/` subdirectory.

To specify the episode count, requested word range, and output directory explicitly:

```bash
bash example/run.sh \
  --episodes 40 \
  --min-words 700 \
  --max-words 1000 \
  --output-dir output/demo_40
```

Use a new output directory for each run.

To check input selection and run settings without making model API requests:

```bash
bash example/run.sh --dry-run
```

## Continue from Existing Outlines

To generate screenplays from an existing run directory containing completed episode outlines:

```bash
bash example/run.sh --from-outline output/existing_run --output-dir output/continued_run
```

This uses the episode count and word range saved with the outlines. Pass the run directory to `--from-outline`, not an individual outline JSON file.



