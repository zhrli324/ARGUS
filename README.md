# ARGUS

Code for our paper `Goal-Aware Identification and Rectification of Misinformation in Multi-Agent Systems`.

## Install

Clone this repo and create virtual env
```shell
git clone https://github.com/zhrli324/ARGUS.git
cd ARGUS
conda create -n ARGUS python=3.9 -y
conda activate ARGUS
```

Install requirements
```shell
pip install -r requirements.txt
python prepare.py
```

## Configuration

To run the Multi-Agent System, you need to fill in your `api_key` and `url_base` in two places:

In `config/api_keys.yaml`:

```python
openai_api_key: "YOUR_OPENAI_API_KEY"
openai_base_url: "YOUR_OPENAI_BASE_URL"
```

In `sandbox/llm.py`:

```python
MODEL_LIST = {
    "gpt-4o-mini": ...,
    "gpt-4o": ...,
    "gemini-2.0-flash": ...,
    "claude-3.5-haiku": ...,
    "deepseek-v3": ...,
}
```

## Quick Start

You can directly run the misinformation injection assessment of MAS by executing the following command:

```shell
python run.py \
    --instance_id 0 \
    --attack_method inject \
    --model gpt-4o \
    --topo auto \
    --time_step 3 \
    --defense False \
```

The parameters are configured as follows:

- `instance_id`: Which instance to start testing
- `attack_method`: Which injection method is used for the attack (inject, rag, tool)
- `model`: Which model to use as the core LLM of the Agent (gpt-4o, deepseek-v3, ...)
- `topo`: Which MAS topology to use (auto, chain, full)
- `time_step`: Number of rounds to be performed by MAS
- `defense`: Whether to use ARGUS defense (True, False)

## Outputs

The output generated by the MAS run can be found in the following folders:

- `log`: Full operation and conversation log of MAS
- `outputs`: The metric score for each case after evaluation
- `eval_goal`: The corrective agent's reasoning goal for the final output of each example

You can refer to the README.md in each folder for more information.

You can freely modify calculate.py to perform the final indicator calculation
