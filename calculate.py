import json
import sys

def calculate_average_from_jsonl(file_path, attribute_name):

    total_sum = 0
    count = 0
    line_num = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                if line_num < 0 or line_num > 80:
                    continue
                try:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if attribute_name in data:
                        value = data[attribute_name][0]
                        # Check if the value is a number (int or float)
                        if isinstance(value, (int, float)):
                            if attribute_name == "task_success":
                                if value >= 5:
                                    total_sum += 1
                            else:
                                total_sum += value
                            count += 1
                        else:
                            print(f"Warning: Attribute '{attribute_name}' in line {line_num} is not a number ({type(value)}). Skipping.", file=sys.stderr)
                    else:
                         print(f"Warning: Attribute '{attribute_name}' not found in line {line_num}. Skipping.", file=sys.stderr)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON on line {line_num}. Skipping: {line.strip()}", file=sys.stderr)
                except Exception as e:
                    pass


    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An error occurred during file processing: {e}", file=sys.stderr)
        return None

    if count > 0:
        average = total_sum / count
        return average
    else:
        print(f"Warning: No valid numeric values found for attribute '{attribute_name}' in the file.", file=sys.stderr)
        return None

# --- Configuration ---
jsonl_file_path = [
    'outputs/gpt-4o-mini_tool_True.jsonl',
    'outputs/gpt-4o-mini_tool_False.jsonl',
    'outputs/gpt-4o-mini_inject_True.jsonl',
    'outputs/gpt-4o-mini_inject_False.jsonl',
    'outputs/gpt-4o-mini_rag_True.jsonl',
    'outputs/gpt-4o-mini_rag_False.jsonl',
    'outputs/gemini-2.0-flash_inject_True.jsonl',
    'outputs/gemini-2.0-flash_inject_False.jsonl',
    'outputs/gemini-2.0-flash_tool_True.jsonl',
    'outputs/gemini-2.0-flash_tool_False.jsonl',
    'outputs/gemini-2.0-flash_rag_True.jsonl',
    'outputs/gemini-2.0-flash_rag_False.jsonl',
    'outputs/deepseek-v3_inject_False.jsonl',
    'outputs/deepseek-v3_tool_False.jsonl',
    'outputs/gpt-4o_inject_True.jsonl',
    'outputs/gpt-4o_inject_False.jsonl',
    'outputs/gpt-4o_tool_True.jsonl',
    'outputs/gpt-4o_tool_False.jsonl',
    'outputs/deepseek-v3_vanilla_False.jsonl',
    'outputs/deepseek-v3_rag_True.jsonl',
    'outputs/deepseek-v3_rag_False.jsonl'
]
attribute_to_average = [
    'step_score',
    'complete_score',
    'final_score',
    'task_success'
]

for a in attribute_to_average:
    for f in jsonl_file_path:
        average_value = calculate_average_from_jsonl(f, a)

        if average_value is not None:
            print(f"The average value of '{a}' in '{f}' is: {average_value}")
        else:
            print("Could not calculate the average.")
    print("--------------------------------------------------")