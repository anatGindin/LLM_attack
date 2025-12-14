import json
import pandas as pd

def convert_json_to_csv(file_path, output_csv):
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                task_data = json.loads(line)
                # Skip lines that have no completions
                if not task_data.get('result'):
                    continue
                prompt = task_data.get('prompt', '').replace('\n', ' ').strip()
                for entry in task_data.get('result', []):
                    extracted_code = entry.get('result', '').replace('\n', ' ').strip()
                    data.append({
                        'prompt': prompt,
                        'extracted_code': extracted_code
                    })
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"CSV file saved to {output_csv}")
    else:
        print("No data extracted. Please check the JSON structure.")

# Example usage
convert_json_to_csv('generated_outputs/exp3/codeLlamaResults/result3.json', 'generated_outputs/exp3/codeLlamaResults/result3.csv')
convert_json_to_csv('generated_outputs/exp3/codeGemmaResults/result3.json', 'generated_outputs/exp3/codeGemmaResults/result3.csv')
convert_json_to_csv('generated_outputs/exp3/deepSeekResults/result3.json', 'generated_outputs/exp3/deepSeekResults/result3.csv')
