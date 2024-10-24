import re
import collections
import string
import pandas as pd
import argparse
import os
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

### 
def evaluate(reference_answer_file, generated_answer_file, output_file):
    if reference_answer_file.endswith('txt'): # txt
      with open(reference_answer_file, 'r') as f:
          reference_answers = f.readlines() # list of string
      with open(generated_answer_file, 'r') as f:
          generated_answers = f.readlines() # list of string
    else: # csv
      try:
        reference_df = pd.read_csv(reference_answer_file)
        generated_df = pd.read_csv(generated_answer_file, header=None)
        generated_df.columns = ['Question', 'Generated Answer']
        merged_df = pd.merge(reference_df, generated_df, on='Question')
        reference_answers = merged_df['Reference Answer'].tolist()
        generated_answers = merged_df['Generated Answer'].tolist()
      except:
        cleaned_lines = []
        with open(generated_answer_file, 'r') as f:
              for line in f:
                cleaned_line = re.sub(r'(?<!^)(?<!,)\"(?!,)(?!$)', '', line)
                cleaned_lines.append(cleaned_line)
        with open('Cleaned_GeneratedAnswer.csv', 'w') as f:
              f.writelines(cleaned_lines)

        generated_df = pd.read_csv('Cleaned_GeneratedAnswer.csv', header=None)
        reference_df = pd.read_csv(reference_answer_file)
        generated_df.columns = ['Question', 'Generated Answer']
        merged_df = pd.merge(reference_df, generated_df, on='Question')
        merged_df.to_csv('merged.csv')
        reference_answers = merged_df['Reference Answer'].tolist()
        generated_answers = merged_df['Generated Answer'].tolist()

      
    assert len(reference_answers) == len(generated_answers)
    
    total = len(reference_answers)
    
    # metrics: exact match, recall, precision, f1
    exact_match_total = 0
    precision_total = 0
    recall_total = 0
    f1_total = 0
    
    for ref_answer, gen_answer in zip(reference_answers, generated_answers):
        ref_answer = str(ref_answer)
        gen_answer = str(gen_answer)
        
        # Compute exact match
        exact_match_total += compute_exact(ref_answer.strip(), gen_answer.strip())
        
        # Compute tokens
        gold_toks = get_tokens(ref_answer.strip())
        pred_toks = get_tokens(gen_answer.strip())
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        
        # Compute precision and recall
        if len(pred_toks) > 0:
            precision = 1.0 * num_same / len(pred_toks)
        else:
            precision = 0
        if len(gold_toks) > 0:
            recall = 1.0 * num_same / len(gold_toks)
        else:
            recall = 0
            
        precision_total += precision
        recall_total += recall
        f1_total += compute_f1(ref_answer.strip(), gen_answer.strip())

    exact_match_avg = 100.0 * exact_match_total / total
    precision_avg = 100.0 * precision_total / total
    recall_avg = 100.0 * recall_total / total
    f1_avg = 100.0 * f1_total / total

    print(f"output_file: {output_file}")
    print(f"Exact Match: {exact_match_avg:.2f}%")
    print(f"Precision: {precision_avg:.2f}%")
    print(f"Recall: {recall_avg:.2f}%")
    print(f"F1 Score: {f1_avg:.2f}%")
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as f:
        f.write(f"Exact Match: {exact_match_avg:.2f}%\n")
        f.write(f"Precision: {precision_avg:.2f}%\n")
        f.write(f"Recall: {recall_avg:.2f}%\n")
        f.write(f"F1 Score: {f1_avg:.2f}%\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_answer_file', type=str, required=True)
    parser.add_argument('--generated_answer_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    evaluate(args.reference_answer_file, args.generated_answer_file, args.output_file)