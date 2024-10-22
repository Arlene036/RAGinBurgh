import re

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
    with open(reference_answer_file, 'r') as f:
        reference_answers = f.readlines() # list of string
    with open(generated_answer_file, 'r') as f:
        generated_answers = f.readlines() # list of string
    
    assert len(reference_answers) == len(generated_answers)
    
    total = len(reference_answers)
    
    # metrics: exact match, recall, precision, f1
    exact_match_total = 0
    f1_total = 0
    
    for ref_answer, gen_answer in zip(reference_answers, generated_answers):
        exact_match_total += compute_exact(ref_answer.strip(), gen_answer.strip())
        f1_total += compute_f1(ref_answer.strip(), gen_answer.strip())

    exact_match_avg = 100.0 * exact_match_total / total
    f1_avg = 100.0 * f1_total / total

    print(f"Exact Match: {exact_match_avg:.2f}%")
    print(f"F1 Score: {f1_avg:.2f}%")
    
    with open(output_file, 'w') as f:
        f.write(f"Exact Match: {exact_match_avg:.2f}%\n")
        f.write(f"F1 Score: {f1_avg:.2f}%\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_answer_file', type=str, required=True)
    parser.add_argument('--generated_answer_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    evaluate(args.reference_answer_file, args.generated_answer_file, args.output_file)