device="cuda:0"
show_progress_bar="--show_progress_bar"
normalize_embeddings="--normalize_embeddings"
directory_path="../data_collect/raw_documents"
query_file="../QA/test_questions.csv"
max_new_tokens_values=(128)
generator_batch_size_values=(1)
top_k_values=(1)

for max_new_tokens in "${max_new_tokens_values[@]}"; do
  for generator_batch_size in "${generator_batch_size_values[@]}"; do
    for top_k in "${top_k_values[@]}"; do

      output_file="../results/answers_max${max_new_tokens}_batch${generator_batch_size}_topk${top_k}.csv"

      echo "Running with max_new_tokens=$max_new_tokens, generator_batch_size=$generator_batch_size, top_k=$top_k"

      python rag.py \
          --device $device \
          $show_progress_bar \
          $normalize_embeddings \
          --query_file $query_file \
          --max_new_tokens $max_new_tokens \
          --generator_batch_size $generator_batch_size \
          --output_file $output_file \
          --directory_path $directory_path \
          --top_k $top_k

    done
  done
done