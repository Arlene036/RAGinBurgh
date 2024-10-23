device="cuda:0"
show_progress_bar="--show_progress_bar"
normalize_embeddings="--normalize_embeddings"
directory_path="../data_collect/raw_documents"
query_file="../QA/questions.txt" 
max_new_tokens_values=(128)
k=(5 10)
top_k_values=(1)

compression_retriever_values=("" "--compression_retriever")
few_shot_values=("" "--few_shot")

for max_new_tokens in "${max_new_tokens_values[@]}"; do
  for k in "${k[@]}"; do
    for top_k in "${top_k_values[@]}"; do
      for compression_retriever in "${compression_retriever_values[@]}"; do
        for few_shot in "${few_shot_values[@]}"; do
        
          output_dir="../results/QA_Music_Symphony/max_new_tokens${max_new_tokens}/model_topk${top_k}/retri_k${k}"
          if [ -n "$compression_retriever" ]; then
            output_dir="${output_dir}/compression"
          else
            output_dir="${output_dir}/non_compression"
          fi
          
          if [ -n "$few_shot" ]; then
            output_dir="${output_dir}/fewshot"
          else
            output_dir="${output_dir}/non_fewshot"
          fi
          
          mkdir -p $output_dir 
          output_file="${output_dir}/answers.csv"

          echo "Running with max_new_tokens=$max_new_tokens, generator_batch_size=$generator_batch_size, top_k=$top_k, compression_retriever=${compression_retriever}, few_shot=${few_shot}"

          python rag.py \
              --device $device \
              --generator "mistralai/Mistral-7B-Instruct-v0.2" \
              --faiss_output_dir "rag_faiss_index_10230232" \
              $show_progress_bar \
              $normalize_embeddings \
              $compression_retriever \
              $few_shot \
              --query_file $query_file \
              --max_new_tokens $max_new_tokens \
              --output_file $output_file \
              --directory_path $directory_path \
              --top_k $top_k


        done
      done
    done
  done
done