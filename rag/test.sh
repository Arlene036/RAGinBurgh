device="cuda:0"
show_progress_bar="--show_progress_bar"
normalize_embeddings="--normalize_embeddings"
directory_path="../data_collect/raw_documents"
query_file="../QA/QA_pair.csv" 
max_new_tokens_values=(128)
k=(1) # 10
top_k_values=(1)

compression_retriever_values=("--compression_retriever") # "") #"")
few_shot_values=("--few_shot") # "") # "" 
create_bm25_values=("--create_bm25")
filter_values=("--filter")

for max_new_tokens in "${max_new_tokens_values[@]}"; do
  for k in "${k[@]}"; do
    for top_k in "${top_k_values[@]}"; do
      for compression_retriever in "${compression_retriever_values[@]}"; do
        for few_shot in "${few_shot_values[@]}"; do
          for create_bm25 in "${create_bm25_values[@]}"; do
            for filter in "${filter_values[@]}"; do
        
              output_dir="../results/QA_pair/max_new_tokens${max_new_tokens}/model_topk${top_k}/retri_k${k}"
              eval_output_dir="../eval_results/QA_pair/max_new_tokens${max_new_tokens}/model_topk${top_k}/retri_k${k}"

              if [ -n "$compression_retriever" ]; then
                output_dir="${output_dir}/compression"
                eval_output_dir="${eval_output_dir}/compression"
              else
                output_dir="${output_dir}/non_compression"
                eval_output_dir="${eval_output_dir}/non_compression"
              fi
              
              if [ -n "$few_shot" ]; then
                output_dir="${output_dir}/fewshot"
                eval_output_dir="${eval_output_dir}/fewshot"
              else
                output_dir="${output_dir}/non_fewshot"
                eval_output_dir="${eval_output_dir}/non_fewshot"
              fi

              if [ -n "$create_bm25" ]; then
                output_dir="${output_dir}/create_bm25"
                eval_output_dir="${eval_output_dir}/create_bm25"
              else
                output_dir="${output_dir}/non_create_bm25"
                eval_output_dir="${eval_output_dir}/non_create_bm25"
              fi

              if [ -n "$filter" ]; then
                output_dir="${output_dir}/filter"
                eval_output_dir="${eval_output_dir}/filter"
              else
                output_dir="${output_dir}/non_filter"
                eval_output_dir="${eval_output_dir}/non_filter"
              fi
              
              mkdir -p $output_dir 
              mkdir -p $eval_output_dir 

              output_file="${output_dir}/answers.csv"
              eval_output_dir="${eval_output_dir}/metric.txt"

              echo "Running with max_new_tokens=$max_new_tokens, top_k=$top_k, compression_retriever=${compression_retriever}, few_shot=${few_shot}"

              python rag.py \
                  --device $device \
                  $create_bm25 \
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
              
              python evaluation.py --reference_answer_file $query_file \
                      --generated_answer_file $output_file \
                      --output_file $eval_output_dir

            done
          done
        done
      done
    done
  done
done