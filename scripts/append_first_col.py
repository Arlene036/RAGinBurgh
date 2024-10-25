import csv

def append_first_column(input_csv, existing_csv, output_csv):
    # Read the input CSV (the one from which we will take the first column)
    with open(input_csv, mode='r', newline='', encoding='utf-8') as input_file:
        input_reader = csv.reader(input_file)
        input_first_column = [row[0] for row in input_reader]  # Extract the first column

    # Read the existing CSV (the one to which we will append the first column)
    with open(existing_csv, mode='r', newline='', encoding='utf-8') as existing_file:
        existing_reader = csv.reader(existing_file)
        existing_data = [row for row in existing_reader]  # Read all data
    
    # Ensure that input and existing CSVs have the same number of rows
    if len(input_first_column) != len(existing_data):
        raise ValueError("The number of rows in the input CSV and existing CSV must match.")

    # Modify existing data by shifting columns
    updated_data = []
    for idx, row in enumerate(existing_data):
        # Ensure that there's at least one column in the existing row
        new_row = [input_first_column[idx]] + row  # Insert the first column from input CSV to the front of the existing row
        updated_data.append(new_row)

    # Write the updated data to the output CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(updated_data)

# Example usage
#append_first_column('../QA/QA_pair.csv', '../results/QA_pair/max_new_tokens128/model_topk1/retri_k1/compression/fewshot/create_bm25/non_filter/answers.csv', '../results/QA_pair/max_new_tokens128/model_topk1/retri_k1/compression/fewshot/create_bm25/non_filter/answers_with_time_sensitive.csv')
append_first_column('../QA/QA_pair.csv', '../results/QA_pair/non-rag/answers.csv', '../results/QA_pair/non-rag/answers_with_time_sensitive.csv')
