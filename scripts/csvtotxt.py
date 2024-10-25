import csv

# def csv_to_txt(input_csv, output_txt):
#     # Open the input CSV file
#     with open(input_csv, mode='r', newline='', encoding='utf-8') as csv_file:
#         # Open the output TXT file
#         with open(output_txt, mode='w', encoding='utf-8') as txt_file:
#             # Read the CSV content
#             csv_reader = csv.reader(csv_file)
            
#             # Iterate over each row in the CSV file
#             for row in csv_reader:
#                 # Assuming the second column contains the answers
#                 answer = row[1]
                
#                 # Write each answer to the TXT file
#                 txt_file.write(answer + '\n')

# #csv_to_txt('answers.csv', 'system_output_1.txt')
# csv_to_txt('answers_organize.csv', 'system_output_1.txt')

def csv_to_txt(input_csv, questions_txt, answers_txt):
    # Open the input CSV file
    with open(input_csv, mode='r', newline='', encoding='utf-8') as csv_file:
        # Read the CSV content
        csv_reader = csv.reader(csv_file)
        
        # Open the output TXT file for questions
        with open(questions_txt, mode='a', encoding='utf-8') as questions_file:
            # Open the output TXT file for answers
            with open(answers_txt, mode='a', encoding='utf-8') as answers_file:
                # Iterate over each row in the CSV file
                for row in csv_reader:
                    # Assuming the second column contains the questions and the third column contains the answers
                    question = row[1]
                    answer = row[2]
                    
                    # Write each question to the questions TXT file
                    questions_file.write(question + '\n')
                    
                    # Write each answer to the answers TXT file
                    answers_file.write(answer + '\n')


# Call the function with your file paths
#csv_to_txt('../../QA/QA_pair.csv', '../../submission_folder/data/testquestions.txt', '../../submission_folder/data/reference_answers.txt')\
csv_to_txt('../../QA/QA_references/1_Symphony.csv', '../../submission_folder/data/questions.txt', '../../submission_folder/data/reference_answers.txt')
csv_to_txt('../../QA/QA_references/2_Opera.csv', '../../submission_folder/data/questions.txt', '../../submission_folder/data/reference_answers.txt')
csv_to_txt('../../QA/QA_references/3_Trustarts.csv', '../../submission_folder/data/questions.txt', '../../submission_folder/data/reference_answers.txt')
csv_to_txt('../../QA/QA_references/Music_Symphony_ALL_Events_Musicians.csv', '../../submission_folder/data/questions.txt', '../../submission_folder/data/reference_answers.txt')