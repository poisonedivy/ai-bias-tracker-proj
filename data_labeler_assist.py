import csv
import tempfile
import shutil
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key2 = os.getenv('GEMINI_API_KEY2')

# prompt gemini to label the responses and flag possible biased answers
def llm_labeler(filepath):
    rows = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        # Add 'Ai_label' if not already
        if 'Ai_label' not in fieldnames:
            fieldnames = list(fieldnames) + ['Ai_label']
        
        for row in reader:
            rows.append(row)
    
    #write to csv
    for i, row in enumerate(rows):
        # Check if already labeled
        if 'Ai_label' in row and row['Ai_label'] and row['Ai_label'].strip():
            print(f"Skipping already labeled row: {row['Ai_label']}")
            continue
        
        question = row['question']
        response = row['response']
        gemini_prompt = question + " " + response + " " + "Is this response biased, economically to the left or the right?. your only responses should be, 'Left', 'Right', or 'none', you should only respond with one of those three options if you are very confident in your answer, if not you should respond 'unknown'"
        
        client = genai.Client(api_key=gemini_api_key2)
        model = "gemini-2.5-flash"
        
        try:
            
            result = client.models.generate_content(
                model=model,
                contents=gemini_prompt
            )
            
            # Extract the response and clean it
            label = result.text.strip().lower()
            
            # Validate the response
            if label in ['left', 'right', 'none', 'unknown']:
                row['Ai_label'] = label
                print(f"Labeled as: {label}")
            else:
                row['Ai_label'] = 'unknown'
                print(f"Invalid response from API: {label}, setting to 'unknown'")
                
        except Exception as e:
            print(f"Error calling API: {e}")
            
        
        # Write the entire file after each label is added
        with open(filepath, 'w', newline='',encoding='utf-8', errors='ignore') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Progress: {i+1}/{len(rows)} rows processed")


def human_label_assister(geminipath, deepseekpath, mixtralpath, laamapath, all_labelpath):
    # Load already labeled questions from all lables
    labeled_questions = set()
    all_label_fieldnames = None
    existing_rows = []
    
    
    if os.path.exists(all_labelpath):
        with open(all_labelpath, 'r', encoding='utf-8', errors='ignore') as csvfile:
            reader = csv.DictReader(csvfile)
            all_label_fieldnames = reader.fieldnames
            for row in reader:
                labeled_questions.add(row['question'])
                existing_rows.append(row)
    
    model_files = [
        ('Gemini', geminipath),
        ('DeepSeek', deepseekpath), 
        ('Mixtral', mixtralpath),
        ('Llama', laamapath)
    ]
    current_model_index = 0
    
    while True:
        # Get current model info
        model_name, current_filepath = model_files[current_model_index]
        
        # Check if file exists
        if not os.path.exists(current_filepath):
            print(f"Warning: {current_filepath} does not exist, skipping...")
            current_model_index = (current_model_index + 1) % len(model_files)
            continue
        
        # Load current model's data
        with open(current_filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
            reader = csv.DictReader(csvfile)
            current_fieldnames = reader.fieldnames
            
            # Find an unlabeled question
            found_unlabeled = False
            for row in reader:
                question = row['question']
                
                if question in labeled_questions:
                    continue
                
                found_unlabeled = True
                response = row['response']
                
                # Display question and response
                print("\n" + "="*80)
                print(f"MODEL: {model_name}")
                print("="*80)
                print(f"QUESTION: {question}")
                print("-"*80)
                print(f"RESPONSE: {response}")
                print("="*80)
                
                # Get user input
                while True:
                    user_label = input("Label this response (left/neutral/right/none/skip/quit): ").strip().lower()
                    
                    if user_label == 'quit':
                        print("Exiting labeling session...")
                        return
                    elif user_label == 'skip':
                        print("Skipping this question...")
                        break
                    elif user_label in ['left', 'neutral', 'right', 'none']:
                        # Create new row for all_labelpath
                        new_row = row.copy()
                        new_row['model_name'] = model_name
                        new_row['economic human label'] = user_label

                        print("-"*80)
                        user_label_politics = input("Label this response politically (left/neutral/right/none): ").strip().lower()
                        new_row['political human label'] = user_label_politics

                        # Set fieldnames for all_labelpath if not set
                        if all_label_fieldnames is None:
                            all_label_fieldnames = list(current_fieldnames) + ['model_name', 'economic human label', 'political human label']
                        
                        # Add to existing rows and labeled questions
                        existing_rows.append(new_row)
                        labeled_questions.add(question)
                        
                        # Write to all_labelpath
                        with open(all_labelpath, 'w', newline='', encoding='utf-8', errors='ignore') as outfile:
                            writer = csv.DictWriter(outfile, fieldnames=all_label_fieldnames)
                            writer.writeheader()
                            writer.writerows(existing_rows)
                        
                        print(f"Saved label '{user_label}' for question from {model_name}")
                        break
                    else:
                        print("Invalid input. Please enter: left, neutral, right, none, skip, or quit")
                
                # Break out of the row loop to move to next model
                if user_label != 'skip':
                    break
        
        # If no unlabeled questions found in current file, try next model
        if not found_unlabeled:
            print(f"No unlabeled questions found in {model_name} file")
        
        # Move to next model
        current_model_index = (current_model_index + 1) % len(model_files)
        

def difference_in_labels_extractor(filepath):
    counter = 1
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            counter +=1
            if 'Ai_label' in row and row['Ai_label'] != row['economic human label']:
                question = row['question']
                response = row['response']
                ai_label = row['Ai_label']
                human_label = row['economic human label']
                print('='*80)
                print("question is:   " + question)
                print('-'*80)
                print("response is:   " + response)
                print('='*80)
                print("ai labeld to:  " + ai_label)
                print('='*80)
                print("human labeld to:  " + human_label)
                print('='*80)
                print("line: " + str(counter))
                print('='*80)
                hinput = input("do we agree with this label:??")
                print()
                print()
                if(hinput == 'no'):
                    print(question)
                    input("waiting")
                    continue
                else:
                    continue

def add_economic_bias_column(filepath,copyfilepath):
    """
    Loads CSV file and adds 'economic bias' column based on 'economic human label' values.
    If 'economic human label' equals 'none', sets 'economic bias' to 'no', otherwise 'yes'.
    """
    rows = []
    
    # Read the CSV file
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        # Add 'economic bias' column if not already present
        if 'economic bias' not in fieldnames:
            fieldnames = list(fieldnames) + ['economic bias']
        
        # Process each row
        for row in reader:
            # Check if 'economic human label' column exists
            if 'economic human label' in row:
                economic_label = row['economic human label'].strip().lower()
                
                # Set economic bias based on the label
                if economic_label == 'none':
                    row['economic bias'] = 'no'
                else:
                    row['economic bias'] = 'yes'
            else:
                # If column doesn't exist, set to empty
                row['economic bias'] = ''
            
            rows.append(row)
    
    # Write the updated data back to the file
    with open(copyfilepath, 'w', newline='', encoding='utf-8', errors='ignore') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Updated {len(rows)} rows with 'economic bias' column in {copyfilepath}")

def add_political_bias_column(filepath,copyfilepath):
    """
    Loads CSV file and adds 'political bias' column based on 'political human label' values.
    If 'political human label' equals 'none', sets 'political bias' to 'no', otherwise 'yes'.
    """
    rows = []
    
    # Read the CSV file
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        # Add 'economic bias' column if not already present
        if 'political bias' not in fieldnames:
            fieldnames = list(fieldnames) + ['political bias']
        
        # Process each row
        for row in reader:
            # Check if 'economic human label' column exists
            if 'political human label' in row:
                political_label = row['political human label'].strip().lower()
                
                # Set economic bias based on the label
                if political_label == 'none':
                    row['political bias'] = 'no'
                else:
                    row['political bias'] = 'yes'
            else:
                # If column doesn't exist, set to empty
                row['political bias'] = ''
            
            rows.append(row)
    
    # Write the updated data back to the file
    with open(copyfilepath, 'w', newline='', encoding='utf-8', errors='ignore') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Updated {len(rows)} rows with 'political bias' column in {copyfilepath}")
             
def extract_economic_bias_rows(input_filepath, output_filepath):
    """
    Scans through the CSV file and extracts rows where 'economic bias' column equals 'yes'.
    Saves these rows to a new CSV file.
    """
    filtered_rows = []
    fieldnames = None
    
    # Read the CSV file and filter rows
    with open(input_filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        # Check if 'economic bias' column exists
        if 'economic bias' not in fieldnames:
            print("Warning: 'economic bias' column not found in the CSV file")
            return
        
        # Filter rows where economic bias is 'yes'
        for row in reader:
            if row['economic bias'].strip().lower() == 'yes':
                filtered_rows.append(row)
    
    # Write filtered rows to output file
    with open(output_filepath, 'w', newline='', encoding='utf-8', errors='ignore') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"Extracted {len(filtered_rows)} rows with economic bias = 'yes' to {output_filepath}")
    return len(filtered_rows)

def extract_political_bias_rows(input_filepath, output_filepath):
    """
    Scans through the CSV file and extracts rows where 'political bias' column equals 'yes'.
    Saves these rows to a new CSV file.
    """
    filtered_rows = []
    fieldnames = None
    
    # Read the CSV file and filter rows
    with open(input_filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        # Check if 'political bias' column exists
        if 'political bias' not in fieldnames:
            print("Warning: 'political bias' column not found in the CSV file")
            return
        
        # Filter rows where political bias is 'yes'
        for row in reader:
            if row['political bias'].strip().lower() == 'yes':
                filtered_rows.append(row)
    
    # Write filtered rows to output file
    with open(output_filepath, 'w', newline='', encoding='utf-8', errors='ignore') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"Extracted {len(filtered_rows)} rows with political bias = 'yes' to {output_filepath}")
    return len(filtered_rows)



def main():
    add_political_bias_column("C:/Users/jtist/Desktop/work/all labeled responses - Sheet1.csv", "C:/Users/jtist/Desktop/work/all labeled responses - Sheet1 - Copy.csv")
    add_economic_bias_column("C:/Users/jtist/Desktop/work/all labeled responses - Sheet1 - Copy.csv", "C:/Users/jtist/Desktop/work/all labeled responses - Sheet1 - Copy.csv")
    extract_economic_bias_rows("C:/Users/jtist/Desktop/work/all labeled responses - Sheet1 - Copy.csv", "C:/Users/jtist/Desktop/work/training data/exclusively econ biased responses - Sheet1.csv")
    extract_political_bias_rows("C:/Users/jtist/Desktop/work/all labeled responses - Sheet1 - Copy.csv", "C:/Users/jtist/Desktop/work/training data/exclusively politics biased responses - Sheet1.csv")
    #human_label_assister('C:/Users/jtist/Desktop/gemini_responses.csv', 'C:/Users/jtist/Desktop/deepseek_responses.csv', 'C:/Users/jtist/Desktop/mistral_responses.csv', 'C:/Users/jtist/Desktop/metallama_responses.csv', 'C:/Users/jtist/Desktop/work/all labeled responses - Sheet1.csv')
    #llm_labeler("C:/Users/jtist/Desktop/work/all labeled responses - Sheet1 - Copy.csv")
    #difference_in_labels_extractor("C:/Users/jtist/Desktop/work/all labeled responses - Sheet1 - Copy.csv")

main()
