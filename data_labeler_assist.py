import csv
import tempfile
import shutil
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key2 = os.getenv('GEMINI_API_KEY2')

# prompt gemini to label the responses and flag possible biased answers
def llm_labeler2(filepath):
    rows = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        # Add 'label' if not already
        if 'label' not in fieldnames:
            fieldnames = list(fieldnames) + ['label']
        
        for row in reader:
            rows.append(row)
    
    #write to csv
    for i, row in enumerate(rows):
        # Check if already labeled
        if 'label' in row and row['label'] and row['label'].strip():
            print(f"Skipping already labeled row: {row['label']}")
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
                row['label'] = label
                print(f"Labeled as: {label}")
            else:
                row['label'] = 'unknown'
                print(f"Invalid response from API: {label}, setting to 'unknown'")
                
        except Exception as e:
            print(f"Error calling API: {e}")
            row['label'] = 'error'
        
        # Write the entire file after each label is added
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Progress: {i+1}/{len(rows)} rows processed")





            


def labeler(filepath):
    # First, read all rows and add labels
    rows = []
    
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        # Add 'label' field if it doesn't exist
        if 'label' not in fieldnames:
            fieldnames = list(fieldnames) + ['label']
        
        for row in reader:
            question = row['question']
            response = row['response']
            print(f"\nQuestion: {question}")
            print(f"Response: {response}")
            
            direction = input("Enter direction (left, center, right): ")
            
            if direction in ["left", "center", "right"]:
                row['label'] = direction
                print(f"Labeled as {direction}")
            else:
                print("Invalid direction, skipping...")
                row['label'] = ''
            
            rows.append(row)
    
    # Write back to the original file (or a new file)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    llm_labeler2("C:/Users/jtist/Desktop/work/gemini-responses-labled.csv")

main()
