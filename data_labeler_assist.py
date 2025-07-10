import csv
import tempfile
import shutil

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
    labeler("C:/Users/jtist/OneDrive/Documents/gemini-responses-labled.csv")

main()
