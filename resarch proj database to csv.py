import csv
import sqlite3



def database_to_csv(model, csv_filename):
    # I'm splitting each model into a different csv file
    # Connect to the SQLite database
    conn = sqlite3.connect('responses.db')
    cursor = conn.cursor()
    
    # Get all rows for the specified model
    cursor.execute("SELECT * FROM Responses WHERE model_name = ?", (model,))
    rows = cursor.fetchall()
    
    # Get column names from the database
    cursor.execute("PRAGMA table_info(Responses)")
    columns_info = cursor.fetchall()
    column_names = [column[1] for column in columns_info]
    
    # Write to CSV file
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row (column names)
        writer.writerow(column_names)
        writer.writerows(rows)
    
    conn.close()
    
    print(f"Successfully exported {len(rows)} rows for model '{model}' to {csv_filename}")


# Example usage:
if __name__ == "__main__":
    # Export data for a specific model
    database_to_csv("gemini-2.5-flash", "gemini_responses.csv")
    database_to_csv("meta-llama/llama-4-maverick-17b-128e-instruct:free", "metallama_responses.csv")
    database_to_csv("deepseek/deepseek-r1-0528-qwen3-8b:free", "deepseek_responses.csv")
    database_to_csv("mistralai/mistral-small-3.2-24b-instruct:free", "mistral_responses.csv")
