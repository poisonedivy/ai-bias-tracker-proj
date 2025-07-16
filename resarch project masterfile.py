import csv
import sqlite3
from google import genai
from datetime import datetime
import requests
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()


questionLst=[]
currentQuestion = ""
extendedQuestionLst = []

responses = {}
currentResponse = ""
subjectLst=[]
cursubject = ""
skewLst=[]
curskew = ""
countryLst=[]
curcountry = ""

openRouterKey = os.getenv('OPENROUTER_API_KEY')
openRouterKey2= os.getenv('OPENROUTER_API_KEY2')
openRouterKey3= os.getenv('OPENROUTER_API_KEY3')
openRouterKey4= os.getenv('OPENROUTER_API_KEY4')
openRouterKey5= os.getenv('OPENROUTER_API_KEY5')
openRouterKey6= os.getenv('OPENROUTER_API_KEY6')
openRouterKey7= os.getenv('OPENROUTER_API_KEY7')
openRouterKey8= os.getenv('OPENROUTER_API_KEY8')
gemini_api_key = os.getenv('GEMINI_API_KEY')
gemini_api_key2 = os.getenv('GEMINI_API_KEY2')

openRouterKeys = [openRouterKey, openRouterKey2, openRouterKey3, openRouterKey4, openRouterKey5, openRouterKey6, openRouterKey7, openRouterKey8]

remaning_keys = openRouterKeys.copy()

#values llms we are using
#meta-llama/llama-4-maverick-17b-128e-instruct:free
#deepseek/deepseek-r1-0528-qwen3-8b:free
#mistralai/mistral-small-3.2-24b-instruct:free
#gemini-2.5-flash



def csvloader(filename):
    #works ish but first 2 entries are garbage
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        #first 2 rows are garbage so skip them
        next(reader)  # Skip first row
        next(reader)  # Skip second row
        for row in reader:
            questionLst.append(row[0])
            subjectLst.append(row[1])
            skewLst.append(row[2])
            countryLst.append(row[3])

def promptExtender():
    #not done yet needs more to be added
    for question in questionLst:
        #republican view
        extendedQuestionLst.append(question + " from a republican point of view")

        #democrate view
        extendedQuestionLst.append(question + " from a democrat point of view")

        #independent view
        extendedQuestionLst.append(question + " from a independent point of view")

        #objective view
        extendedQuestionLst.append(question + " from a objective point of view")

        #as an expert in the subject
        extendedQuestionLst.append("as an expert in the subject " + question)


def databasebuilder():
    #works
    global prompts, promptscurs
    prompts = sqlite3.connect("prompts.db")
    promptscurs = prompts.cursor()
    promptscurs.execute("CREATE TABLE IF NOT EXISTS Prompts(question TEXT NOT NULL,subject TEXT,political_skew TEXT,country_bias TEXT)")
    for i in range(len(questionLst)):
        curquestion = questionLst[i]
        cursubject = subjectLst[i]
        curskew = skewLst[i]
        curcountry = countryLst[i]
        # Escape single quotes in the question string for the SQL query
        escaped_question = curquestion.replace("'", "''")

        if promptscurs.execute(f"SELECT COUNT(*) FROM Prompts WHERE question = '{escaped_question}';").fetchone()[0] != 1:
            promptscurs.execute(f"INSERT INTO Prompts VALUES ('{escaped_question}', '{cursubject}', '{curskew}', '{curcountry}')")
        prompts.commit()

def dataBasePrinter(database):
    #works
    if database == "prompts":    
        for row in promptscurs.execute("SELECT * FROM Prompts"):
            print(row)
    elif database == "responses":
        for row in responsescurs.execute("SELECT * FROM Responses"):
            print(row)
    else:
        print("failed, check spelling or smthing")

def responsesbuilder():
    #works
    global responses, responsescurs
    responses = sqlite3.connect("responses.db")
    responsescurs = responses.cursor()
    responsescurs.execute("CREATE TABLE IF NOT EXISTS Responses(question TEXT, response TEXT, model_name TEXT, model_version TEXT, time TEXT)")

def prompt_gemini(inputPrompt):
    #works
    try:
        client = genai.Client(api_key= gemini_api_key)
        model="gemini-2.5-flash"
        currentQuestion = inputPrompt
        modelVersion = "2.5-flash"

        if responsescurs.execute("SELECT COUNT(*) FROM responses WHERE question = ? AND model_name = ?", (inputPrompt, model)).fetchone()[0] == 0:
            response = client.models.generate_content(
                model=model,
                contents= inputPrompt,
            )
            now = datetime.now()
            current_time = now.strftime('%m/%d/%Y, %H:%M:%S')
            responsescurs.execute("INSERT INTO Responses VALUES (?, ?, ?, ?, ?)", (currentQuestion, response.text, model,modelVersion, current_time))
            responses.commit()
    except:
        client = genai.Client(api_key= gemini_api_key2)
        model="gemini-2.5-flash"
        currentQuestion = inputPrompt
        modelVersion = "2.5-flash"

        if responsescurs.execute("SELECT COUNT(*) FROM responses WHERE question = ? AND model_name = ?", (inputPrompt, model)).fetchone()[0] == 0:
            response = client.models.generate_content(
                model=model,
                contents= inputPrompt,
            )
            now = datetime.now()
            current_time = now.strftime('%m/%d/%Y, %H:%M:%S')
            responsescurs.execute("INSERT INTO Responses VALUES (?, ?, ?, ?, ?)", (currentQuestion, response.text, model,modelVersion, current_time))
            responses.commit()




def prompt_openRouter(inputPrompt, modelname):
    if responsescurs.execute("SELECT COUNT(*) FROM responses WHERE question = ? AND model_name = ?", (inputPrompt, modelname)).fetchone()[0] == 0:
        
        for api_key in remaning_keys:
            if api_key is None:  
                continue
                
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": api_key,
                        "Content-Type": "application/json",
                    },
                    data=json.dumps({
                        "model": modelname,
                        "messages": [
                            {
                                "role": "user",
                                "content": inputPrompt
                            }
                        ],
                    })
                )
                
                # Check if the response is successful
                if response.status_code == 200:
                    response_data = response.json()
                    now = datetime.now()
                    current_time = now.strftime('%m/%d/%Y, %H:%M:%S')
                    responsescurs.execute("INSERT INTO Responses VALUES (?, ?, ?, ?, ?)", 
                                        (inputPrompt, response_data.get("choices")[0].get("message").get("content"), 
                                         modelname, "openrouter", current_time))
                    responses.commit()
                    time.sleep(1)
                    return 
                
                elif response.status_code == 429:
                    # Rate limit or quota exceeded - remove this key
                    print(f"API key exhausted (429 error). Removing from available keys...")
                    if api_key in remaning_keys:
                        remaning_keys.remove(api_key)
                    time.sleep(1)
                    continue
                
                elif response.status_code == 400:
                    print(f"API key returned 400 error. Trying next key...")
                    time.sleep(1)
                    continue  
                
                else:
                    print(f"API key returned status code {response.status_code}. Trying next key...")
                    time.sleep(1)
                    continue  
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed with API key: {e}. Trying next key...")
                time.sleep(1)
                continue
            except Exception as e:
                print(f"Unexpected error with API key: {e}. Trying next key...")
                time.sleep(1)
                continue
        
        # If we get here, all keys have been exhausted
        print(f"All API keys exhausted for prompt: {inputPrompt[:50]}...")
        raise Exception("All OpenRouter API keys have been exhausted or returned errors")




def main():
    csvloader('C:/Users/jtist/Downloads/prompts for ai tracker - Sheet1.csv')
    promptExtender()
    databasebuilder()
    #dataBasePrinter("prompts")
    responsesbuilder()
    for i in range(2, len(extendedQuestionLst)):
        currentQuestion = extendedQuestionLst[i]
        prompt_openRouter(currentQuestion, "deepseek/deepseek-r1-0528-qwen3-8b:free")
        print(str(len(extendedQuestionLst)-i) + " remaining")
        #prompt_gemini(currentQuestion)
    #dataBasePrinter("responses")
    responselength = 0
    for row in responsescurs.execute("SELECT * FROM Responses WHERE model_name = 'deepseek/deepseek-r1-0528-qwen3-8b:free'").fetchall():
        responselength += 1
    print(responselength)
    




if __name__ == "__main__":
    main()
