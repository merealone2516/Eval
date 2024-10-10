from Preprocessing import is_whitespace_only_diff, preprocess_total
import json
import argparse
import re
from transformers import AutoTokenizer
from LLMs.GPT import getGPTResponse
from LLMs.Llama import getLlamaResponse
from LLMs.Mistral import getMistralResponse
import csv
import pandas as pd
from tqdm import tqdm

#Preprocess Diffs and Messages
def preprocess_diffs_messages(HFKEY, data):
    msgs = list(data['message'])
    diffs = list(data['diff'])
    filtered_diff_msg = []
    pattern = re.compile(r'\d+ \. \d+( \. \d+)*')
    filtered_msgs, filtered_diffs = [],[]
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", token = HFKEY)
    for i in range(0,len(msgs)):
        n_tokens = len(tokenizer.encode(diffs[i]))
        if (not pattern.search(msgs[i])) and (not is_whitespace_only_diff(diffs[i])) and n_tokens<131072:
            filtered_msgs.append(msgs[i])
            filtered_diffs.append(diffs[i])
            filtered_diff_msg.append({'diff':diffs[i], 'message':msgs[i]})
    return filtered_diff_msg

#Preprocess Diffs and Messages
def preprocess_diffs_LLM_Msg(HFKEY, data):
    pattern = re.compile(r'\d+ \. \d+( \. \d+)*')
    filtered_diff_msg = []
    diffs = list(data['diff'])
    A = list(data['A'])
    B = list(data['B'])
    C = list(data['C'])
    D = list(data['D'])
    E = list(data['E'])
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Large-Instruct-2407", token = HFKEY)
    for i in range(0,len(A)):
        n_tokens = len(tokenizer.encode(getPromptEvaluate(diffs[i],A[i],B[i],C[i],D[i],E[i])))
        if (not pattern.search(A[i])) and \
            (not pattern.search(B[i])) and \
                (not pattern.search(C[i])) and \
                    (not pattern.search(D[i])) and \
                        (not pattern.search(E[i])) and \
                            (not is_whitespace_only_diff(diffs[i])) and n_tokens<131072:
            filtered_diff_msg.append({'diff':diffs[i], 'A':A[i], 'B':B[i], 'C':C[i], 'D':D[i], 'E':E[i]})
    return filtered_diff_msg

#Get Prompts for Classification
def getPromptClassify(diff, msg):
    instructions = """
    Definitions:
    For each given source code diff and its corresponding commit message, read and analyze the content to determine if the message accurately describes the changes made in the diff. If the commit message is a relevant description of the diff, label it as 1. Otherwise, label it as 0.

    Label Descriptions:

    Label 1: The commit message belongs to the diff (i.e., it correctly explains the changes made).
    Label 0: The commit message does not belong to the diff (i.e., it is unrelated or irrelevant to the changes made).
    """
    prompt = f"{instructions}\n\ndiff: {diff}\n\nCommit Message: \"{msg}\"\n\nCategory:"
    return prompt

#Get Prompts for choosing the best commit message
def getPromptEvaluate(diff, a, b, c, d, e):
    instructions = """
    In this dataset, each source code diff is paired with different commit message options: A, B, C, D, and E. Using the provided definition, select the best commit message for each diff.

    Definition: A good commit message clearly describes what change was made, using concise language to help others understand the purpose without looking at the code.
    """
    prompt = f"{instructions}\n\ndiff: {diff}\n\nCommit Message: \nA:\"{a}\"\nB:\"{b}\"\nC:\"{c}\"\nD:\"{d}\"\nE:\"{e}\"\n\nBest Commit Message is given by :"
    return prompt

if __name__ == '__main__':
    #Get all API Keys
    parser = argparse.ArgumentParser(description='Process necessary API keys.')
    parser.add_argument('--HFKEY', type=str, required=True, help='Required Hugging Face API Key')
    parser.add_argument('--GROQ', type=str, required=True, help='Required Groq API Key')
    parser.add_argument('--OPENAI', type=str, required=True, help='Required OpenAI API Key')
    parser.add_argument('--MISTRAL', type=str, required=True, help='Required Mistral API Key')
    
    #Get File name
    parser.add_argument('--FILE', type=str, required=True, help='Required Input File name')
    parser.add_argument('--OFILE', type=str, required=True, help='Required Output File name')
    
    #Get Task 
    parser.add_argument('--TASK', type=str, required=True, help='Required to execute which task')
      
    #Store API keys
    args = parser.parse_args()
    HFKEY = args.HFKEY
    GROQ = args.GROQ
    OPENAI_API_KEY = args.OPENAI
    MISTRAL = args.MISTRAL
    
    #Store File name
    filename = args.FILE
    output_file = args.OFILE
    
    #Store Task
    task = args.TASK
    
    #Preprocess File of diffs and messages
    data = pd.read_csv(filename)
    filtered_diff_msg=[]
    if(task == 'A'):
        filtered_diff_msg=preprocess_diffs_messages(HFKEY,data)
    else:
        filtered_diff_msg=preprocess_diffs_LLM_Msg(HFKEY,data)
    
    if(task=='A'):
        print("Generating Classification Output using Multiple LLMs")
    else:
        print("Generating Evaluation Output using Multiple LLMs")
    #Initialize containers to store failures
    cf_1, cf_2, cf_3, cf_4 = [], [], [], []
    
    print('Starting Llama3.1 70B')
    #Llama3.1 70B
    cf_1=[]
    for i in tqdm(range(0, len(filtered_diff_msg))):
        if ('llama-70b-output' not in filtered_diff_msg[i] or filtered_diff_msg[i]['llama-70b-output'] == 'ERROR'):
            prompt = ""
            if(task == 'A'):
                prompt = getPromptClassify(filtered_diff_msg[i]['diff'], filtered_diff_msg[i]['message'])
            else:
                prompt = getPromptEvaluate(filtered_diff_msg[i]['diff'], filtered_diff_msg[i]['A'], filtered_diff_msg[i]['B'], filtered_diff_msg[i]['C'], filtered_diff_msg[i]['D'], filtered_diff_msg[i]['E'])
            try:
                filtered_diff_msg[i]['llama-70b-output'] = getLlamaResponse(GROQ,"llama-3.1-70b-versatile",prompt)
            except:
                cf_1.append(i)
                filtered_diff_msg[i]['llama-70b-output'] = 'ERROR'
                pass
    
    print("--Llama3.1 70B model output generated")
    
    #Llama3.1 8B    
    print('Starting Llama3.1 8B')   
    cf_2=[]
    for i in tqdm(range(0, len(filtered_diff_msg))):
        if ('llama3.1-8b-output' not in filtered_diff_msg[i] or filtered_diff_msg[i]['llama3.1-8b-output'] == 'ERROR'):
            prompt = ""
            if(task == 'A'):
                prompt = getPromptClassify(filtered_diff_msg[i]['diff'], filtered_diff_msg[i]['message'])
            else:
                prompt = getPromptEvaluate(filtered_diff_msg[i]['diff'], filtered_diff_msg[i]['A'], filtered_diff_msg[i]['B'], filtered_diff_msg[i]['C'], filtered_diff_msg[i]['D'], filtered_diff_msg[i]['E'])
            try:
                filtered_diff_msg[i]['llama3.1-8b-output'] = getLlamaResponse(GROQ,"llama-3.1-8b-instant",prompt)
            except:
                cf_2.append(i)
                filtered_diff_msg[i]['llama3.1-8b-output'] = 'ERROR'
                pass
    
    print("--Llama3.1 8B model output generated")
    
    #Mistral-Large  
    print('Starting Mistral Large')
    cf_3=[]
    for i in tqdm(range(0, len(filtered_diff_msg))):
        if ('mistral-large-output' not in filtered_diff_msg[i] or filtered_diff_msg[i]['mistral-large-output'] == 'ERROR'):
            prompt = ""
            if(task == 'A'):
                prompt = getPromptClassify(filtered_diff_msg[i]['diff'], filtered_diff_msg[i]['message'])
            else:
                prompt = getPromptEvaluate(filtered_diff_msg[i]['diff'], filtered_diff_msg[i]['A'], filtered_diff_msg[i]['B'], filtered_diff_msg[i]['C'], filtered_diff_msg[i]['D'], filtered_diff_msg[i]['E'])
            try:
                filtered_diff_msg[i]['mistral-large-output'] = getMistralResponse(MISTRAL,prompt)
            except:
                cf_3.append(i)
                filtered_diff_msg[i]['mistral-large-output'] = 'ERROR'
                pass
    
    print("--Mistral Large model output generated")
    
    #GPT-4o
    print('Starting GPT-4o')
    cf_4 = []
    for i in tqdm(range(0, len(filtered_diff_msg))):
        if('gpt-4o-output' not in filtered_diff_msg[i] or filtered_diff_msg[i]['gpt-4o-output'] == 'ERROR'):
            prompt = ""
            if(task == 'A'):
                prompt = getPromptClassify(filtered_diff_msg[i]['diff'], filtered_diff_msg[i]['message'])
            else:
                prompt = getPromptEvaluate(filtered_diff_msg[i]['diff'], filtered_diff_msg[i]['A'], filtered_diff_msg[i]['B'], filtered_diff_msg[i]['C'], filtered_diff_msg[i]['D'], filtered_diff_msg[i]['E'])
            try:
                filtered_diff_msg[i]['gpt-4o-output'] = getGPTResponse(OPENAI_API_KEY,prompt)
            except:
                cf_4.append(i)
                filtered_diff_msg[i]['gpt-4o-output'] = 'ERROR'
                pass
    
    print("--GPT-4o model output generated")
    
    #Save Outputs for Total Diffs
    print("Saving Outputs.....")
    folder='Temp'
    if(task == 'A'):
        folder = 'Classify'
    else:
        folder = 'Evaluate'
    csv_file_name = f'Outputs/{folder}/{output_file}.csv'
    headers = filtered_diff_msg[0].keys()
    with open(csv_file_name, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(filtered_diff_msg)
        
    print(f'Output of all models saved to Outputs/{folder}/{output_file}.csv')