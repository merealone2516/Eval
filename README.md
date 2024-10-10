# Automatic Commit Message Generation

## Overview
This repository contains the code and resources for evaluating generated automated commit messages using state-of-the-art Large Language Models (LLMs), such as GPT-4o, Llama 3.1 (70B and 8B), and Mistral Large. The project addresses a significant challenges in automated commit message generation:

**Can we use LLM as Evaluator:** Existing evaluation approaches, such as automatic metrics like BLEU, ROUGE and METEOR or manual human assessments, have notable limitations.

## Prerequisites
Before you begin, ensure you have met the following requirements:
- [ ]  You must be running the code in an Unix based OS.
- [ ]  You have downloaded and setup Python version >= 3.10.
- [ ]  You took permission to use Mistral tokenizer in the Hugging Face platform.
- [ ]  You have generated API token from Hugging Face, Groq, OpenAI, Mistral.
- [ ]  You might need to add credits for each platform if needed.

## Run the Code

To Run this project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/merealone2516/LLMs-as-evaluator-in-commit-messages.git

# Navigate to the project directory
cd LLMs-as-evaluator-in-commit-messages

#Set up Virtual Environment
python3 -m venv env

#Activate the Environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

#Run the models - Replace Tokens with the Actual Tokens
./run_all.sh Hugging_Face_Token Groq_Token OpenAI_Token Mistral_Token

```
