# Automatic Commit Message Generation

## Overview
Evaluating the quality of commit messages is a challenging task in software engineering. Existing evaluation approaches, such as automatic metrics like BLEU, ROUGE and METEOR or manual human assessments, have notable limitations. Automatic metrics often overlook semantic relevance and context, while human evaluations are time consuming and costly. To address these challenges, we explore the potential of using Large Language Models (LLMs) as an alternative method for commit message evaluation. We conducted two tasks using state-of-the-art LLMs, GPT-4o, LLaMA 3.1 (70B and 8B), and Mistral Large, to assess their capability in evaluating commit messages. Our findings show that LLMs can effectively identify relevant commit messages and align well with human judgment, demonstrating their potential to serve as reliable automated evaluators. This study provides a new perspective on utilizing LLMs for commit message assessment, paving the way for scalable and consistent evaluation methodologies in software engineering.

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
