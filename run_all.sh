#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 HFKEY GROQ OPENAI MISTRAL"
    exit 1
fi

# Assign arguments to variables
HFKEY=$1
GROQ=$2
OPENAI=$3
MISTRAL=$4

# Run main.py with the passed arguments
python3 main.py --HFKEY "$HFKEY" --GROQ "$GROQ" --OPENAI "$OPENAI" --MISTRAL "$MISTRAL" --FILE "data/shuffled_combined_diff_files_no_label.csv" --OFILE "shuffle_no_label_output" --TASK "A"
python3 main.py --HFKEY "$HFKEY" --GROQ "$GROQ" --OPENAI "$OPENAI" --MISTRAL "$MISTRAL" --FILE "data/shuffled_combined_diff_files_with_label.csv" --OFILE "shuffle_with_label_output" --TASK "A"
python3 main.py --HFKEY "$HFKEY" --GROQ "$GROQ" --OPENAI "$OPENAI" --MISTRAL "$MISTRAL" --FILE "data/To_evaluate.csv" --OFILE "to_evaluate_output" --TASK "B"