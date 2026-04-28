#!/bin/bash

LOG_FILE="training_log.txt"

for i in {1..10}
do
   echo "Starting Run $i..."
   echo "=== RUN $i START ===" >> $LOG_FILE
   
   # Replace with your actual command
   python mimic_mamba3.py >> $LOG_FILE 2>&1
   
   echo "=== RUN $i END ===" >> $LOG_FILE
   echo "Run $i finished."
done