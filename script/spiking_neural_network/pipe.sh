!/bin/bash
echo Enter the number of epochs:
read epoch #NUMBER OF EPOCHS

echo "initializing pump..."
python3 initializing.py

echo "Starting:"
for (( c=1; c<=$epoch; c++ ))
do
    python3 snn_test.py
    python3 snn_model.py
    python3 model_yamada_snn.py
done
