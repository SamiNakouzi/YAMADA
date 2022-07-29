!/bin/bash
echo Enter the number of epochs:
read epoch #NUMBER OF EPOCHS

echo "initializing pump..."
python3 initializing.py #file that sets the pump to the value of %mu1 before each learning session
rm ../../data/data_epoch.txt
rm ../../data/data_ta.txt

echo " "
echo "Starting:"
for (( c=1; c<=$epoch; c++ ))
do
    python3 snn_test.py
    python3 snn_model.py
    python3 model_yamada_snn.py
done
echo " "
echo "Network has learned"
echo "Plotting evolution of the output and creating gif"

python3 learning_study.py
