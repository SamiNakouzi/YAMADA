!/bin/bash
    for VARIABLE in {1..75}
    do
        python3 snn_test.py
        python3 snn_model.py
        python3 pulses.py
    done

#    for VARLIABLE in {1..1000}
#    do
#        python3 pulses.py
#    done
