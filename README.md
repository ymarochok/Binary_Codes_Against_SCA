# Protecting neural networks with binary codes from Side-Channel Attacks

This repositroy contains project which is scientific part of our bachlor`s work. 

## Current state of the project
While network development went great and applying quantization technique and translating network to the convertable into hex file format too, taking measurements and simulating the attack not so well. Network is not triggers on special command sended by script, and not executing. In the result we have captured trace which looks just like noise, and cant somehow serve to extract hidden parameters of the network.

**Plan to finish by the end of second semeter:**
1. Capture clear traces with highly visible trigger activation.
2. Carry the attack on quantized unprotected network.
3. Implement protection technique on the network.

4. Execute the attack on protected network and compare the results.

Additionally would be great to investigate another approaches of designing protection technique (encoding values and lookup-table implementation) comparing the final results to chose the most effective technique.

## Project structure
This project contains all related files to the project, some of which are only inter steps in achieving final goal. On January 2026 to the main three folder in this respository can be assigned linked to them stages of final solution development, which allows to look through all the proces in need of debugging, or researching more detailed specific steps.


1. **Network Implementation in Python** -> Initial network model development.
    - design a neural network with specific task and create model
    - quantize model to 4-bits
    - extract quantized model values

2. **Network Implementation in Python** -> Development of model in C linked to the python model.
    - design simple network in C
    - assign values extracted from python quantized model into this new model.

3. **Network compiled for STM32** -> Adepting C code for the microcontroller target.
    - add main.c file as main instruction which contains endless loop of things which microcontroller will do.
    - compile C code into hex file and upload it onto the ChipWhisperer board.


## Explanation of project structure:

### Network in Python:
- **main.py** -> The root of all solution. Here first model with only one goal, to detect anomaly in sequence of number, is designed. Then in this file exectures quantization inference which rebuild model into quantized version of model, with following extracting of params.

- **CSV files** -> contains dataset for training and evaluating of the model. Later test_sensor.csv will be copied into the *network in c* folder, with the same goal.

- **TXT files** -> asnwering the name of each file, each contains specific values of quantized model. All the parameters for the C model is taking from here.

### Network in C:
- **network.c** -> contains forward function for the network.
- **network_config.h** -> contains network parameters.
- **nn** -> compiled exectuable file

### Network for STM32:
- **main.c** -> contains main exectuable loop of commands which will be running on the chip: trigger activtion, network execution, command processing.
- **makefile** -> main file which will compile the project into simplseserial-files for assigned target. Contains all the specific parameters like files to include in compilation, location of libraries and their versions.

- **HAL** and **SIMPLESERIAL** -> libraries for successfull compilation of whole project.

### ChipWhisperer Communication
- **upload_program.py** -> File which is responsible for successfull upload of hex file onto the board.
- **trigger_test.py** -> executes code for triggering a start of neural network computation and capturing the power_consumption trace.

In addition this folder contains hex files used for debugging the project as files for just getting response back from the chip and one with artifically delayed network execution for clearer trigger recognition on captured traces.

## Network model description
Network model contains one hidden layer, one input and one output.
Goal of the network is to predict if in the sequence of numbers hides anomaly.

Input layer takes on input 10 numbers, hidden layer work within 5 neurons, and output layer return 1 if anomaly is detected and 0 if not.

Average accuracy of standard network -> 87%-92%
Average accuracy of quantized network -> 80%-85%


## How to compile C code into hex files
You need to execture commands from the network_for_stm32 folder.
1. ``` make clean ``` - to delete all leftovers from the previous compilation.
2. ```make PLATFORM=CWLITEARM``` - compiles files which can later be uploaded on the device.  
