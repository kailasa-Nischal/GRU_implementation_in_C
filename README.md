# GRU_implementation_in_C
The GRU forward pass in C involves calculating the hidden state at each time step using the input, previous hidden state, and gates (update and reset). Weights are pre-initialized, and the hidden state is updated iteratively.

To run the GRU forward pass in C, follow these steps:

1)Generate Pre-Trained Weights: Execute the GRU_implementation.ipynb to create pre-trained model files. During this process, provide the following inputs:
Number of time frames per sentence,
Size of input 𝑥𝑡,
​Size of hidden layer ℎ𝑡,
​one sequences

2)Download: Obtain the folder containing the text files with the pre-trained weights.
(Code to zip all txt files and to download them are written in GRU_implementation.ipynb file itself in a separate cell)

3)Load and Execute: Load these files into your C program(by changing to the corresponding path in your pc) and execute the forward pass function.
The output will include the hidden states for each word in the sequences for each time stamp.
