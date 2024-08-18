#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to perform matrix-vector multiplication
void matrix_vector_multiply(float *matrix, float *vector, float *result, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        result[i] = 0;
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

//function to load files
void load_weights(const char *filename, float *weights, int size) {
    FILE *file = fopen(filename, "r"); //reading the file
    if (!file) { //if file is null
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        fscanf(file, "%f", &weights[i]); //loads the files components to weights array
    }
    fclose(file);
}


// Forward pass for a GRU unit
void gru_forward_pass(
    int time_frames,
    int input_size,
    int hidden_size,
    float *inputs,
    float *h_prev,
    float *W_z, float *U_z, float *b_z,  //Weights and biases for update gate
    float *W_r, float *U_r, float *b_r,  //Weights and biases for reset gate    
    float *W_h, float *U_h, float *b_h,  //Weights and biases for candidate state or current memory content
    float *output
) {
    //traverse through all time frames 
        for (int t = 0; t < time_frames ; t++) {
            float z[hidden_size];       //update gate
            float r[hidden_size];       //reset gate
            float h_hat[hidden_size];   //candidate state
            float h[hidden_size];       //hidden state

            // Extract the input at time step t for sequence seq by incrementing the inputs pointer
            float *x_t = inputs + t * input_size;

            // Compute z_t = sigmoid(W_z * x_t + U_z * h_prev + b_z)
            matrix_vector_multiply(W_z, x_t, z, hidden_size, input_size);                           //W_z * x_t
            matrix_vector_multiply(U_z, h_prev, z, hidden_size, hidden_size);  //U_z * h_prev
            for (int i = 0; i < hidden_size; ++i) {
                z[i] = 1.0 / (1.0 + exp(-(z[i] + b_z[i]))); //applying sigmoid activation function
            }

            // Compute r_t = sigmoid(W_r * x_t + U_r * h_prev + b_r)
            matrix_vector_multiply(W_r, x_t, r, hidden_size, input_size);                           //W_r * x_t
            matrix_vector_multiply(U_r, h_prev, r, hidden_size, hidden_size);   //U_r * h_prev
            for (int i = 0; i < hidden_size; ++i) {
                r[i] =  1.0 / (1.0 + exp(-(r[i] + b_r[i]))); //applying sigmoid activation function
            }


            // Compute h_hat_t = tanh(W_h * x_t + U_h * (r_t * h_prev) + b_h)
            matrix_vector_multiply(W_h, x_t, h_hat, hidden_size, input_size);   //W_h * x_t

            //U_h * (r_t * h_prev)
            float r_h_prev[hidden_size];
            for (int i = 0; i < hidden_size; ++i) {
                r_h_prev[i] = r[i] * h_prev[i];
            }
            float Wh_r_h_prev[hidden_size];
            matrix_vector_multiply(U_h, r_h_prev, Wh_r_h_prev, hidden_size, hidden_size);

            for (int i = 0; i < hidden_size; ++i) {
                h_hat[i] +=Wh_r_h_prev[i] + b_h[i];  
                h_hat[i] = tanh(h_hat[i]);  // Apply the tanh activation function
            }


            // Compute h_t = (1 - z_t) * h_prev + z_t * h_hat_t
            for (int i = 0; i < hidden_size; ++i) {
                h[i] = ( z[i]) * h_prev[i] + (1.0 -z[i]) * h_hat[i];
            }


            // Store the output hidden state for this time step
            for (int i = 0; i < hidden_size; ++i) {
                output[ t * hidden_size + i] = h[i];
            }



            // Update h_prev for the next time step
            for (int i = 0; i < hidden_size; ++i) {
                h_prev[ i] = h[i];
            }
        }
    }

int main() {
    int num_sequences, time_frames, input_size, hidden_size;

    // Taking user input for the dimensions    
    printf("Enter the number of time frames per sequence: ");
    scanf("%d", &time_frames);
    printf("Enter the input size (number of features): ");
    scanf("%d", &input_size);
    printf("Enter the hidden size (number of units in GRU): ");
    scanf("%d", &hidden_size);

    float inputs[ time_frames * input_size]; // inputs
    float h_prev[ hidden_size]; // hidden state

    // weights and biases
    float W_z[hidden_size * input_size];
    float U_z[hidden_size * hidden_size];
    float b_z[hidden_size];

    float W_r[hidden_size * input_size];
    float U_r[hidden_size * hidden_size];
    float b_r[hidden_size];

    float W_h[hidden_size * input_size];
    float U_h[hidden_size * hidden_size];
    float b_h[hidden_size];

    float output[ time_frames * hidden_size];

    /*Use the following code if we want to initialize to random values rather than user input/pre-trained weights
     for (int i = 0; i < num_sequences * time_frames * input_size; ++i) {
         inputs[i] = (float)rand() / RAND_MAX; 
     }   
     for (int i = 0; i < hidden_size * input_size; ++i) {
         U_z[i] = U_r[i] = U_h[i] = (float)rand() / RAND_MAX;
     }
     for (int i = 0; i < hidden_size * hidden_size; ++i) {
         W_z[i] = W_r[i] = W_h[i] = (float)rand() / RAND_MAX;
     }
     for (int i = 0; i < hidden_size; ++i) {
         b_z[i] = b_r[i] = b_h[i] = 0;
     }
    */
    
    //loading data from files to our matrices
    //load input from user into inputs.txt file 
    load_weights("D:\\new job\\IPHIPI\\txt\\inputs.txt",inputs,  time_frames * input_size);

    //initializing the hidden units for 1st time as all zeros
     for (int i = 0; i <  hidden_size; ++i) {
        h_prev[i] = 0;
    }

    //loading pre_trained weights from respective txt files
    load_weights("D:\\new job\\IPHIPI\\txt\\U_z.txt", W_z, hidden_size * input_size);
    load_weights("D:\\new job\\IPHIPI\\txt\\U_r.txt", W_r, hidden_size * input_size);
    load_weights("D:\\new job\\IPHIPI\\txt\\U_h.txt", W_h, hidden_size * input_size);
    
    load_weights("D:\\new job\\IPHIPI\\txt\\W_z.txt", U_z, hidden_size * hidden_size);
    load_weights("D:\\new job\\IPHIPI\\txt\\W_r.txt", U_r, hidden_size * hidden_size); 
    load_weights("D:\\new job\\IPHIPI\\txt\\W_h.txt", U_h, hidden_size * hidden_size);

    load_weights("D:\\new job\\IPHIPI\\txt\\b_z.txt", b_z, hidden_size);
    load_weights("D:\\new job\\IPHIPI\\txt\\b_r.txt", b_r, hidden_size);
    load_weights("D:\\new job\\IPHIPI\\txt\\b_h.txt", b_h, hidden_size);

    //forward pass function for GRU
    gru_forward_pass( time_frames, input_size, hidden_size,inputs, h_prev,
                     W_z, U_z, b_z,
                     W_r, U_r, b_r,
                     W_h, U_h, b_h,
                     output);

    // printing results
    printf("Outputs for each time frame of each sequence are \n");
    for (int i = 0; i <  time_frames * hidden_size; ++i) {
        printf("%f ", output[i]);
        if ((i + 1) % hidden_size == 0) printf("\n");
        if ((i + 1) % (hidden_size*time_frames) == 0) printf("\n");
    }

    return 0;
}
