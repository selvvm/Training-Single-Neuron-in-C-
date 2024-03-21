#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

using namespace std;

// Activation function (sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

// Neural Network Class
class NeuralNetwork {
private:
    int inputNodes;
    int hiddenNodes;
    int outputNodes;
    double learningRate;
    
    // Weights
    vector<vector<double>> weightsInputHidden;
    vector<vector<double>> weightsHiddenOutput;
    
    // Bias
    vector<double> biasHidden;
    vector<double> biasOutput;
    
public:
    NeuralNetwork(int input, int hidden, int output, double lr) {
        inputNodes = input;
        hiddenNodes = hidden;
        outputNodes = output;
        learningRate = lr;
        
        // Initialize weights randomly between -1 and 1
        weightsInputHidden.resize(inputNodes, vector<double>(hiddenNodes));
        weightsHiddenOutput.resize(hiddenNodes, vector<double>(outputNodes));
        for (int i = 0; i < inputNodes; ++i) {
            for (int j = 0; j < hiddenNodes; ++j) {
                weightsInputHidden[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }
        for (int i = 0; i < hiddenNodes; ++i) {
            for (int j = 0; j < outputNodes; ++j) {
                weightsHiddenOutput[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }
        
        // Initialize bias randomly between -1 and 1
        biasHidden.resize(hiddenNodes);
        biasOutput.resize(outputNodes);
        for (int i = 0; i < hiddenNodes; ++i) {
            biasHidden[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        for (int i = 0; i < outputNodes; ++i) {
            biasOutput[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    
    // Feedforward
    vector<double> feedForward(const vector<double>& input) {
        vector<double> hiddenOutput(hiddenNodes);
        vector<double> output(outputNodes);
        
        // Calculate hidden layer's output
        for (int i = 0; i < hiddenNodes; ++i) {
            double sum = 0.0;
            for (int j = 0; j < inputNodes; ++j) {
                sum += input[j] * weightsInputHidden[j][i];
            }
            sum += biasHidden[i];
            hiddenOutput[i] = sigmoid(sum);
        }
        
        // Calculate output layer's output
        for (int i = 0; i < outputNodes; ++i) {
            double sum = 0.0;
            for (int j = 0; j < hiddenNodes; ++j) {
                sum += hiddenOutput[j] * weightsHiddenOutput[j][i];
            }
            sum += biasOutput[i];
            output[i] = sigmoid(sum);
        }
        
        return output;
    }
    
    // Backpropagation
    void train(const vector<double>& input, const vector<double>& target) {
        // Feedforward
        vector<double> hiddenOutput(hiddenNodes);
        vector<double> output(outputNodes);
        
        // Calculate hidden layer's output
        for (int i = 0; i < hiddenNodes; ++i) {
            double sum = 0.0;
            for (int j = 0; j < inputNodes; ++j) {
                sum += input[j] * weightsInputHidden[j][i];
            }
            sum += biasHidden[i];
            hiddenOutput[i] = sigmoid(sum);
        }
        
        // Calculate output layer's output
        for (int i = 0; i < outputNodes; ++i) {
            double sum = 0.0;
            for (int j = 0; j < hiddenNodes; ++j) {
                sum += hiddenOutput[j] * weightsHiddenOutput[j][i];
            }
            sum += biasOutput[i];
            output[i] = sigmoid(sum);
        }
        
        // Backpropagation
        // Output layer
        vector<double> outputError(outputNodes);
        for (int i = 0; i < outputNodes; ++i) {
            outputError[i] = (target[i] - output[i]) * sigmoidDerivative(output[i]);
        }
        
        // Hidden layer
        vector<double> hiddenError(hiddenNodes);
        for (int i = 0; i < hiddenNodes; ++i) {
            double error = 0.0;
            for (int j = 0; j < outputNodes; ++j) {
                error += outputError[j] * weightsHiddenOutput[i][j];
            }
            hiddenError[i] = error * sigmoidDerivative(hiddenOutput[i]);
        }
        
        // Update weights and biases
        for (int i = 0; i < outputNodes; ++i) {
            for (int j = 0; j < hiddenNodes; ++j) {
                weightsHiddenOutput[j][i] += learningRate * outputError[i] * hiddenOutput[j];
            }
            biasOutput[i] += learningRate * outputError[i];
        }
        
        for (int i = 0; i < hiddenNodes; ++i) {
            for (int j = 0; j < inputNodes; ++j) {
                weightsInputHidden[j][i] += learningRate * hiddenError[i] * input[j];
            }
            biasHidden[i] += learningRate * hiddenError[i];
        }
    }
};

double meanSquaredError(const vector<vector<double>>& predictions, const vector<vector<double>>& targets) {
    double sumSquaredError = 0.0;
    int numSamples = predictions.size();
    int numOutputs = predictions[0].size();
    
    for (int i = 0; i < numSamples; ++i) {
        for (int j = 0; j < numOutputs; ++j) {
            double error = targets[i][j] - predictions[i][j];
            sumSquaredError += error * error;
        }
    }
    
    return sumSquaredError / (numSamples * numOutputs);
}

int main() {
    srand(time(NULL)); // Seed random number generator
    
    // Define network parameters
    int inputNodes = 2;
    int hiddenNodes = 2;
    int outputNodes = 1;
    double learningRate = 0.1;
    
    // Create a neural network
    NeuralNetwork neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate);
    
    // Define input and target for AND gate
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{1}, {0}, {0}, {1}};
    
    // Train the network
    int epochs = 10000; // Number of training iterations
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < inputs.size(); ++i) {
            neuralNetwork.train(inputs[i], targets[i]);
        }
        
        // Calculate and display cost function
        vector<vector<double>> predictions;
        for (int i = 0; i < inputs.size(); ++i) {
            vector<double> output = neuralNetwork.feedForward(inputs[i]);
            predictions.push_back(output);
        }
        double cost = meanSquaredError(predictions, targets);
        
        // Display result after each iteration
        cout << "Iteration " << epoch + 1 << ", Cost: " << cost << ":\n";
        for (int i = 0; i < inputs.size(); ++i) {
            vector<double> output = neuralNetwork.feedForward(inputs[i]);
            cout << inputs[i][0] << " XOR " << inputs[i][1] << " = " << output[0] << endl;
        }
        cout << "--------------------------\n";
    }
    
    return 0;
}
