mod tensor;
mod network;
mod activations;
mod loss;

use network::{Network, Layer};
use tensor::Tensor;
use activations::Activation;
use loss::LossFunction;

fn main() {
    // Create a simple XOR problem
    let inputs = Tensor::new(vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ]);
    
    let targets = Tensor::new(vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ]);

    // Creating a network with structure:
    // Input layer (2 neurons) -> Hidden layer (3 neurons) -> Output layer (1 neuron)
    let mut network = Network::new(vec![
        Layer::new(2, 3, Activation::Sigmoid),
        Layer::new(3, 1, Activation::Sigmoid),
    ]);

    // Training parameters
    let learning_rate = 0.05;
    let epochs = 100000;
    let loss_function = LossFunction::MeanSquaredError;

    // Training loop
    println!("Training network for XOR problem...");
    for epoch in 0..epochs {
        // Forward pass
        let outputs = network.forward(&inputs);
        
        // Compute loss
        let loss = loss_function.compute(&outputs, &targets);
        
        // Only print loss sometimes to avoid flooding the console
        if epoch % 1000 == 0 {
            println!("Epoch {}, Loss: {}", epoch, loss);
        }
        
        // Backward pass - compute gradients and update weights
        network.backward(&inputs, &targets, &loss_function, learning_rate);
    }

    // Test the network after training
    println!("\nTesting network after training:");
    let outputs = network.forward(&inputs);
    
    println!("XOR Results:");
    // Fixed: access tuple element with .0 instead of [0]
    for i in 0..inputs.shape().0 {
        let input_row = inputs.get_row(i);
        let output = outputs.get_value(i, 0);
        println!("Input: [{:.1}, {:.1}], Output: {:.6}, Expected: {:.1}", 
                 input_row[0], input_row[1], output, targets.get_value(i, 0));
    }
}