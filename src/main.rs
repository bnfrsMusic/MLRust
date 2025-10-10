pub mod lib;
use std::vec;


use lib::{network::NeuralNetwork, activations::Activation, loss::LossFunction};


fn main() {
    
    // network: 2 inputs -> 5 hidden -> 1 output
    let mut n = NeuralNetwork::new(2);
    n.add_layer(5, Activation::Sigmoid);
    n.add_layer(1, Activation::Sigmoid);
    
    println!("=== Initial Network ===");
    n.print_network();
    
    // XOR gate
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    
    let learning_rate = 0.5;
    let epochs = 10000;
    
    println!("\n=== Training XOR Gate ===");
    
    // Training loop
    for epoch in 0..epochs {
        let total_loss = 0.0;
        
        // Train
        for (inputs, targets) in &training_data {

            n.back_propagation(
                inputs.clone(),
                targets.clone(),
                LossFunction::MeanSquaredError,
                learning_rate,
            );
        }
        
        // Print progress every 1000 epochs
        if epoch % 1000 == 0 {
            let avg_loss = total_loss / training_data.len() as f64;
            println!("Epoch {}: Average Loss = {:.6}", epoch, avg_loss);
        }
    }
    
    println!("\n=== Testing Trained Network ===");
    
    // Test all XOR combinations
    for (inputs, expected) in &training_data {
        let output = n.feed_forward_with_cache(inputs.clone());
        let predicted = if output[0] > 0.5 { 1.0 } else { 0.0 };
        
        println!(
            "Input: {:?} | Expected: {:.1} | Output: {:.4} | Predicted: {:.1}",
            inputs, expected[0], output[0], predicted
        );
    }
    
    println!("\n=== Final Network ===");
    n.print_network();
}




















// fn main(){


//     let mut n = NeuralNetwork::new(2);
//     n.add_layer(5, Activation::Linear);
//     n.add_layer(2, Activation::Linear);

//     n.print_network();


//     let x =  n.feed_forward(vec![0.0,1.0]);

//     println!("Feed Forward {:?}", x);

// }



// fn main() {
//     // Create a simple XOR problem
//     let inputs = Tensor::new(vec![
//         vec![0.0, 0.0],
//         vec![0.0, 1.0],
//         vec![1.0, 0.0],
//         vec![1.0, 1.0],
//     ]);
    
//     let targets = Tensor::new(vec![
//         vec![0.0],
//         vec![1.0],
//         vec![1.0],
//         vec![0.0],
//     ]);

//     // Creating a network with structure:
//     // Input layer (2 neurons) -> Hidden layer (3 neurons) -> Output layer (1 neuron)
//     let mut network = Network::new(vec![
//         Layer::new(2, 3, Activation::ReLU),
//         // Layer::new(3, 3, Activation::Sigmoid),
//         Layer::new(3, 1, Activation::Sigmoid),

//     ]);

//     // Training parameters
//     let learning_rate = 0.08;
//     let epochs = 100000;
//     let loss_function = LossFunction::MeanSquaredError;
    

//     // Training loop
//     println!("Training network for XOR problem...");
//     for epoch in 0..epochs {
//         // Forward pass
//         let outputs = network.forward(&inputs);
        
//         // Compute loss
//         let loss = loss_function.compute(&outputs, &targets);
        
//         // Only print loss sometimes to avoid flooding the console
//         if epoch % 1000 == 0 {
//             println!("Epoch {}, Loss: {}", epoch, loss);
//         }
        
//         // Backward pass - compute gradients and update weights
//         network.backward(&inputs, &targets, &loss_function, learning_rate);
//     }

//     // Test the network after training
//     println!("\nTesting network after training:");
//     let outputs = network.forward(&inputs);
    
//     println!("XOR Results:");
//     // Fixed: access tuple element with .0 instead of [0]
//     for i in 0..inputs.shape().0 {
//         let input_row = inputs.get_row(i);
//         let output = outputs.get_value(i, 0);
//         println!("Input: [{:.1}, {:.1}], Output: {:.6}, Expected: {:.1}", 
//                  input_row[0], input_row[1], output, targets.get_value(i, 0));
//     }

// }



