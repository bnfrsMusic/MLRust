pub mod lib;
use std::vec;


use lib::{network::NeuralNetwork, activations::Activation, loss::LossFunction};


fn main() {
    
    // network: 2 inputs -> 5 hidden -> 1 output
    let mut n = NeuralNetwork::new(2);
    n.add_layer(4, Activation::TanH);
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
    let epochs = 1000;
    
    println!("\n=== Training XOR Gate ===");
    
    // Training loop
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        
        // Train
        for (inputs, targets) in &training_data {

            n.back_propagation(
                inputs.clone(),
                targets.clone(),
                LossFunction::MeanSquaredError,
                learning_rate,
            );
            let output = n.feed_forward_with_cache(inputs.clone());
            let predicted = if output[0] > 0.5 { 1.0 } else { 0.0 };

            total_loss += (predicted - output[0]).abs();

        }
        

        // Print progress every 100 epochs
        if epoch % 100 == 0 {
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




