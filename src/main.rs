use lib::{activations::SIGMOID, tensor::Tensor};
use std::vec;
pub mod lib;
use lib::cpu_tensor_network::CPUTensorNetwork;

fn main() {
    use std::time::Instant;
    let now = Instant::now();

    let mut network = CPUTensorNetwork::new(2);
    //network.add_tensor_layer(2, vec![SIGMOID]);
    network.add_tensor_layer(3, SIGMOID);
    network.add_tensor_layer(3, SIGMOID);
    network.add_tensor_layer(2, SIGMOID);

    let input_tensor = Tensor::from(vec![2, 1], vec![1.0, 0.0]); // Create a Tensor with shape [2] and data [1.0, 0.0]
    let target_tensor = Tensor::from(vec![2, 1], vec![1.0, 0.0]); // Create a Tensor with shape [2] and data [1.0, 0.0]
    let output_tensor = network.feed_forward(input_tensor.clone()); // Feed forward through the network

    //network.train(input_tensor.clone(), target_tensor.clone(), 10, 0.5);
    println!("------------------------------------");

    network.print_network();
    network.train(input_tensor, target_tensor, 1, 0.5);
    println!("Output Tensor: {:?}", output_tensor.data); // Print the output Tensor
    println!("Elapsed time: {:.2?}", now.elapsed());
}
