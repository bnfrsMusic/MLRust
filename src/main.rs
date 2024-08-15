use lib::{activations::SIGMOID, tensor::Tensor};
use std::vec;
pub mod lib;
use lib::cpu_tensor_network::CPUTensorNetwork;

fn main() {
    let mut network = CPUTensorNetwork::new(2);
    //network.add_tensor_layer(2, vec![SIGMOID]);
    network.add_tensor_layer(3, vec![SIGMOID]);
    network.add_tensor_layer(3, vec![SIGMOID]);
    network.add_tensor_layer(1, vec![SIGMOID]);

    let input_tensor = Tensor::from(vec![2], vec![1.0, 0.0]); // Create a Tensor with shape [2] and data [1.0, 0.0]
    let output_tensor = network.feed_forward(input_tensor); // Feed forward through the network

    println!("Output Tensor: {:?}", output_tensor.data); // Print the output Tensor
}
