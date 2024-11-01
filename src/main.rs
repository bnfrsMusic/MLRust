use lib::{activations::SIGMOID, tensor::Tensor};
use std::vec;
pub mod lib;
use lib::cpu_tensor_network::CPUTensorNetwork;

fn main() {
    use std::time::Instant;
    let now = Instant::now();

    let mut network = CPUTensorNetwork::new(2);
    network.add_tensor_layer(3, SIGMOID); //Backprop currently can NOT handle diff dimensions

    network.add_tensor_layer(2, SIGMOID);
    network.add_tensor_layer(2, SIGMOID);
    //network.add_tensor_layer(2, SIGMOID);
    let input_arr: Vec<Tensor> = vec![
        Tensor::from(vec![2, 1], vec![1.0, 0.0]),
        Tensor::from(vec![2, 1], vec![0.0, 1.0]),
        Tensor::from(vec![2, 1], vec![1.0, 1.0]),
        //       Tensor::from(vec![2, 1], vec![0.0, 0.0]),
    ];
    let target_arr: Vec<Tensor> = vec![
        Tensor::from(vec![2, 1], vec![1.0, 0.0]),
        Tensor::from(vec![2, 1], vec![1.0, 0.0]),
        Tensor::from(vec![2, 1], vec![0.0, 0.0]),
        //       Tensor::from(vec![2, 1], vec![0.0, 0.0]),
    ];

    let input_tensor = Tensor::from(vec![2, 1], vec![1.0, 0.0]);
    let target_tensor = Tensor::from(vec![2, 1], vec![0.0, 1.0]); //Desired Output

    //Before training
    let mut output_tensor = network.feed_forward(input_tensor.clone());

    network.print_network();

    // //------------------------Training------------------------
    // for n in 0..input_arr.len() {
    //     //does not seem to work if i train with more than two data points (two input -> output pairs)
    //     //might be due to model complexity being limited by the current dimension support.
    //     /*
    //     Possible Fixes:
    //     - Allow any size neural networks so that there are more parameters (most likely to fix)
    //     - ...idk

    //     */

    //     network.train(input_arr[n].clone(), target_arr[n].clone(), 10000, 0.01);
    // }

    // //------------------------Printing Results----------------
    // println!("-----------------BEFORE-------------------");
    // println!("Output Tensor: {:?}", output_tensor.data); // Print the output Tensor
    // output_tensor = network.feed_forward(input_tensor.clone());
    // println!("-----------------AFTER-------------------");
    // println!("Output Tensor: {:?}", output_tensor.data); // Print the output Tensor
    // println!("Elapsed time: {:.2?}", now.elapsed());
}
