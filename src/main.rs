use lib::{activations::SIGMOID, tensor::Tensor};
use std::{fs::OpenOptions, io::Write, vec};
pub mod lib;
use lib::cpu_tensor_network::CPUTensorNetwork;

fn main() {
    use std::time::Instant;
    let now = Instant::now();

    let mut network = CPUTensorNetwork::new(2);
    network.add_tensor_layer(3, SIGMOID); //Backprop currently can NOT handle diff dimensions
    network.add_tensor_layer(3, SIGMOID);

    //    network.add_tensor_layer(3, SIGMOID);
    network.add_tensor_layer(1, SIGMOID);

    let input_arr: Vec<Tensor> = vec![
        Tensor::from(vec![2, 1], vec![1.0, 0.0]),
        Tensor::from(vec![2, 1], vec![0.0, 1.0]),
        Tensor::from(vec![2, 1], vec![1.0, 1.0]),
        // Tensor::from(vec![2, 1], vec![0.0, 0.0]),
    ];
    let target_arr: Vec<Tensor> = vec![
        Tensor::from(vec![1, 1], vec![0.0]),
        Tensor::from(vec![1, 1], vec![1.0]),
        Tensor::from(vec![1, 1], vec![1.0]),
        // Tensor::from(vec![1, 1], vec![0.0]),
    ];

    let input_tensor = Tensor::from(vec![2, 1], vec![1.0, 0.0]);

    //Before training
    let mut output_tensor = network.feed_forward(input_tensor.clone());

    network.print_network();

    // //------------------------Training------------------------
    // network.train(input_arr.clone(), target_arr.clone(), 1000, 2, 0.05);
    network.train(input_arr.clone(), target_arr.clone(), 1000, 1, 0.05);

    //------------------------Printing Results----------------
    println!("-----------------BEFORE-------------------");
    // println!("Output Tensor: {:?}", output_tensor.data); // Print the output Tensor
    // for n in 0..input_arr.len() {
    //     println!(
    //         "Output {}: {:?}",
    //         n,
    //         network.feed_forward(input_arr[n].clone()).data
    //     );
    // }
    output_tensor = network.feed_forward(input_arr[0].clone());

    println!("-----------------AFTER-------------------");

    for n in 0..input_arr.len() {
        println!(
            "Output {}: {:?}",
            n,
            network.feed_forward(input_arr[n].clone()).data
        );
    }
    println!("Elapsed time: {:.2?}", now.elapsed());
    println!("-----------------Debug-------------------");

    network.print_network();
}
