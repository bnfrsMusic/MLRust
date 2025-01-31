use lib::{activations::SIGMOID, tensor::Tensor};
use std::{fs::OpenOptions, io::Write, vec};
pub mod lib;
use lib::cpu_tensor_network::CPUTensorNetwork;

fn main() {
    //Clears out the file
    let mut data_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("delta.txt")
        .expect("Cannot open file");

    data_file
        .write("--\n".as_bytes())
        .expect("Unable to write to file");

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
        Tensor::from(vec![2, 1], vec![0.0, 0.0]),
    ];
    let target_arr: Vec<Tensor> = vec![
        Tensor::from(vec![1, 1], vec![1.0]),
        Tensor::from(vec![1, 1], vec![0.0]),
        Tensor::from(vec![1, 1], vec![0.0]),
        Tensor::from(vec![1, 1], vec![0.0]),
    ];

    let input_tensor = Tensor::from(vec![2, 1], vec![1.0, 0.0]);

    //Before training
    let output_tensor = network.feed_forward(input_tensor.clone());

    network.print_network();

    // //------------------------Training------------------------
    for n in 0..input_arr.len() {
        //does not seem to work if i train with more than two data points (two input -> output pairs)
        //might be due to model complexity being limited by the current dimension support.
        /*
        Possible Fixes:
        - IT IS OVER-FITTING TO THE LAST GIVEN DATA POINTS
        - Allow any size neural networks so that there are more parameters
        - ...idk

        */
        network.train(input_arr[n].clone(), target_arr[n].clone(), 100, 0.05);
        println!(
            "Output {}: {:?}",
            n,
            network.feed_forward(input_arr[n].clone()).data
        );
    }
    //------------------------Printing Results----------------
    println!("-----------------BEFORE-------------------");
    println!("Output Tensor: {:?}", output_tensor.data); // Print the output Tensor
                                                         //output_tensor = network.feed_forward(input_arr[0].clone());
    println!("-----------------AFTER-------------------");
    for n in 0..input_arr.len() {
        let mut results_file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open("results.txt")
            .expect("Cannot open results file");

        writeln!(
            results_file,
            "Output {}: {:?}",
            n,
            network.feed_forward(input_arr[n].clone()).data
        )
        .expect("Unable to write to results file");

        println!(
            "Output {}: {:?}",
            n,
            network.feed_forward(input_arr[n].clone()).data
        );
    }
    //println!("Output Tensor: {:?}", output_tensor.data); // Print the output Tensor
    println!("Elapsed time: {:.2?}", now.elapsed());
}
