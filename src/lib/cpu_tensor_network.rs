use rand::distributions::weighted;

use crate::lib::{activations, pooling};

use super::{activations::Activation, loss::LossFunction, tensor::Tensor};
use std::{any::Any, collections::VecDeque};

//#[derive(Clone)]
pub enum Layer {
    InputLayer {
        size: usize,
    },
    TensorLayer {
        weights: Tensor,
        biases: Tensor,
        activations: Vec<Activation<'static>>,
    },
}
//
pub struct CPUTensorNetwork {
    layers: VecDeque<Layer>,
    learning_rate: f64,
}

//
impl CPUTensorNetwork {
    //Constructor
    pub fn new(input_size: usize) -> CPUTensorNetwork {
        //Initiallizes the network
        let mut layers = VecDeque::new();
        layers.push_back(Layer::InputLayer { size: input_size });
        CPUTensorNetwork {
            layers: layers,
            learning_rate: 0.5,
        }
    }

    //--------------------------------------------------------------Layers---------------------------------------------------------------------

    //
    pub fn add_tensor_layer(&mut self, amount: usize, act: Vec<Activation<'static>>) {
        let mut w: Tensor = Tensor::new(Vec::new());
        if let Some(Layer::TensorLayer {
            weights, biases, ..
        }) = self.layers.back()
        {
            //            println!("Previous layer weights:");
            //
            //            println!("weight data: {:?}", weights.data); // Replace this with appropriate method to print Tensor contents
            //            println!("weight shape: {:?}", weights.shape);
            //
            //            println!("Previous layer biases:");
            //            println!("bias data: {:?}", biases.data);
            //            println!("bias shape: {:?}", biases.shape);
            //Calculate Weight based off previous inputs and the amount of nodes
            w = Tensor::random(vec![biases.data.len(), amount]);

        //            println!(
        //                "Biases: {:?} and ideal weight shape: {:?}, ideal weight data: {:?}",
        //                biases.data, w.shape, w.data,
        //            );
        } else if let Some(Layer::InputLayer { size }) = self.layers.back() {
            //            println!("Input Layer Neurons: {:?}", size);
            w = Tensor::random(vec![size.clone(), amount]);
            //            println!(
            //                "Coming ftom inpuy shape {:?} and data {:?}",
            //                w.shape, w.data
            //            );
        }

        //Adds tensor layers to the network depending on amount specified
        let weights: Tensor = w; //Creates iterator and creates that many tensors and puts it into a Vector
        let biases: Tensor = Tensor::random(vec![amount]); //Creates iterator and creates that many tensors and puts it into a Vector
        let activations = act;

        println!("------------------------------------");
        println!(
            "Weights shape: {:?} \n Weights data: {:?} \n Biases shape: {:?} \n Biases data: {:?}",
            weights.data, weights.shape, biases.data, biases.shape
        );
        self.layers.push_back(Layer::TensorLayer {
            weights,
            biases,
            activations,
        });
    }

    //

    //--------------------------------------------------------------Feed Forward / Back Propogation---------------------------------------------------------------------

    pub fn feed_forward(&self, input: Tensor) -> Tensor {
        println!(
            "Feeding forward {:?} with shape {:?}",
            input.data, input.shape
        );

        let mut current_output = input.clone();
        let mut final_output: Option<Tensor> = None;

        for layer in &self.layers {
            match layer {
                Layer::TensorLayer {
                    weights,
                    biases,
                    activations,
                } => {
                    println!("Processing through Tensor Layer");
                    // Perform matrix multiplication
                    current_output = current_output.multiply(weights);

                    // Add the biases
                    current_output = current_output.add(biases);

                    // Apply each activation function in sequence
                    for activation in activations {
                        current_output = current_output.map(activation.function);
                    }
                }
                Layer::InputLayer { size } => {
                    print!("Input Layer")
                }
            }
        }

        current_output
        //.expect("No tensor output found. Network might be empty or improperly configured.")
    }
}
