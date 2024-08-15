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
            //Calculate Weight based off previous inputs and the amount of nodes
            w = Tensor::random(vec![amount, biases.data.len()]);
        } else if let Some(Layer::InputLayer { size }) = self.layers.back() {
            w = Tensor::random(vec![amount, size.clone()]);
        }

        //Adds tensor layers to the network depending on amount specified
        let weights: Tensor = w; //Creates iterator and creates that many tensors and puts it into a Vector
        let mut biases: Tensor = Tensor::random(vec![amount]); //Creates iterator and creates that many tensors and puts it into a Vector
        if biases.shape.len() < 2 {
            biases.increase_dim(1);
        }
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
        if input.shape.len() < 2 {
            current_output.increase_dim(1);
        }
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
                    let mut w: Tensor = weights.clone();
                    //w.increase_dim(1);
                    println!(
                        "Current = shape:\n{:?}\ndata:\n{:?}",
                        current_output.shape, current_output.data
                    );
                    current_output = w.multiply(&current_output);

                    // Add the biases
                    current_output.add(biases);

                    // Apply each activation function
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

    pub fn back_propogate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) {
        for layer in &self.layers {
            match layer {
                Layer::TensorLayer {
                    weights,
                    biases,
                    activations,
                } => {

                    //code goes here
                }
                Layer::InputLayer { size } => {
                    print!("Input Layer")
                }
            }
        }
    }
}
