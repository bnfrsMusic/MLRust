use crate::lib::loss::MSE;

use super::{activations::Activation, tensor::Tensor};
use std::collections::VecDeque;

//#[derive(Clone)]
pub enum Layer {
    InputLayer {
        size: usize,
    },
    TensorLayer {
        weights: Tensor,
        biases: Tensor,
        activations: Activation<'static>,
    },
}
//
pub struct CPUTensorNetwork {
    layers: VecDeque<Layer>,
}

//
impl CPUTensorNetwork {
    //Constructor
    pub fn new(input_size: usize) -> CPUTensorNetwork {
        //Initiallizes the network
        let mut layersVec = VecDeque::new();
        layersVec.push_back(Layer::InputLayer { size: input_size });
        CPUTensorNetwork { layers: layersVec }
    }

    //--------------------------------------------------------------Layers---------------------------------------------------------------------

    //
    pub fn add_tensor_layer(&mut self, amount: usize, act: Activation<'static>) {
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
            weights.shape, weights.data, biases.shape, biases.data
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
                    //for activation in activations {
                    current_output = current_output.map(activations.function);
                    //}
                }
                Layer::InputLayer { size } => {
                    print!("Input Layer")
                }
            }
        }

        current_output
        //.expect("No tensor output found. Network might be empty or improperly configured.")
    }

    pub fn back_propogate(&mut self, outputs: Tensor, targets: Tensor, learning_rate: f64) {
        //targets => the correct value

        let mut tot_res: Vec<Tensor> = vec![];
        for layer in self.layers.iter_mut().rev() {
            match layer {
                Layer::InputLayer { size } => {
                    println!("Input shape: {:?}", size);
                }
                Layer::TensorLayer {
                    weights,
                    biases,
                    activations,
                } => {
                    //Finish this

                    println!("Layer weight shape: {:?}", weights.shape);
                    println!("Layer weight data: {:?}", weights.data);
                    println!("Layer biases shape: {:?}", biases.shape);
                    println!("Activation: {:?} ", activations.name);
                    println!(
                        "\nMSE: {:?}",
                        (MSE.derivative)(&outputs, &targets, activations).data
                    );
                    tot_res.push((MSE.derivative)(&outputs, &targets, activations));
                    //println!("RESULT SHAPE: {:?}\nRESULT DATA: {:?}", res.shape, res.data)
                }
            }
        }
        for tensor in tot_res {
            println!(
                "RESULT Tensor Shape: {:?}\nData: {:?}",
                tensor.shape, tensor.data
            )
        }
    }
    pub fn train(&mut self, input: Tensor, targets: Tensor, epoch: usize, learning_rate: f64) {
        for i in 0..epoch {
            println!("\n\n-------Current Epoch: {:?}-------", i);
            self.back_propogate(
                self.feed_forward(input.clone()),
                targets.clone(),
                learning_rate,
            )
        }
    }

    //-------------------------------Debug Tools----------------------------------
    pub fn print_network(&mut self) {
        //targets => the correct value

        for layer in self.layers.iter_mut() {
            match layer {
                Layer::InputLayer { size } => {
                    println!("Input shape: {:?}", size);
                }
                Layer::TensorLayer {
                    weights,
                    biases,
                    activations,
                } => {
                    //Finish this

                    println!("Layer weight shape: {:?}", weights.shape);
                    println!("Layer weight data: {:?}", weights.data);
                    println!("Layer biases shape: {:?}", biases.shape);
                    println!("Activation: {:?} ", activations.name);
                }
            }
        }
    }
}
