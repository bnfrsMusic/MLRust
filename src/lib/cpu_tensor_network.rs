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
        result: Tensor,
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
        let result: Tensor = Tensor::new(vec![amount]);

        if biases.shape.len() < 2 {
            biases.increase_dim(1);
        }
        let activations = act;

        // println!("------------------------------------");
        // println!(
        //     "Weights shape: {:?} \n Weights data: {:?} \n Biases shape: {:?} \n Biases data: {:?}",
        //     weights.shape, weights.data, biases.shape, biases.data
        // );
        self.layers.push_back(Layer::TensorLayer {
            weights,
            biases,
            activations,
            result,
        });
    }

    //

    //--------------------------------------------------------------Feed Forward / Back Propogation---------------------------------------------------------------------

    pub fn feed_forward(&mut self, input: Tensor) -> Tensor {
        println!(
            "Feeding forward {:?} with shape {:?}",
            input.data, input.shape
        );

        let mut current_output = input.clone();

        for layer in &mut self.layers {
            match layer {
                Layer::TensorLayer {
                    weights,
                    biases,
                    activations,
                    result,
                } => {
                    // println!("Processing through Tensor Layer");

                    // Perform matrix multiplication
                    current_output = weights.multiply(&current_output);

                    // Add the biases
                    current_output.add(biases);

                    // Apply the activation function
                    current_output = current_output.map(activations.function);

                    // Assign the computed output to the result
                    *result = current_output.clone(); // Store the output in result
                }
                Layer::InputLayer { size } => {
                    println!("Input Layer")
                }
            }
        }

        current_output
    }

    pub fn back_propogate(&mut self, _input: &Tensor, targets: Tensor, learning_rate: f64) {
        let mut outputs = self.feed_forward(_input.clone());

        let init_activation = match self
            .layers
            .iter()
            .rev()
            .find(|layer| matches!(layer, Layer::TensorLayer { .. }))
        {
            Some(Layer::TensorLayer { activations, .. }) => activations,
            _ => panic!("No TensorLayer found in the network"),
        };

        //get all the results and put them into a vector for access during backprop
        let results: Vec<Tensor> = self
            .layers
            .iter()
            .rev()
            .filter_map(|layer| match layer {
                Layer::TensorLayer { result, .. } => Some(result.clone()),
                _ => None,
            })
            .collect();
        let mut delta = (MSE.derivative)(&outputs, &targets, init_activation);

        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            if let Layer::TensorLayer {
                weights,
                biases,
                activations,
                result,
            } = layer
            {
                // Calculate delta for the next layer (if any)
                if i < results.len() - 1 {
                    // Update biases
                    biases.subtract(&delta.multiply_scalar(learning_rate));

                    // Calculate weight gradient
                    let mut weight_gradient =
                        results[i + 1].multiply(&delta.transpose()).transpose();

                    // panic!(
                    //     delta.shape, biases.shape, weights.shape, weight_gradient.shape
                    // );

                    assert_eq!(&weights.shape, &weight_gradient.shape);

                    weights.subtract(&weight_gradient.multiply_scalar(learning_rate));

                    delta = weights.transpose().multiply(&delta);
                    delta = delta.multiply(&outputs.map(activations.derivative));
                    outputs = results[i + 1].clone(); // Set outputs for the next layer
                }
            }
        }
    }

    pub fn train(&mut self, input: Tensor, targets: Tensor, epoch: usize, learning_rate: f64) {
        for i in 0..epoch {
            println!("\n\n-------Current Epoch: {:?}-------", i);

            self.back_propogate(&input, targets.clone(), learning_rate);
            //self.print_network();
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
                    result,
                } => {
                    println!("Layer weight shape: {:?}", weights.shape);
                    println!("Layer biases shape: {:?}", biases.shape);
                    println!("Activation: {:?} ", activations.name);
                    println!("Layer result shape: {:?}", result.shape);
                    println!("Layer result data: {:?}", result.data);
                }
            }
        }
    }
}
