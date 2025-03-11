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
        let mut layers_vec = VecDeque::new();
        layers_vec.push_back(Layer::InputLayer { size: input_size });
        CPUTensorNetwork { layers: layers_vec }
    }

    //--------------------------------------------------------------Layers---------------------------------------------------------------------

    //
    pub fn add_tensor_layer(&mut self, amount: usize, act: Activation<'static>) {
        let mut w: Tensor = Tensor::new(Vec::new());
        if let Some(Layer::TensorLayer {
            weights:_, biases, ..
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
                Layer::InputLayer { size:_ } => {
                    println!("Input Layer")
                }
            }
        }

        current_output
    }

    // Returns the deltas from each layer in the network to be used in training
    fn get_delta(&mut self, _input: &Tensor, targets: Tensor) -> (Vec<Tensor> , Vec<Tensor>) {
        let mut bias_delta: Vec<Tensor> = Vec::new();
        let mut weight_delta: Vec<Tensor> = Vec::new();

        let mut outputs = self.feed_forward(_input.clone());
        // Gets the initial activation function (when going backwards)
        let init_activation = match self
            .layers
            .iter()
            .rev()
            .find(|layer| matches!(layer,   Layer::TensorLayer { .. }))
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
        let mut delta = (MSE.derivative)(&outputs, &targets, &init_activation);

        //iterate though each layer in reverse
        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            if let Layer::TensorLayer {
                weights,
                biases: _,
                activations,
                result: _,
            } = layer
            {
                // Calculate delta for the next layer (if any)
                if i < results.len() - 1 {
                    // Update biases
                    bias_delta.push(delta.clone());
                    // Calculate weight gradient
                    let weight_gradient =
                        // delta.multiply(&outputs.map(activations.derivative));
                        results[i + 1].multiply(&delta.transpose()).transpose();

                    assert_eq!(&weights.shape, &weight_gradient.shape);
                    weight_delta.push(weight_gradient.clone());

                    delta = weights.transpose().multiply(&delta);
                    delta = delta.multiply(&outputs.map(activations.derivative));
                    outputs = results[i + 1].clone(); // Set outputs for the next layer
                } else {
                    println!("Delta: {:?}", delta.data);
                }
            }
        }
        (bias_delta, weight_delta)
    }


    // Average deltas returned by get_delta
    fn avg_delta(deltas: &Vec<Tensor>) -> Tensor {
        assert!(!deltas.is_empty(), "Cannot average an empty vector of Tensors");
    
        // Start with the first delta as the initial sum.
        let mut sum = deltas[0].clone();
    
        // Sum the remaining Tensors.
        for delta in deltas.iter().skip(1) {
            sum.add(delta);
        }
    
        // Divide by the number of Tensors to get the average.
        let count = deltas.len() as f64;
        sum.multiply_scalar(1.0 / count)
    }



    pub fn back_propogate(&mut self, _input: &Tensor, targets: Tensor, learning_rate: f64) {
        let mut outputs = self.feed_forward(_input.clone());
        // Gets the initial activation function (when going backwards)
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
        let mut delta = (MSE.derivative)(&outputs, &targets, &init_activation);
        // delta = delta.map(&init_activation.derivative);
        println!(
            "init DELTA: \nShape{:?}\nData:{:?}\n",
            delta.shape, delta.data
        );

        //iterate though each layer in reverse
        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            if let Layer::TensorLayer {
                weights,
                biases,
                activations,
                result: _,
            } = layer
            {
                // Calculate delta for the next layer (if any)
                if i < results.len() - 1 {
                    // Update biases
                    biases.subtract(&delta.multiply_scalar(learning_rate));

                    // Calculate weight gradient
                    let mut weight_gradient =
                        results[i + 1].multiply(&delta.transpose()).transpose();
                    println!(
                        "Weight gradient Shape {:?}, Data {:?}",
                        weight_gradient.shape, weight_gradient.data
                    );
                    assert_eq!(&weights.shape, &weight_gradient.shape);

                    weights.subtract(&weight_gradient.multiply_scalar(learning_rate));
                    delta = weights.transpose().multiply(&delta);
                    delta = delta.multiply(&outputs.map(activations.derivative));
                    outputs = results[i + 1].clone(); // Set outputs for the next layer
                } else {
                    println!("Delta: {:?}", delta.data);
                }
            }
        }
    }


    pub fn train(
        &mut self,
        input: Vec<Tensor>,
        targets: Vec<Tensor>,
        epoch: usize,
        batch_size: usize,
        learning_rate: f64,
    ) {
        let batches = input.len() / batch_size;
    
        // Get the network layer structure to understand what we are working with
        let layer_shapes = self.layers.iter().enumerate()
            .filter_map(|(idx, layer)| match layer {
                Layer::TensorLayer { weights, biases, .. } => {
                    Some((idx, weights.shape.clone(), biases.shape.clone()))
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        

        // Print the network structure
        println!("Network structure:");
        for (idx, w_shape, b_shape) in &layer_shapes {
            println!("Layer {}: Weights shape: {:?}, Biases shape: {:?}", idx, w_shape, b_shape);
        }
    
        
        for i in 0..epoch {
            println!("\n\n-------Current Epoch: {:?}-------", i);
            
            // Process full batches
            for j in 0..batches {
                // Collect deltas for each sample in the batch
                let mut batch_bias_deltas: Vec<Vec<Tensor>> = Vec::new();
                let mut batch_weight_deltas: Vec<Vec<Tensor>> = Vec::new();
                
                // Process each sample in the batch
                for k in j * batch_size..(j + 1) * batch_size {
                    let (bias_delta, weight_delta) = self.get_delta(&input[k], targets[k].clone());
                    
                    // Add the deltas to our collection
                    batch_bias_deltas.push(bias_delta);
                    batch_weight_deltas.push(weight_delta);
                }
                
                // Only proceed if we have collected deltas
                if !batch_bias_deltas.is_empty() && !batch_weight_deltas.is_empty() {
                    // Reorganize deltas by layer
                    let mut bias_deltas_by_layer: Vec<Vec<Tensor>> = vec![Vec::new(); layer_shapes.len()];
                    let mut weight_deltas_by_layer: Vec<Vec<Tensor>> = vec![Vec::new(); layer_shapes.len()];
                    
                    // Group deltas by layer
                    for sample_bias_deltas in batch_bias_deltas {
                        for (layer_idx, delta) in sample_bias_deltas.iter().enumerate() {
                            if layer_idx < bias_deltas_by_layer.len() {
                                bias_deltas_by_layer[layer_idx].push(delta.clone());
                            }
                        }
                    }
                    
                    for sample_weight_deltas in batch_weight_deltas {
                        for (layer_idx, delta) in sample_weight_deltas.iter().enumerate() {
                            if layer_idx < weight_deltas_by_layer.len() {
                                weight_deltas_by_layer[layer_idx].push(delta.clone());
                            }
                        }
                    }
                    
                    // Average deltas for each layer and apply them
                    let mut tensor_layer_index = 0;
                    for (layer_idx, layer) in self.layers.iter_mut().rev().enumerate() {
                        if let Layer::TensorLayer { weights, biases, .. } = layer {
                            // Only process layers that have deltas
                            if tensor_layer_index < bias_deltas_by_layer.len() && 
                               !bias_deltas_by_layer[tensor_layer_index].is_empty() {
                                
                                // Average the bias deltas for this layer
                                let mut avg_bias_delta = Self::avg_delta(&bias_deltas_by_layer[tensor_layer_index]);
                                
                                // Average the weight deltas for this layer
                                let mut avg_weight_delta = Self::avg_delta(&weight_deltas_by_layer[tensor_layer_index]);
                                
                                // Apply the averaged deltas
                                biases.subtract(&avg_bias_delta.multiply_scalar(learning_rate));
                                weights.subtract(&avg_weight_delta.multiply_scalar(learning_rate));
                                
                                println!("Applied averaged deltas to layer {}", layer_idx);
                                
                                tensor_layer_index += 1;
                            }
                        }
                    }
                }
            }
            
            // Handle remaining samples that don't fit into a full batch
            if input.len() > batches * batch_size {
                let mut remainder_bias_deltas: Vec<Vec<Tensor>> = Vec::new();
                let mut remainder_weight_deltas: Vec<Vec<Tensor>> = Vec::new();
                
                for j in batches * batch_size..input.len() {
                    let (bias_delta, weight_delta) = self.get_delta(&input[j], targets[j].clone());
                    remainder_bias_deltas.push(bias_delta);
                    remainder_weight_deltas.push(weight_delta);
                }
                
                // Process the remainder similarly to batches
                if !remainder_bias_deltas.is_empty() && !remainder_weight_deltas.is_empty() {
                    let mut bias_deltas_by_layer: Vec<Vec<Tensor>> = vec![Vec::new(); layer_shapes.len()];
                    let mut weight_deltas_by_layer: Vec<Vec<Tensor>> = vec![Vec::new(); layer_shapes.len()];
                    
                    //remainder bias delta
                    for sample_bias_deltas in remainder_bias_deltas {
                        for (layer_idx, delta) in sample_bias_deltas.iter().enumerate() {
                            if layer_idx < bias_deltas_by_layer.len() {
                                bias_deltas_by_layer[layer_idx].push(delta.clone());
                            }
                        }
                    }
                    
                    //remainder weight delta
                    for sample_weight_deltas in remainder_weight_deltas {
                        for (layer_idx, delta) in sample_weight_deltas.iter().enumerate() {
                            if layer_idx < weight_deltas_by_layer.len() {
                                weight_deltas_by_layer[layer_idx].push(delta.clone());
                            }
                        }
                    }
                    
                    // Apply the remainder deltas
                    let mut tensor_layer_index = 0;
                    for (layer_idx, layer) in self.layers.iter_mut().rev().enumerate() {
                        if let Layer::TensorLayer { weights, biases, .. } = layer {
                            if tensor_layer_index < bias_deltas_by_layer.len() && 
                               !bias_deltas_by_layer[tensor_layer_index].is_empty() {
                                
                                let mut avg_bias_delta = Self::avg_delta(&bias_deltas_by_layer[tensor_layer_index]);
                                let mut avg_weight_delta = Self::avg_delta(&weight_deltas_by_layer[tensor_layer_index]);
                                
                                biases.subtract(&avg_bias_delta.multiply_scalar(learning_rate));
                                weights.subtract(&avg_weight_delta.multiply_scalar(learning_rate));
                                
                                println!("Applied remainder deltas to layer {}", layer_idx);
                                
                                tensor_layer_index += 1;
                            }
                        }
                    }
                }
            }
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
    pub fn return_network(&mut self) -> String {
        //targets => the correct value
        let mut network_str = String::new();
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
                    network_str.push_str(&format!("Layer weight shape: {:?}\n", weights.shape));
                    network_str.push_str(&format!("Layer biases shape: {:?}\n", biases.shape));
                    network_str.push_str(&format!("Activation: {:?}\n", activations.name));
                    network_str.push_str(&format!("Layer result shape: {:?}\n", result.shape));
                    network_str.push_str(&format!("Layer result data: {:?}\n", result.data));
                }
            }
        }
        network_str
    }
}
