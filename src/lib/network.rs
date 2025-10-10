use ndarray::prelude::*;
use ndarray::{Array, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use crate::{Activation, LossFunction};

#[derive(Clone, Debug)]
pub enum Layer
{
    InputLayer{
        size: usize
    },

    TensorLayer{
        weights: Array<f64, IxDyn>,
        biases: Array<f64, IxDyn>,
        activation: Activation,

        //Cache for backprop
        last_input: Option<Array1<f64>>,
        last_z: Option<Array1<f64>>,  // pre-activation
        last_output: Option<Array1<f64>>,  // post-activation
    }
}

pub struct NeuralNetwork
{
    layers: Vec<Layer>
}

impl NeuralNetwork
{

    //Creates a new network based on Input Size
    pub fn new(input_size: usize) -> NeuralNetwork {
    
        NeuralNetwork { layers: vec![Layer::InputLayer { size: input_size }] }

    }

    pub fn add_layer(&mut self, amount: usize, activation: Activation){

        let w:Array<f64, IxDyn>;

        // Get last layer
        if let Some(Layer::TensorLayer { 
            biases, .. 
        }) = self.layers.last() {
        
            //Calculate Weight based off previous inputs and the amount of nodes
            w = Array::random((amount, biases.len()), Uniform::new(0., 1.)).into_dyn();
        
        } else if let Some(Layer::InputLayer { size }) = self.layers.last() {
        
            // Handle input layer case
            w = Array::random((amount, *size), Uniform::new(0., 1.)).into_dyn();
        
        } else {
        
            // Handle case where there are no previous layers
            panic!("Cannot add layer: no previous layers found");
        
        }

        //Create biases
        let b = Array::random(amount, Uniform::new(0., 1.)).into_dyn();

        let new_layer = Layer::TensorLayer {
            weights: w,
            biases: b,
            activation,
            last_input: None,
            last_z: None,
            last_output: None,
        };

        self.layers.push(new_layer);
    }

    


    //--------------------------------------------------------------Feed Forward / Back Propogation---------------------------------------------------------------------

    /// Feed forward
    pub fn feed_forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        let mut current_output = input;
        
        for layer in &mut self.layers {
            match layer {
                Layer::TensorLayer {
                    weights,
                    biases,
                    activation, ..
                } => {
                    let weights_2d = weights.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                    let biases_1d = biases.view().into_dimensionality::<ndarray::Ix1>().unwrap();
                    
                    let input_len = current_output.len();
                    let output_len = biases_1d.len();
                    
                    //Manual matrix-vector multiplication
                    let mut new_output = vec![0.0; output_len];
                    for i in 0..output_len {
                        let mut sum = biases_1d[i]; //start with bias
                        for j in 0..input_len {
                            sum += weights_2d[[i, j]] * current_output[j];
                        }
                        new_output[i] = activation.as_func()(&sum);
                    }
                    
                    current_output = new_output;
                }
                Layer::InputLayer { size: _ } => {
                    println!("Input Layer");
                }
            }
        }
        
        current_output
    }

    /// Feed forward with caching for backpropagation
    pub fn feed_forward_with_cache(&mut self, input: Vec<f64>) -> Vec<f64> {
        let mut current_output = Array1::from(input);
    
        for layer in &mut self.layers {
            match layer {
                Layer::TensorLayer {
                    weights,
                    biases,
                    activation,
                    last_input,
                    last_z,
                    last_output,
                } => {
                    // Store input to this layer
                    *last_input = Some(current_output.clone());
                    
                    let weights_2d = weights.view().into_dimensionality::<Ix2>().unwrap();
                    let biases_1d = biases.view().into_dimensionality::<Ix1>().unwrap();
                
                    let input_len = current_output.len();
                    let output_len = biases_1d.len();
                
                    // Manual matrix-vector multiplication
                    let mut z = Array1::zeros(output_len);  // pre-activation
                    for i in 0..output_len {
                        let mut sum = biases_1d[i];
                        for j in 0..input_len {
                            sum += weights_2d[[i, j]] * current_output[j];
                        }
                        z[i] = sum;
                    }
                    
                    //Store pre-activation values
                    *last_z = Some(z.clone());
                    
                    //activation
                    let mut activated = Array1::zeros(output_len);
                    for i in 0..output_len {
                        activated[i] = activation.as_func()(&z[i]);
                    }
                    
                    //Store post-activation output
                    *last_output = Some(activated.clone());
                    current_output = activated;
                }
                Layer::InputLayer { size: _ } => {
                    // println!("Input Layer");
                }
            }
        }
    
        current_output.to_vec()
    }

    /// Backprop thru SGD on single sample at a time
    pub fn back_propagation(
        &mut self,
        inputs: Vec<f64>,
        targets: Vec<f64>,
        loss: LossFunction,
        learning_rate: f64,
    ) {
        // Step 1: Forward pass with caching --> to store the caches
        let outputs = self.feed_forward_with_cache(inputs.clone());
        
        // Step 2: Compute loss derivative with respct to outputs
        let output_array = Array1::from(outputs);
        let target_array = Array1::from(targets);
        
        //Get gradient
        let mut delta = loss.derivative(&output_array, &target_array);
        
        // Step 3: Backpropagate through layers in reverse
        for layer in self.layers.iter_mut().rev() {
            match layer {
                Layer::TensorLayer {
                    weights,
                    biases,
                    activation,
                    last_input,
                    last_output,
                    ..
                } => {
                    
                    
                    // Get cached values 
                    let layer_output = last_output
                        .as_ref()
                        .expect("No cached output - did you run feed_forward first?");
                    let layer_input = last_input
                        .as_ref()
                        .expect("No cached input - did you run feed_forward first?");
                    
                    
                    // Step 3a: Apply activation derivative: δ = δ ⊙ σ'(activated_output)
                    let activation_derivative = activation.derivative_array1(layer_output);
                    delta = &delta * &activation_derivative;
                    
                    let output_len = delta.len();
                    let input_len = layer_input.len();
                    
                    // Step 3b: Compute weight gradients (outer product): ∂L/∂W = δ ⊗ aᵀ
                    let mut weight_gradients = Array2::zeros((output_len, input_len));
                    for i in 0..output_len {
                        for j in 0..input_len {
                            weight_gradients[[i, j]] = delta[i] * layer_input[j];
                        }
                    }
                    
                    
                    // Step 3c: Compute gradient for previous layer first (before mutating weights)
                    // δ_prev = Wᵀ · δ
                    let mut delta_prev = Array1::zeros(input_len);
                    {
                        //Scope immutable borrow into this block as it was giving me errors
                        let weights_2d = weights
                            .view()
                            .into_dimensionality::<Ix2>()
                            .expect("Weights must be 2D");
                        
                        for j in 0..input_len {
                            let mut sum = 0.0;
                            for i in 0..output_len {
                                sum += weights_2d[[i, j]] * delta[i];
                            }
                            delta_prev[j] = sum;
                        }
                    } 
                    
                    // Step 3d: Update biases: b = b - η * δ
                    {
                        let mut biases_1d = biases
                            .view_mut()
                            .into_dimensionality::<Ix1>()
                            .expect("Biases must be 1D");
                        
                        for i in 0..output_len {
                            biases_1d[i] -= learning_rate * delta[i];
                        }
                    }
                    
                    // Step 3e: Update weights: W = W - η * ∇W
                    {
                        let mut weights_2d_mut = weights
                            .view_mut()
                            .into_dimensionality::<Ix2>()
                            .expect("Weights must be 2D");
                        
                        for i in 0..output_len {
                            for j in 0..input_len {
                                weights_2d_mut[[i, j]] -= learning_rate * weight_gradients[[i, j]];
                            }
                        }
                    }
                    
                    // Update delta for next iteration (previous layer)
                    delta = delta_prev;
                }
                Layer::InputLayer { .. } => {
                    // Stop at input layer
                    break;
                }
            }
        }
    }

    // pub fn mini_batch_gradient_descent(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, batch_size: usize, learning_rate: f64, epochs: usize) {

    // }
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
                    activation, ..
                } => {
                    println!("Layer weight shape: {:?}", weights.shape());
                    println!("Layer biases shape: {:?}", biases.shape());
                    println!("Activation: {:?} ", activation);
                }
            }
        }
    }



}


