use ndarray::{Array, Array1, IxDyn};
use std::fmt::{Display, Formatter, Result};

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Sigmoid,
    ReLU,
    TanH,
    Linear,
}

/// Allows the name of the Activaiton being used to be returned
impl Display for Activation {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let name = match self {
            Activation::Sigmoid => "Sigmoid",
            Activation::ReLU => "ReLU",
            Activation::TanH => "TanH",
            Activation::Linear => "Linear",
        };
        write!(f, "{}", name)
    }
}

impl Activation{
    ///Apply an activation function to an ndarray
    pub fn apply(&self, x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        match self {
            Activation::Sigmoid => x.mapv(|val| 1.0 / (1.0 + (-val).exp())),
            Activation::ReLU => x.mapv(|val| if val > 0.0 { val } else { 0.0 }),
            Activation::TanH => x.mapv(|val| val.tanh()),
            Activation::Linear => x.clone(),
        }
    }
    pub fn as_func(&self) -> impl FnMut(&f64) -> f64 {
        match self {
            Activation::Sigmoid => |val: &f64| 1.0 / (1.0 + (-val).exp()),
            Activation::ReLU => |val: &f64| if *val > 0.0 { *val } else { 0.0 },
            Activation::TanH => |val: &f64| val.tanh(),
            Activation::Linear => |val: &f64| *val,
        }
    }
   
    /// Calculates the derivative of the activation function
    pub fn derivative(&self, activated_values: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        match self {
            Activation::Sigmoid => {
                //Sigmoid: f'(x) = f(x) * (1 - f(x))
                activated_values.mapv(|val| val * (1.0 - val))
            },
            Activation::ReLU => {
                //ReLU: f'(x) = 1 if x > 0, 0 otherwise
                activated_values.mapv(|val| if val > 0.0 { 1.0 } else { 0.0 })
            },
            Activation::TanH => {
                //tanh: f'(x) = 1 - f(x)^2
                activated_values.mapv(|val| 1.0 - val * val)
            },
            Activation::Linear => {
                //linear: f'(x) = 1
                activated_values.mapv(|_| 1.0)
            },
        }
    }
    
    /// Helper method for backpropagation
    pub fn derivative_array1(&self, activated_values: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::Sigmoid => {
                // Sigmoid: f'(x) = f(x) * (1 - f(x))
                activated_values.mapv(|val| val * (1.0 - val))
            }
            Activation::ReLU => {
                // ReLU: f'(x) = 1 if x > 0, 0 otherwise
                activated_values.mapv(|val| if val > 0.0 { 1.0 } else { 0.0 })
            }
            Activation::TanH => {
                // tanh: f'(x) = 1 - f(x)^2
                activated_values.mapv(|val| 1.0 - val * val)
            }
            Activation::Linear => {
                // linear: f'(x) = 1
                Array1::ones(activated_values.len())
            }
        }
    }



}