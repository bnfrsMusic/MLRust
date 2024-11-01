use super::{activations::Activation, tensor::Tensor};

pub struct LossFunction {
    pub name: &'static str,
    pub function: fn(&Tensor, &Tensor) -> f64,
    pub derivative: fn(&Tensor, &Tensor, &Activation<'static>) -> Tensor,
}

pub const MSE: LossFunction = LossFunction {
    name: "Mean Squared Error",
    function: |predicted: &Tensor, actual: &Tensor| {
        //Function

        //MSE = 1/n Sum[i=1] (actual - predicted)^2
        let mut diff = actual.clone();

        //(actual - predicted)^2
        diff.subtract(predicted); //calculates the difference
        diff = diff.dot(&diff); //Squares the difference

        //Sum[i=1] / n
        diff.data.iter().sum::<f64>() / (actual.data.len() as f64)
    },
    derivative: |predicted: &Tensor, actual: &Tensor, _activation: &Activation<'static>| {
        // Derivative of MSE
        assert!(
            predicted.shape == actual.shape,
            "Shape mismatch: predicted shape {:?} and actual shape {:?}",
            predicted.shape,
            actual.shape
        );

        let n = predicted.data.len() as f64; // number of elements
        let mut gradient = Tensor::new(predicted.shape.clone());

        for i in 0..predicted.data.len() {
            gradient.data[i] = (predicted.data[i] - actual.data[i]) * (2.0 / n);
        }

        gradient
    },
};
