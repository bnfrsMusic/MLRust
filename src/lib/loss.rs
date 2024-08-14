use super::tensor::Tensor;

pub struct LossFunction {
    pub name: &'static str,
    pub function: fn(&Tensor, &Tensor) -> f64,
    pub derivative: fn(&Tensor, &Tensor) -> Tensor,
}

pub const MSE: LossFunction = LossFunction {
    name: "Mean Squared Error",
    function: |predicted: &Tensor, actual: &Tensor| {
        let diff = predicted.substract(actual);
        let squared = diff.dot(&diff);
        squared.data.iter().sum::<f64>() / (predicted.data.len() as f64)
    },
    derivative: |predicted: &Tensor, actual: &Tensor| {
        let mut diff = predicted.substract(actual);
        let scale = 2.0 / (predicted.data.len() as f64);
        diff.map(&|x| x * scale)
    },
};
