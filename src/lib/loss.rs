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
        diff.substract(predicted); //calculates the difference
        diff.dot(&diff); //Squares the difference

        //Sum[i=1] / n
        diff.data.iter().sum::<f64>() / (actual.data.len() as f64)
    },
    derivative: |actual: &Tensor, predicted: &Tensor, act: &Activation<'static>| {
        print!("TARGETS: {:?}", actual.data);
        //println!("\n\n\nACT shape: {:?}", actual.shape);
        let mut der = predicted.clone();
        let mut cost = predicted.clone();

        der = der.map(act.derivative);
        //makes sure the actual and predicted shape are the same (just in case I screwed up somewhere)
        assert_eq!(actual.shape, predicted.shape);

        cost.substract(actual);
        der.multiply(&cost);

        assert_eq!(der.shape, predicted.shape);
        println!("DER Shape: {:?}", der.shape);
        der.map(&|x| x * -2.0);
        assert_eq!(der.shape, predicted.shape);
        //panic!("Predicted: {:?}\nDer: {:?}", predicted.data, der.data);

        der
    },
};
