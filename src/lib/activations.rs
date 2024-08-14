use std::f64::consts::E;

#[derive(Clone)]

pub struct Activation<'a> {
    // lifetime of struct is 'a
    pub function: &'a dyn Fn(f64) -> f64,
    pub derivative: &'a dyn Fn(f64) -> f64,
    pub name: &'a str,
}
//Sigmoid Function
pub const SIGMOID: Activation = Activation {
    function: &|x| 1.0 / (1.0 + E.powf(-x)), //function of sigmoid
    derivative: &|x| x * (1.0 - x),          //derivative of sigmoid
    name: "Sigmoid",                         //Sigmoid
};
