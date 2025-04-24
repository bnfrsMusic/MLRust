use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

#[derive(Clone)]
pub struct NdArray<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

impl<T> NdArray<T>
where
    T: Default + Clone,
    Standard: Distribution<T>, // Correct trait bound: Standard can produce T values
{
    pub fn new(shape: Vec<usize>) -> NdArray<T> {
        NdArray {
            shape: shape.clone(),
            data: vec![T::default(); shape.iter().product()],
        }
    }
    
    pub fn rand(shape: Vec<usize>) -> NdArray<T> {
        let mut rng = thread_rng();
        NdArray {
            shape: shape.clone(),
            data: (0..shape.iter().product())
                .map(|_| rng.gen::<T>())
                .collect(),
        }
    }

    //---------------------------------------------Getters---------------------------------------------

    pub fn shape(&self)-> Vec<usize>{
        return self.shape.clone();
    }
    
    pub fn size(&self)->usize{
        return self.shape.iter().product();
    }

    pub fn data(&self)-> Vec<T>{
        return self.data.clone()
    }





}
