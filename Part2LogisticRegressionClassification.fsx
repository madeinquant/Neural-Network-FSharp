// http://peterroelants.github.io/posts/neural_network_implementation_intermezzo01/
#load "../packages/FsLab.1.0.2/FsLab.fsx"
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions

open XPlot.GoogleCharts

// Define and generate the samples
let numOfSamplesPerClass = 20;;  // The number of sample in each class
let v1 = vector [-1.;0.];;
let z = [for i in 1..20 -> (v1)];;
let redMean = DenseMatrix.OfRowVectors z;;  // The mean of the red class
let v2 = vector [1.;0.];;
let z2 = [for i in 1..20 -> (v2)];;
let blueMean = DenseMatrix.OfRowVectors z2;;  // The mean of the blue class
let stdDev = 1.2;;  // standard deviation of both classes
// Generate samples from both classes
// let xRed = np.random.randn(nb_of_samples_per_class, 2) * stdDev + redMean;;
let xRed = (DenseMatrix.random<float> numOfSamplesPerClass 2 (Normal(0.0, sqrt 1.0))) * stdDev + redMean;;
// let xBlue = np.random.randn(nb_of_samples_per_class, 2) * stdDev + blueMean;;
let xBlue = (DenseMatrix.random<float> numOfSamplesPerClass 2 (Normal(0.0, sqrt 1.0))) * stdDev + blueMean;;

// Merge samples in set of input variables x, and corresponding set of output variables t
let X = DenseMatrix.stack [xRed; xBlue];;
let t = DenseMatrix.stack [DenseMatrix.init numOfSamplesPerClass 1 (fun _ _ -> 0.); 
                           DenseMatrix.init numOfSamplesPerClass 1 (fun _ _ -> 1.)];;
let points1 = [ for i in 0..numOfSamplesPerClass-1  -> xRed.[i,0], xRed.[i,1] ];;
let points2 = [ for i in 0..numOfSamplesPerClass-1  -> xBlue.[i,0], xBlue.[i,1] ];;
let options1 =
  Options
    ( title = "Gradient descent updates plotted on cost function", 
      curveType = "function",
      hAxis = Axis(title = "X1"),
      vAxis = Axis(title = "X2") );;

[points1;points2]
  |> Chart.Scatter
  |> Chart.WithOptions options1;;
