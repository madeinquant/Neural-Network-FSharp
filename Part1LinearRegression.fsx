#load "../packages/FsLab.1.0.2/FsLab.fsx"
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions

open XPlot.GoogleCharts

// Define the vector of input samples as x, with 20 values sampled from a uniform distribution
// between 0 and 1
// let x = DenseVector.random<float> 20 (Normal(0.0, 1.0));;
let numVector = 20;;

// let x = DenseVector.random<float> numVector (ContinuousUniform());;
//let x = DenseVector.random<float> numVector (Normal(0.0, 1.0));;
let x = vector [  4.17022005e-01;   7.20324493e-01;   1.14374817e-04;
         3.02332573e-01;   1.46755891e-01;   9.23385948e-02;
         1.86260211e-01;   3.45560727e-01;   3.96767474e-01;
         5.38816734e-01;   4.19194514e-01;   6.85219500e-01;
         2.04452250e-01;   8.78117436e-01;   2.73875932e-02;
         6.70467510e-01;   4.17304802e-01;   5.58689828e-01;
         1.40386939e-01;   1.98101489e-01];;

// Generate the target values t from x with small gaussian noise so the estimation won't be perfect.
// Define a function f that represents the line that generates t without noise
let f x = x * 2.0;; f 2.0;;

// Create the targets t with some gaussian noise
let noiseVariance = 0.2;; // Variance of the gaussian noise
// Gaussian noise error for each sample in x
let noise = (DenseVector.random<float> numVector (Normal(0.0, sqrt 1.0))) * noiseVariance;;

// Create targets t
// let t = (Vector.map f x) + noise;;
let t = vector [  1.06079790e+00;   1.22067073e+00;  -3.42568919e-02;
         4.29093462e-01;   3.01954531e-01;   3.01240232e-01;
         1.52396587e-01;   9.20066196e-01;   9.73853093e-01;
         1.17813234e+00;   1.01856022e+00;   1.23369343e+00;
         3.84326454e-01;   1.56908099e+00;   1.19757047e-03;
         1.44700611e+00;   6.96277454e-01;   1.03802895e+00;
         1.43339337e-01;   2.27161850e-01];;

let points = [ for i in 0 .. numVector-1 -> x.[i], t.[i] ]
let options =
  Options
    ( title = "inputs (x) vs targets (t)", curveType = "function",
      legend = Legend(position = "bottom") )

[points]
|> Chart.Scatter
|> Chart.WithOptions options;;

// Define the neural network function y = x * w
let nn (x:Vector<float>) (w:float) = x * w;; // to be correct

// Define the cost function
//let cost y t = ((t - y)^2).sum();;
let cost (y2:Vector<float>) (t2:Vector<float>) = (t2 - y2) * (t2 - y2);;
let square x = x * x * 1.0;;

// let ws = Generate.LinearSpaced(100, 0.0, 0.4);;
let ws = vector [ 0.0;  0.04040404;  0.08080808;  0.12121212;  0.16161616;
        0.2020202 ;  0.24242424;  0.28282828;  0.32323232;  0.36363636;
        0.4040404 ;  0.44444444;  0.48484848;  0.52525253;  0.56565657;
        0.60606061;  0.64646465;  0.68686869;  0.72727273;  0.76767677;
        0.80808081;  0.84848485;  0.88888889;  0.92929293;  0.96969697;
        1.0;  1.05050505;  1.09090909;  1.13131313;  1.17171717;
        1.21212121;  1.25252525;  1.29292929;  1.33333333;  1.37373737;
        1.41414141;  1.45454545;  1.49494949;  1.53535354;  1.57575758;
        1.61616162;  1.65656566;  1.6969697 ;  1.73737374;  1.77777778;
        1.81818182;  1.85858586;  1.8989899 ;  1.93939394;  1.97979798;
        2.02020202;  2.06060606;  2. ;  2.14141414;  2.18181818;
        2.22222222;  2.26262626;  2.3030303 ;  2.34343434;  2.38383838;
        2.42424242;  2.46464646;  2.50505051;  2.54545455;  2.58585859;
        2.62626263;  2.66666667;  2.70707071;  2.74747475;  2.78787879;
        2.82828283;  2.86868687;  2.90909091;  2.94949495;  2.98989899;
        3.03030303;  3.07070707;  3.11111111;  3.15151515;  3.19191919;
        3.23232323;  3.27272727;  3.31313131;  3.35353535;  3.39393939;
        3.43434343;  3.47474747;  3.51515152;  3.55555556;  3.5959596 ;
        3.63636364;  3.67676768;  3.71717172;  3.75757576;  3.7979798 ;
        3.83838384;  3.87878788;  3.91919192;  3.95959596;  4.        ];;

let tmp = nn x ws.[0];; 
// nn x ws.[0];;            // Testing function nn
// cost tmp t;;             // Testing function cost
let costws = [for i in 0..99 -> ((t-(nn x ws.[i])) * (t-(nn x ws.[i])))];;
let points2 = [ for i in 0 .. 99 -> ws.[i], costws.[i] ]
let options2 =
  Options
    ( title = "cost vs. weight", curveType = "function",
      hAxis = Axis(title = "Cost"),
      vAxis = Axis(title = "Weigth"),
      legend = Legend(position = "bottom") )

[points2]
  |> Chart.Line
  |> Chart.WithOptions options2;;

// define the gradient function. Remember that y = nn(x, w) = x * w
let gradient (w:float) (x:Vector<float>) (t:Vector<float>) = 
    2.0 * x * ((nn x w) - t);;

// define the update function delta w
let deltaW (wk:float) (x:Vector<float>) (t:Vector<float>) (learningRate:float) =
    learningRate * gradient wk  x  t;;  // gradient(wk  x  t).sum() 

// Set the initial weight parameter
let mutable w = 0.1;;
// Set the learning rate
let learningRate = 0.1;;
// Start performing the gradient descent updates, and print the weights and cost:
let numOfIterations = 6;;  // number of gradient descent updates
let mutable wCost = [w, cost (nn x w) t];;
let mutable dw = 1.0;;

for i in 1..numOfIterations do 
  dw <- deltaW w x t learningRate // Get the delta w update
  w <- w - dw   // Update the current weight parameter
  wCost <- List.append wCost [w, (cost (nn x w) t)];; // Add weight,cost to list
let points4 = [ for i in 0 ..wCost.Length-2 -> fst(wCost.[i]), snd(wCost.[i]) ]

// let a = wCost |> List.map (fun (x,y) -> [x; y]);;

let points3 = [ for i in 0 .. 99 -> ws.[i], costws.[i] ];;

let options3 =
  Options
    ( title = "Gradient descent updates plotted on cost function", 
      curveType = "function",
      hAxis = Axis(title = "Cost"),
      vAxis = Axis(title = "Weigth") )

[points3;points4]
  |> Chart.Line
  |> Chart.WithOptions options2;;

