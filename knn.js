class Matrix {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
  }
  static fromArray(arr) {
    return new Matrix(arr.length, 1).map((e, i) => arr[i]);
  }

  static subtract(a, b) {
    if (a.rows !== b.rows || a.cols !== b.cols) {
      console.log('Columns and Rows of A must match Columns and Rows of B.');
      return;
    }

    // Return a new Matrix a-b
    return new Matrix(a.rows, a.cols)
      .map((_, i, j) => a.data[i][j] - b.data[i][j]);
  }

  toArray() {
    let arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  randomize() {
    return this.map(e => Math.random() * 2 - 1);
  }

  add(n) {
    if (n instanceof Matrix) {
      if (this.rows !== n.rows || this.cols !== n.cols) {
        console.log('Columns and Rows of A must match Columns and Rows of B.');
        return;
      }
      return this.map((e, i, j) => e + n.data[i][j]);
    } else {
      return this.map(e => e + n);
    }
  }
  static add(a, b) {
    if (a.rows !== b.rows || a.cols !== b.cols) {
      console.log('Columns and Rows of A must match Columns and Rows of B.');
      return;
    }
    // Return a new Matrix a-b
    return new Matrix(a.rows, a.cols)
      .map((_, i, j) => a.data[i][j] + b.data[i][j]);
  }
  static transpose(matrix) {
    return new Matrix(matrix.cols, matrix.rows)
      .map((_, i, j) => matrix.data[j][i]);
  }

  static multiply(a, b) {
    // Matrix product
    if (a.cols !== b.rows) {
      console.log('Columns of A must match rows of B.')
      return;
    }

    return new Matrix(a.rows, b.cols)
      .map((e, i, j) => {
        // Dot product of values in col
        let sum = 0;
        for (let k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        return sum;
      });
  }


  multiply(n) {
    if (n instanceof Matrix) {
      if (this.rows !== n.rows || this.cols !== n.cols) {
        console.log('Columns and Rows of A must match Columns and Rows of B.');
        return;
      }
      // hadamard product
      return this.map((e, i, j) => e * n.data[i][j]);
    } else {
      // Scalar product
      return this.map(e => e * n);
    }
  }

  map(func) {
    // Apply a function to every element of matrix
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        let val = this.data[i][j];
        this.data[i][j] = func(val, i, j);
      }
    }
    return this;
  }

  static map(matrix, func) {
    // Apply a function to every element of matrix
    return new Matrix(matrix.rows, matrix.cols)
      .map((e, i, j) => func(matrix.data[i][j], i, j));
  }

  print() {
    console.table(this.data);
    return this;
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data == 'string') {
      data = JSON.parse(data);
    }
    let matrix = new Matrix(data.rows, data.cols);
    matrix.data = data.data;
    return matrix;
  }
}


class ActivationFunction {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

let sigmoid = new ActivationFunction(
  x => 1 / (1 + Math.exp(-x)),
  y => y * (1 - y)
);

let tanh = new ActivationFunction(
  x => Math.tanh(x),
  y => 1 - (y * y)
);


class NeuralNetwork {
  /*
   * if first argument is a NeuralNetwork the constructor clones it
   * USAGE: cloned_nn = new NeuralNetwork(to_clone_nn);
   */
  constructor(in_nodes, hid_nodes, out_nodes) {
    if (in_nodes instanceof NeuralNetwork) {
      let a = in_nodes;
      this.input_nodes = a.input_nodes;
      this.hidden_nodes = a.hidden_nodes;
      this.output_nodes = a.output_nodes;

      this.weights_ih = a.weights_ih.copy();
      this.weights_ho = a.weights_ho.copy();

      this.bias_h = a.bias_h.copy();
      this.bias_o = a.bias_o.copy();
    } else {
      this.input_nodes = in_nodes;
      this.hidden_nodes = hid_nodes;
      this.output_nodes = out_nodes;

      this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
      this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
      this.weights_ih.randomize();
      this.weights_ho.randomize();

      this.bias_h = new Matrix(this.hidden_nodes, 1);
      this.bias_o = new Matrix(this.output_nodes, 1);
      this.bias_h.randomize();
      this.bias_o.randomize();
    }

    // TODO: copy these as well
    this.setLearningRate();
    this.setActivationFunction();


  }

  predict(input_array) {

    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    // activation function!
    hidden.map(this.activation_function.func);

    // Generating the output's output!
    let output = Matrix.multiply(this.weights_ho, hidden);
    output.add(this.bias_o);
    output.map(this.activation_function.func);

    // Sending back to the caller!
    return output.toArray();
  }

  setLearningRate(learning_rate = 0.1) {
    this.learning_rate = learning_rate;
  }

  setActivationFunction(func = sigmoid) {
    this.activation_function = func;
  }

  train(input_array, target_array) {
    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    // activation function!
    hidden.map(this.activation_function.func);

    // Generating the output's output!
    let outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(this.activation_function.func);

    // Convert array to matrix object
    let targets = Matrix.fromArray(target_array);

    // Calculate the error
    // ERROR = TARGETS - OUTPUTS
    let output_errors = Matrix.subtract(targets, outputs);

    // let gradient = outputs * (1 - outputs);
    // Calculate gradient
    let gradients = Matrix.map(outputs, this.activation_function.dfunc);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);


    // Calculate deltas
    let hidden_T = Matrix.transpose(hidden);
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

    // Adjust the weights by deltas
    this.weights_ho.add(weight_ho_deltas);
    // Adjust the bias by its deltas (which is just the gradients)
    this.bias_o.add(gradients);

    // Calculate the hidden layer errors
    let who_t = Matrix.transpose(this.weights_ho);
    let hidden_errors = Matrix.multiply(who_t, output_errors);

    // Calculate hidden gradient
    let hidden_gradient = Matrix.map(hidden, this.activation_function.dfunc);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);

    // Calcuate input->hidden deltas
    let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

    this.weights_ih.add(weight_ih_deltas);
    // Adjust the bias by its deltas (which is just the gradients)
    this.bias_h.add(hidden_gradient);

    // outputs.print();
    // targets.print();
    // error.print();
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data == 'string') {
      data = JSON.parse(data);
    }
    let nn = new NeuralNetwork(data.input_nodes, data.hidden_nodes, data.output_nodes);
    nn.weights_ih = Matrix.deserialize(data.weights_ih);
    nn.weights_ho = Matrix.deserialize(data.weights_ho);
    nn.bias_h = Matrix.deserialize(data.bias_h);
    nn.bias_o = Matrix.deserialize(data.bias_o);
    nn.learning_rate = data.learning_rate;
    return nn;
  }


  // Adding function for neuro-evolution
  copy() {
    return new NeuralNetwork(this);
  }

  // Accept an arbitrary function for mutation
  mutate(func) {
    this.weights_ih.map(func);
    this.weights_ho.map(func);
    this.bias_h.map(func);
    this.bias_o.map(func);
  }
}

class KNN {
  constructor(il, k) {
    this.k = k || 1
    this.length = il
    this.model = new NeuralNetwork(this.length, 200, 10)
    this.examples = new Array()
    this.label = new Array()
  }

  static pixels(image) {
    let i = image
    let canva = document.createElement("canvas")
    canva.height = i.height
    canva.width = i.width
    let ctx = canva.getContext("2d")
    let red = []
    let green = []
    let blue = []
    let alpha = []
    ctx.drawImage(i, 0, 0, i.width, i.height)
    let d = ctx.getImageData(0, 0, canva.width, canva.height).data
    for (let i = 0; i < d.length; i += 4) {
      red.push(d[i]);
      green.push(d[i + 1]);
      blue.push(d[i + 2]);
      alpha.push(d[i + 3]);
    }
    return {
      "red": red,
      "green": green,
      "blue": blue,
      "alpha": alpha
    }
  }

  addExample(image, l) {
    let i = new Image(Math.sqrt(this.length), Math.sqrt(this.length))
    i.src = image.src
    let data = KNN.pixels(i)
    let inputArray = new Array(data.red.length)
    for (let i = 0; i < data.red.length; i++) {
      inputArray[i] = data.green[i] + data.red[i] + data.blue[i]
    }
    this.examples.push({
      infer: this.model.predict(inputArray),
      label: l
    })
    if (this.label.indexOf(l) < 0) {
      this.label.push(l)
    }
  }

  static distance(v1, v2) {
    let sum = 0
    for (let i = 0; i < v1.length; i++) {
      sum += Math.pow(v2[i] - v1[i], 2)
    }
    return Math.sqrt(sum)
  }

  static kNearest(distances, k) {
    let kn = []
    for (let i = 0; i < k; i++) {
      let smallest = Infinity
      let n, index
      for (let j = 0; j < distances.length; j++) {
        if (distances[j].distance < smallest) {
          smallest = distances[j].distance
          n = distances[j]
          index = j
        }
      }
      kn.push(n)
      distances.splice(index, 1)
    }
    return kn
  }
  static mostDuplicate(arr) {
    let result = {},
      max = 0,
      res;

    for (var i = 0, total = arr.length; i < total; ++i) {
      let val = arr[i],
        inc = (result[val] || 0) + 1;

      result[val] = inc;

      if (inc > max) {
        max = inc;
        res = val;
      }
    }
    return res
  }
  classify(image) {
    let i = new Image(Math.sqrt(this.length), Math.sqrt(this.length))
    i.src = image.src
    let data = KNN.pixels(i)
    let distances = new Array()
    let inputArray = new Array(data.red.length)
    for (let i = 0; i < data.red.length; i++) {
      inputArray[i] = data.green[i] + data.red[i] + data.blue[i]
    }
    let infer = this.model.predict(inputArray)

    for (let example of this.examples) {
      let d = KNN.distance(infer, example.infer)
      distances.push({
        distance: d,
        label: example.label
      })
    }
    let kn = KNN.kNearest(distances, this.k)
    let values = []
    for (let i = 0; i < kn.length; i++) {
      values.push(this.label.indexOf(kn[i].label))
    }
    let v = this.label[KNN.mostDuplicate(values)]
    let sum = 0
    for (let i = 0; i < values.length; i++) {
      if (values[i] == KNN.mostDuplicate(values)) {
        sum++
      }
    }
    let confidence = sum / values.length
    return {
      label: v,
      confidence: confidence,
      index: this.label.indexOf(v)
    }
  }

}
