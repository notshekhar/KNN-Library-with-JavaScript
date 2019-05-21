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


class fnn {
  constructor(arr, lr) {
    this.neurons = [];
    this.weights = [];
    this.lr = lr || 0.01;
    let arrlen = arr.length;
    for (let i = 0; i < arrlen; i++) {
      this.neurons.push(arr[i]);
    }
    for (let i = 0; i < arrlen - 1; i++) {
      let weight = new Matrix(this.neurons[i + 1], this.neurons[i]);
      weight.randomize();
      this.weights.push(weight);
    }
    
  }
  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }
  dsigmoid(y) {
    // return sigmoid(x) * (1s - sigmoid(x));
    return y * (1 - y);
  }
  tanh(x) {
    var y = Math.tanh(x);
    return y;
  }
  
  dtanh(x) {
    var y = 1 / (pow(Math.cosh(x), 2));
    return y;
  }
  predict(inputarr) {
    let inputs = Matrix.fromArray(inputarr);
    let outputs = [];
    let weightlen = this.weights.length;
    for (let i = 0; i < weightlen; i++) {
      inputs = Matrix.multiply(this.weights[i], inputs);
      inputs.map(this.sigmoid);
      outputs.push(inputs);
    }
    return outputs;
    
  }
  query(arr) {
    let outputs = this.predict(arr);
    let output = outputs[outputs.length - 1].toArray();
    return output;
    
  }
  
  learn(input, outputarr) {
    let inputs = Matrix.fromArray(input);
    let weightlen = this.weights.length;
    for (let i = 0; i < weightlen; i++) {
      inputs = Matrix.multiply(this.weights[i], inputs);
      inputs.map(this.sigmoid);
    }
    let output = inputs;
    let answer = Matrix.fromArray(outputarr);
    let err = Matrix.subtract(answer, output);
    let errors = [];
    for (var i = this.weights.length - 1; i >= 0; i--) {
      errors.push(err);
      err = Matrix.multiply(Matrix.transpose(this.weights[i]), err);
    }
    errors.reverse();
    let outputs = [];
    let inpout = [];
    inpout.push(Matrix.fromArray(input));
    let inp = Matrix.fromArray(input)
    for (let i = 0; i < weightlen; i++) {
      inp = Matrix.multiply(this.weights[i], inp);
      inp.map(this.sigmoid);
      outputs.push(inp);
      inpout.push(inp);
    }
    
    for (let i = 0; i < errors.length; i++) {
      let gradient = errors[i].multiply(Matrix.map(outputs[i], this.dsigmoid));
      let dweight = Matrix.multiply(gradient, Matrix.transpose(inpout[i]));
      dweight.multiply(this.lr);
      this.weights[i] = this.weights[i].add(dweight);
    }
    
    
    
  }
  setLearningRate(learn) {
    this.lr = learn;
    
  }
  
  
  download(filename) {
    let arr = {
      "weights": this.weights
    }
    
    let datStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(arr));
    let downloadNode = document.createElement("a");
    downloadNode.setAttribute("href", datStr);
    downloadNode.setAttribute("download", filename + ".json");
    downloadNode.click();
    downloadNode.remove();
    
  }
  upload(weights) {
    for (let i = 0; i < this.weights.length; i++) {
      for (let m = 0; m < this.weights[i].rows; m++) {
        for (let n = 0; n < this.weights[i].cols; n++) {
          this.weights[i].data[m][n] = weights[i].data[m][n];
        }
      }
    }
  }
  accuracy(input, arr) {
    let answer = arr;
    let output = this.query(input);
    answer = math.findmax(answer);
    output = math.findmax(output);
    let score = [];
    if (answer == output) {
      score.push(1);
    } else {
      score.push(0);
    }
    let accuracy = math.sumArray(score) / score.length * 100;
    return accuracy;
    
  }
  
}

class KNN{
  constructor(il, k){
    this.k || 1
    this.length = il
    this.model = new fnn([this.length, 200, 10])
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
  
  addExample(image, l){
    let i = new Image(Math.sqrt(this.length), Math.sqrt(this.length))
    i.src = image.src
    let data = KNN.pixels(i)
    let inputArray = new Array(data.red.length)
    for(let i=0; i<data.red.length; i++){
      inputArray[i] = data.green[i]+data.red[i]+data.blue[i]
    }
    this.examples.push({infer: this.model.query(inputArray), label: l})
    if (this.label.indexOf(l) < 0) {
      this.label.push(l)
    }
  }
  
  static distance(v1, v2) {
    let sum = 0
    for(let i=0; i<v1.length; i++){
      sum += Math.pow(v2[i]-v1[i], 2)
    }
    return Math.sqrt(sum)
  }
  
  static kNearest(distances, k){
    let kn = []
    for(let i=0; i<k; i++){
      console.log(1)
      let smallest = Infinity
      let n, index
      for(let j=0; j<distances.length; j++){
        if(distances[j].distance<smallest){
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
  classify(image){
    let i = new Image(Math.sqrt(this.length), Math.sqrt(this.length))
    i.src = image.src
    let data = KNN.pixels(i)
    let distances = new Array()
    let inputArray = new Array(data.red.length)
    for (let i = 0; i < data.red.length; i++) {
      inputArray[i] = data.green[i] + data.red[i] + data.blue[i]
    }
    let infer = this.model.query(inputArray)
    
    for(let example of this.examples){
      let d = KNN.distance(infer, example.infer)
      distances.push({distance: d, label: example.label})
    }
    return KNN.kNearest(distances, 1)
    
  }
  
}



let i = document.querySelector('img')
let i2 = document.querySelectorAll('img')[1]
let m = new KNN(100, 1)
m.addExample(i, "shekhar")
m.addExample(i2, "shekha")
console.log(m.classify(i))