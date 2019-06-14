class KNN {
  constructor(k) {
    this.k = k || 1
    this.examples = new Array()
    this.label = new Array()
  }
  
  addExample(data, l) {
    this.examples.push({
      infer: data,
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
  classify(data) {
    let infer = data
    let distances = new Array()
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
