import { matrix_multiplication, calc_net_and_act_iter } from "./numlib.js";

function matrix_multiplication_cpu(a, b) {
  var aNumRows = a.length, aNumCols = a[0].length,
      bNumRows = b.length, bNumCols = b[0].length,
      m = new Array(aNumRows);  // initialize array of rows
  for (var r = 0; r < aNumRows; ++r) {
    m[r] = new Array(bNumCols); // initialize the current row
    for (var c = 0; c < bNumCols; ++c) {
      m[r][c] = 0;             // initialize the current cell
      for (var i = 0; i < aNumCols; ++i) {
        m[r][c] += a[r][i] * b[i][c];
      }
    }
  }
  return m;
}


async function run(size) {
/*
    const f =  [[1, 2, 3, 4],[5, 6, 7, 8]], // Original matrices
          s = [[1, 2],
               [3, 4],
               [5, 6],
               [7, 8]];
*/
    let m = [], sm;
    for (let i = 1; i <= size; i++) {
        sm = [];
        for (let j = 1; j <= size; j++) {
            sm.push(j);
        }
        m.push(sm);
    }

   // console.log("Matrix: ",m);

    console.time('gpu');
    const g = await matrix_multiplication(m,m);
    // console.log("Matrix: ",g);
    console.timeEnd('gpu');

    console.time('cpu');
    const c = matrix_multiplication_cpu(m,m);
    // console.log("Matrix: ",c);
    console.timeEnd('cpu');

}

//run(1000);


async function run_iter(size,iter) {

    let m1 = [], v2 = [], v, v1, sm, smi, m_indices = [];

    for (let i = 1; i <= size; i++) {
        sm = [];
        smi = [];
        for (let j = 1; j <= size; j++) {
            sm.push(j/(5));
            smi.push(2); // Pseudo index
        }
        m1.push(sm); // 2-dimensional 'weight' vector (full connectivity)
        m_indices.push(smi); // (pseudo) indices of sparse weights
    }

    for (let j = 1; j <= size; j++) {
        v2.push(j/(5));
    }
    v2 = [v2]; // 1-dimensional 'activation' vector

    console.time('gpu'+iter);
    v = await calc_net_and_act_iter(v2,m1,m_indices,iter);
    console.log("Matrix GPU: ",v);
    console.timeEnd('gpu'+iter);

    console.time('cpu'+iter);
    v1 = v2.slice();
    for (let i = 0; i < iter; ++i) {
        v1 = matrix_multiplication_cpu(v1,m1);
    }
    console.log("Matrix CPU final: ",v1);
    console.timeEnd('cpu'+iter);
}

//run_iter(2000,1000);
run_iter(4,5);

