Notes with webgpu test February 2020
====================================

*Jaap Murre, University of Amsterdam, 7-19 Feb 2020*
----------------------------------------------------


# Introduction

WebGPU (or webgpu) is a new project supported by all large browser vendors, like Google,
Apple, Microsoft, Mozilla, etc. Together which WebAssembly, which has comparable support,
it proves to be the next important development in JavaScript-based applications. It is,
however, still at the experimental 'bleeding edge' making implementations difficult. I 
don't think it will go the way of CPU-SIMD on Mozilla (which was abandoned eventually, though
it is now making its comeback in WebAssembly).

In the past week, I have re-examined the state-of-the-art of webgpu and I have now succeeded
in implementing a working pseudo-neural network that is up to 8 times more efficient than a
native implementation on a machine with (only) 24 GPU ALUs (arithmetic logic units), but this
efficiency is only achieved with large networks (1000 neurons, 1000,000 weights) and many 
iterations (say 1000). These, however, are numbers that are still far below the sizes and times
we aim to simulate in the AmyCon project.

On a fast PC with a GTX 680, which as 1536 ALUs, matrix multiplication with a 4000x4000 matrix took (only!) 
2.8 s including reading it back
to the CPU. Pure CPU multiplication took about 848 s. That's over 300 times faster! With 6000x6000 we have 4.3 s and 
2073 s: nearly 500 times faster. Note that we have 36 million synapses and this approximates applying the learning rule
to each and calculating net input and activation. 

This machine has 1 GB memory, which divided by 4 (for floats) translates to 250 million weights. However,
if we need to copy the buffer after the learning rule, we not only lose a massive speed increase but
also limit the number of weights to 125 million. If we take 1000 inweights and losing some bytes to activations
and such, we should be able to simulate a model with 100,000 neurons.

If we get funding (or even if not...), I could get a bigger graphics card, e.g., upgrade
to an Nvidia GeForce GTX 1080 Ti, which has 3584 CUDA cores and 11 GB memory. (About 450 euro new, or used
gaming PC for about 1200 euro.) This will only double the network we can run though, so this is 
probably an unnecessary investment.

## Details of the test implementation

The new implementation is based on a tutorial by François Beaufort, available from Google at:
https://developers.google.com/web/updates/2019/08/get-started-with-gpu-compute-on-the-web. 
Without an example like that, it is currently impossible to get started with webgpu programming
given that the APIs and even languages involved are still shifting. To implement and run the 
code, the following obstacles must be overcome:

 - Webgpu is only supported on experimental versions of browsers. In the case of Chrome, one
   needs to download and install the Canary Browser.

 - Even so, it is still necessary to enable the chrome://flags/#enable-unsafe-webgpu flag.

 - The API relies heavily on GPU and array buffers, where the name 'buffer' is used everywhere,
   rather than distinguishing between, say, array buffers and gpuArrays. To get values from
   the CPU on the GPU, one must attache (map) an array buffer to a gpuArray and then one must
   fill the array buffer with a value from a JavaScript array. By virtue of the mapping, the 
   data are transferred to the gpuArray. This must then be detached to make clear to the run-
   time that no more data can be sent and that the gpuArray is now no longer modifiable. The
   various (many!) use-cases of buffers and such take a while to get used to.

 - Resources (in our case mainly arrays with node and connection values) can be bundled in 
   several 'bind' statements, which are unnecessarily cumbersome (probably due to a role for
   graphics programming not used here).

 - The core code must be written in C shader language in a very specific format, which is then translated
   to WebAssembly. The latter is currently still an experimental technique. 

 - Code like above cannot be run from the file system because of recently introduced CORS
   checks, which are violated. Running it from a server works but is yet an unnecessary and
   distracting step.

 - Having said that, I could not easily get the code to run with Node.js, though this is 
   almost certainly possible. Here, the trouble was with getting the WebAssembly to work
   among others.

 - Debugging is difficult, especially if the mistake resides with the 'bindings' or
   the C-code. The resulting error messages are often very general and don't pinpoint 
   specifics.

 - Because WebGPU is recent and still under development there is virtual no online community
   (in February 2020). Searches online for error message fragments usually only yield 
   discussion about whether or not to include a certain feature and how.

None of the above problems are insurmountable but they make the development process less
than a 'clear sailing'. 


# Initial observations

I first got the matrix multiplication code to work; I did not try to run the graphics-based
code (i.e., rotating triangles and such), though some of the examples could become useful in 
the future, for example, to directly
show results (e.g., spikes, activations, or weights) on the canvas. This will be challenging,
because we would need to integrate the webgpu output with the Cytoscape code, to show the 
output at the appropriate locations. 

In its 'raw' form the code is already faster with large matrices, even on systems like my
laptop with only 24 ALUs or CUDA cores (where my system at home has 1536), compared with CPU-based
code (though not highly optimized, also keeping in mind it can be divided over several
webworkers to achieve some degree of parallellism; my home PC can have 8 webworkers in parallel). 

The webgpu code spends quite a bit of time transferring data to the GPU, compiling the C-code
to WebAssembly, and then transferring the code back to the CPU. Even so, it is still faster
but modestly so (factor 2 or 3). Most of this, however, only needs to be done once in a neural
network, after which many cycles (iterations) can be run. On a biologically plausible neuron,
these are likely to be simulated time-slices like milliseconds.


# Comparison with  WebGL-based gpu.js, which is unusable for neural networks

The gpu.js library was similarly promising when I tried it out a year ago. Matrix multiplication
was fast and could be specified entirely in JavaScript without many of the problems mentioned
above, though it was still cumbersome to import resources like activation parameters to the
GPU. What I found, however, was that the more iterations one ran a program (e.g., a matrix to
the n-th power re-iterated without a roundtrip to the CPU), the longer it took to read out the
memory. This suggested some kind of progressive memory allotment, which caused a lengthy 
freeing operation at readout to the CPU. 

It may be my lack of programming skills, but I could not get this to work. An issue filed did
not get a usable response. When I tried the old code with the latest libraries recently, I found
the same problems. Also, even though the matrix multiplication sample is duplicated everywhere,
there are no implementations with many iterations (matrix power, neural networks). Even worse,
the cited speedups over CPU-based implementations routinely leave out the readout time, which
would give the gpu.js library only a small advantage.

All re-iterated GPU examples are entirely graphics-based, like rotating objects and such, which
write directly to canvas and do not need a trip to the CPU. For out purpuses, the transfer to
CPU is a great bottleneck and there is probably a bug deep in the code that misallocates memory
or fails to free it. 

So, for now the move to WebGPU is the only option to achieve massive parallellism with 
JavaScript.


# Moving forward with Walnut 2.0 on WebGPU

I have now only shown that net input can be calculated very efficiently, assuming that applying
a learning rule will add very little to slow down the process, given that it only takes a few
machine cycles usually. I have also show that it is easy to add additional resources, such as
weight indices, which are used with sparse connections. Several issues need to be investigated
further.


## Learning

The matrix multiplication example is interesting in the sense that it shows that a matrix,
like a weight matrix, can efficiently be transformed (e.g., multiplied). So, we may expect
learning to benefit even more from GPU-based parallellism. For learning to be possible,
we will need to send not just the activations of the to-nodes (called to-acts) but also those
of the from-acts. In case of backprogation, we will also need to send to-error and in case of
the Bi and Poo (1998) learning rule, we will need to send the last timestamp of the from-spike,
where we can count time in ms from starting. An unsigned 32 bit integer has over 4,000,000,000
as its maximum value, given 4,000,000 seconds of simulated run time, which is about 1.5 months.
And we can always reset to 0, keeping in mind that there is a limit to the lag that is still
meaningful. In that sense, we can constantly reset, using only a single byte with a smart reset
system. E.g., at time 255, we subtract 100 from every number, assuming 100 is largest meaningful
number. Then, the system can run forever. Time can be kept with a different system, which allows
longer runs, e.g., a uint64 or double.


## Resources

Different neural paradigms require different neural resources, for example, minimally just
act (activation), where net (net input) remains implicit as copying it from GPU to CPU is time
consuming. 

(Can we devise a message-passing system where we can at anytime pass a single 
variable to indicate we need a certain array of values now? We could easily do this with
WebWorkers with postMessage().)

The main problem for implementation is that we need to set up binding for each resource and
then only map it to the C-WebAssembly code. It seems that most of that can be achieved auto-
matically. For C, we need to generate code from the binding indices, keeping in mind to set
the the readonly specifier correctly.

Resources occur in two variants: mutable and immutable. Activations are typically mutable,
whereas weights need not (in non-learning networks), nor are activation rule parameters, like
a threshold. 

Immutable resources are readonly and shared. Mutable resources are more complex, as they are
first readonly and shared, then transformed (e.g., new act values), and then buffer-copied
to their original buffers. Or they are written to CPU, possibly at certain intervals. 


## Rules

Activation, learning, and other rules must be executed in separate command-encoder passes. Else,
the synchronization may be off. This need not affect efficiency as long as gpu buffers remain
at the gpu and are not transferred continuously to cpu and back. 

Given that rules are implemented in specific types of C-code, it is not trivial to transpile
it from JavaScript. This means that each rule in Walnut 2.0 must provide the required C-code.
For this, we can provide certain elements that can be assumed to be present, given the neural
paradigm. 

We will need to do some experimenting to see what is handy in this respect. How much can we keep
constant? The rule will be provided as a string of C-code, next to a JavaScript implementation.

Can we transpile the Javascript to C? Given that the JavaScript is mainly an expression, it 
may be pretty straightforward. We can use Esprima for that, as an experimental feature. Many
things will be easy such as the shape of the expresions. Types can be handled by assume float
at all times. Javascript `Math.log()` will become `log()` in C.

However, because indices of nodes are handled very differently, it may be preferred to release
a set of notes on how to write C-code and include this with a number of guidelines.

We need to extend the current rules with implementation-variants, like the gpu example. Another
example is a SIMD implementation to be used with WebWorkers, able to achieve an 8x4 = 32 times
speedup on most modern PCs, or even more with a higher number of cores, and without the transfer
problems. 



## Notes on GPU implementation of Walnut 2.0

### Class

We can turn the main routine into a Class that then hides much of the details, like binding (layout)
and C preambulum (which echoes the binding). Binding numbers can be assigned automatically. 

Etc.


### CPU implementation

Can we also wrap a CPU implementation inside the class? How would we go about that? If we allow other paradigms, 
like backpropagation and convolutional networks (= backprop with constraints on weights), can we still use the
gpu as intended here? Activations  of layers must first be updated in sequence and then weight changes must be
backpropagated. Not interesting for neuroscience, so let's leave this out for now. Better use keras or tensorflow.js
for that; they will soon come out with a webgpu implementation as well.


### Weight changes combined with net-act?

It seems that the weight changing (i.e., learning) algorithm can be applied right after the net input
and activation (spike) have been updated, because those weights will not be used by any other code. So,
there are no serial dependencies, which means we can do:

 - Update net input
 - Calc act parameters (v, u, spike, spike time)
 - Change inweights with Bi-Poo-1998 rule, which uses inact spike times, outact spike times, current time.

There is probably not all that much to be gained from combining net-act with weight in terms of speed.

On reason to separate them is that the inacts may or may not be updated once the outacts have been calculated.
For sake of reproducibility, it is better then to separate them. This also makes it easier to run without weight
updates.


### Sparse neural networks only

To simplify (for now) the gpu implementation, we will assume that all networks are sparse and hence use
an array in innode indices. If we have 100,000 neurons we could easily use an uint32 as index (allows over 4 billion
neurons). If these have 1000 inweights (connections), we have 100,000,000 connections. If we have a fairly complex
network with 10 areas, that will give us 10000 neurons per area, hopefully enough to approximate the real areas to
an interesting degree. 

Is it useful to have format where we only keep a list of spike addresses? Or we can have two lists:

 - spikes now
 - last spike before now

Then again, if we assign memory for two arrays of n_nodes (e.g., n = 100,000), then that is still quite small compaired
to the number of weights (e.g., n = 100,000,000). Also, having non-sparse activations will speedup the calculations.


### Odd-even or swap prevents copying, but is this necessary? Answer: No.

We should have two 'buffers' for anything we modify, such as weights and node parameters (act, v, u, spikes, ...). Then,
we can write to say weights2 on even (starting at 0) cycles, using weights1 as input, and reverse this on the odd cycles.

Now consider this. We do not really need to swap out odd-even buffers in this manner.

 1. We first have a net_cycle() and calculate (which implies modify here) all the net inputs on the basis of old acts

 2. Then we calculate and update all new node parameters on the basis of the net input

 3. Then we calculate the weights on the basis of the new activations

Given that a weight is between two nodes that are not dependent on it during the subcycle 3, there is no danger of
race conditions or other weird effects. The weight is in a sense only a local parameter

This means that we can give all modifiable parameters, weights, net, act, v, u, spike, etc. a STORAGE usage without
read-only. Even buffer copying is then not necessary, except on transfer to the CPU.

I should test out whether I can get this to work. 

Names for the subcycles:

 - subcycleNet()
 - subcycleAct()
 - subcycleWeight()


### Application to backprop

Deep learning algorithms use a feedforward-feedback update scheme in which layers of nodes are updated. This means, we cannot
use the above subcycle schema. We need to decompose the layers into a partially ordered set, such that we can identify which
layers can safely be updated in parallel, e.g., ((1),(2,3),(4,5,6),(7)). We then do:

 - for layers in partially ordered set in order: (i.e., first 1, then 2,3 in parallel, then 4,5,6 in parallel, then 7)
   - subcycleNet()
   - subcycleAct()
   - calcError()
 - for layers in partiall order set in reverse order:
   - backpropError()
   - calcWeight()

This may be a bit slower than full parallellism because local caches are invalidated on each element of the partially ordered
set. But this remains to be seen. Conclusion is that backprop will use a completely different cycle approach, although the
basic subcycles can be retained.


### Shared weights

In convolutional networks, weights are shared and bundled in kernels. We should probably also follow this scheme. This will
make the subcycles quite different as well and ensures that we cannot use the same ones as for synchronous or indeed non-
convolutional backpropagation layers. We need to add some type of encoding scheme that identifies this to the subcycles.


### Pattern parallellisation

We can also parallize over patterns and send these to different ALUs, each of which will then do a complete cycle with all 
the subcycles as above. The weight changes will then be recorded, added, and redistributed. Depending on the problem at hand
this might work well. And it will give a completely different approach.


### Using 16-bit or 8-bit precision instead of 32-bit precision for the weights

If we store weights and other parameters as 8bit values, we can have a four fold speedup, although we have to somehow watch
for overflow, which will completely mess up the computations. So, if we have:

    u = a(bv-u)

and parameters are scaled as 8bit decimal values, we can calculate four u-values in one go. Given that this will give us
only a range of -122 to 123 this may still work in terms of parameter encoding but it may lead to overflow conditions during
calculations. We can switch to 1-6bit during calculations but that would probably not be worth the effort. 

This would be a good Computer Science student project, to implement and test SIMD. We could combine this with WebAssembly
SIMD, which is using the true 4 x 32bit parallellism present in all CPUs (not GPUs!) at the moment and being largely unused.
Earlier experimental ('nightly') versions of FireFox could use this but this was taken out again, but rumour has it that it 
has now been added to WebAssembly. With 4 cores or 8 (virtual) processors that would give 24 32-bit CPU (not GPU) processors,
which is much less than 1536 GPU Cuda Cores but still.

There is also another reason to reduce the precision: We can use lower
precision weights (16-bit or even 8-bit) to allow larger neural networks in the available memory on the graphics card. E.g.,
a GTX 1080 Ti has 11 GB. If we use 10 GB for weights and did not need to buffer these, this means we could use about 
5,000,000,000 16-bit weights, which is more than the uint32 address can encode. So, the upper-bound of that for now. 
This is enough to run the entorhinal cortex and hippocampal cortex in 1:1 scale. We should also be able to approach a 
3000 times speedup because the 1080 has well over 3000 Cuda cores.

However, if we want to use sparse neural networks, for each weight we also need to store the node address and limiting this
to 64k is probably too small. This means we will need 8 bytes per weight. So, with 8,000,000,000 bytes memory, we can 
store 1 billion weights or say 1,000,000 neurons each with 1000 incoming weights.


### Module-specific Singleton for device and adaptor

Because the webgpu library does not allow multiple calls to get the device and adaptor, we must wrap these in a 
module-specific singleton pattern. 


### Gross layout of the nngpu.js library

It will use a module pattern, as used now. In terms of options etc., we will follow as much as possible paradigm.js,
with which nngpu.js library must work.


### Sparseness is run at GPU assuming a constant number of inweights

Right now, the code for parallel execution is:

      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(computePipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatch(firstMatrix[0] /* x */, secondMatrix[1] /* y */);
      passEncoder.endPass();

In other words, the number of inweights per node is assumed constant (though it can be 0 for 'no weight').

We can (in distant future...) make networks with large numbers of nodes with very small numbers of weights more 
efficient by running the `netsubcycle()` twice: once for the densely connected nodes and once for the sparse ones.

What we should probably do for now, is to allow varying inweights, set nonexisting ones to 0 before copying to GPU and
then copying only the ones that exist back. Or perhaps this is unnecessary and should we simply say: 0 === nonexistent.

Or... we could use the following scheme. We have a sparse JS array where the rows are say Float32Arrays (or other type):

      weights = [
         [0.1,.2,.3,.01],
         [-.5,-.4,-.2],
         [0.2,.5,.6,.8,0.1,0.3]
      ]          

      indices [
         [0,11,2,4],
         [3,9,6],
         [7,8,5,10,1,12]
      ]

The access of the weights looks like:

        ivec2 resultCell = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
      
        float result = 0.0;
      
        for (int i = 0; i < inacts.size.y; i++) {
          int a = i + resultCell.x * int(inacts.size.y);
          int b = resultCell.y + i * int(weights.size.y);
          result += inacts.numbers[a] * weights.numbers[b];
        }

This means we still have calculate the location and length of the weights anyway. To enable us to do this, we need to have
two additional arrays:

 - insize: number of incoming weights per node, which in this case would be: [4,3,6]

and

 - indexstart: index of the first innode-index in the index array, which would be [0,4,7].

So, then we know that the second set of innodes, to node 1 (0-based index), starts at 4 and is 3 nodes long. Any connection
variables are encoded at these indices, such as index of innode and inweight (called weight here).

We shall use this general scheme, which allows optimal use of the memory and speed. For the dispatch, we shall use
for size.x the number of nodes and for size.y the largest number of innodes. 

As proposed above, we can complicate this scheme by splitting nodes with many innodes, but this will remain an excercise for
the future. This action is complicated by the fact that we need to unify net input before applying act and weight updates.


### Bi and Poo (1998) learning rule: spike timing plasticity using only in-node spike times

When a neuron A spikes and its spike arrives at another neuron B's dendrite, then if B has spiked recently the weight
w_AB (from A to B) will *decrease*. This is easiest enfored by checking the outweights, but storing these would require
having both an array in in-node indices and out-node indices per node, greatly increasing the total storage space. Can
we do it without this and without losing efficiency (assuming 1000+ GPU ALUs)?

Whenever we update a weight, looking from a node A back to its in-nodes, we have two situations:

 1. A has spiked now: Strengthen all incoming connections.

 2. A has not spiked now but recently (e.g., 8 ms ago). If we have an in-node B that has spiked now, we weaken the
    the incoming connection, unless it was too long ago.

This would work. A connection is only weakened if an in-node B has spiked now (i.e., at this timestamp, which is in ms) and
the current node B has not spiked but has spiked recently enough for the weakening to be significant (e.g., less then 30 
or 50 ms).

In-nodes that have spike not-now but earlier are disregarded because they have been dealt with in an earlier cycle.


### Matrices everywhere

It is tempting to alter the code to use both vectors like [1,2] and matrices like [[1,2],[3,4]] rather than just matrices,
but this will probably get us almost no speed-up and it will cause more complex code. 

In general, will encode activations as [[1,2]]-type matrices and weights as sparse [[1,2],[3,4,5,6]]-type matrices where each
row k represents the in-parameters to node k, but using 0-based indices everywhere. Each row can have different elements,
including none at all.


### Do we need to store net input in a buffer? And do we need separate old/new activations buffers?

There are very few cases where we need access to net input. We therefore do not need to store it. In the cycle(), we can 
first calculate net, calculate act immediately on the basis if net, then have a memoryBarrier() and a barrier() to make sure
all acts have been updated. Then, we can update the weights on the basis of the new activations.

If we update the activations, we will have spikes on some nodes. We can then update the weights. There does not seem to be
a case for old and new activations here, as the net input buffers the activations. That is, we will not update some activations
on the basis old and others on the basis of new activations.

This also eliminates the need for a swap buffer or ping-pong approach.


### Recording spikes and such

We can at any point call readGPUBuffer('act') to get the spikes, but only of the last cycle. We should set up an
internal recording facility to store spikes so we can generate spike plots. Spikes are recorded per node per timestamp. We 
could set up a buffer of say n by b, where n is the number of nodes, and b the spike buffer size, e.g., 200. This would 
give us at least 1 s of recording time. 

We should also be able to turn spike recording on or off, because recording make possibly take a lot of resources. 
If recording is turned on, we will probably want to read out the result fairly often to update plots and such, e.g., every
50 or 100 ms. 

If the spikes are not necessary for detailed (statistical) analysis, we can possibly write the spikes directly to canvas,
so that we do not have to buffer them in the GPU. This is probably very efficient and lightens the processing and memory
loads.


### Error-backprogation and deep learning schemes

Using the sparse connection scheme, we should be able to implement some of the less demanding deep learning architectures,
which may be interesting for certain applications, even though this is not the main purpose of this project. There are a few
aspects to keep in mind here.

#### Layer ordering

If we implement deep learning architectures, we should probably store the activations in (partially) ordered layers: 
act_1, act_2, ..., act_L, where each layer *must* be updated before the next. Each layer could be divided in sublayers
but we will ignore this completely as it does not affect parallelization. We could define separate bind groups and
arrays and have the updates in order with barriers between them. But how would this work? Because a given element E only
knows its x-value on the ALU. We can do multiple dispatches etc. but this will probably be very inefficient. 

Given that we have index E, and given layer l (lower-case L) with n_l nodes (E < n_l), we can simply update node E iff 
E < n_l and else do nothing. Layers are unfolded explicitly in code with barriers between them, which guarantee that earlier
layers in the ordering have been updated. In each case, we calculate net and then immediately act variables.

This describes the feedforward sweep. 

#### Errorback sweep

Here, we first calculate errors Delta at the upper layer L, then update the weights to layer L. Then we repeat this for
Layer L-1, and so forth. Again, we unfold this explicity in the code, which will become verbose but this is not a problem
as nobody will read it.

So, we have Forward update-act-and-net layer 1 to layer L. Then, Backward update-delta-and-weight: layer L to layer 1.

#### Weights

In convolutional networks, weights are organized in kernels etc. We can mimic the same organization. There is no need to
adhere to the sparse weight approach, though we may give the organization of weights in general some thought.

Compared to the more biologically realistic networks, this is relatively straightforward.

#### Patterns

So far, we have ignored input patterns. In reality, however, each cycle is repeated for each pattern (covering both inputs
and potential outputs). These must be stored in a buffer and applied. If their number is large, this will affect the total
size of the network. E.g., a network with a million input nodes and 10,000 patterns will take 10 GB memory, leaving virtually
no space on a GFX 1080 Ti graphics card (with 11 GB). 

This is not really a problem, but rather a limitation.


# Situation 24 Feb 2020

In the past few days I wrote the Centi Testing Framework to allow unit tests of the Walnut 2.0. This works but needs to 
undergo field testing. 

I was about to implement the activation rule in nngpu.js around line 87 and then the learning rule. 


# Parallelization over Internet revisited

We could do a learning neural network where each neuron stores its own inweights. That means that only the spikes need to
be exchanged. Suppose, we aim for a iteration rate of 10 Hz. This means we have about 25 ms to send the spikes to the server
and the server than has another 25 ms to send it to other clients (redistribute). We then have 50 ms calculation time.

If we have 100,000 neurons and 10,000 spikes, this gives us 10kB data to send over. 

Note that we could use the following scheme:

byte 0: send address of first spike in neurons 0-254 or send 0xff if no spikes in 0-254
  if spike at neuron n1, n is n1, else n = 255
byte 1: send address of next spike in neurons n1-n1+254 or send 0xff if none
  etc.

In this way we can cover all spikes with rather few bytes. In case of 10% spiking neurons out of 100,000, we would need about
10,000 bytes.

With 1,000,000 neurons we would have 1MB,
still doable, though it would keep the server very busy to process and redistribute the spikes. With several thousands clients,
we could then approach the size of the human brain. 