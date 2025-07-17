This is how the AI works essentially:

Context on the weights:
    The weights affect how much impact an input has on a nueron

1) There is a feedforward neural network with:
    - 5 inputs (bird y pos, bird y velocity, dist to next pipe, the top of the next pipe, and the bottom of the next pipe)
    -----------------------------------------------------------
    - 1 hidden layer with 6 neurons
    The hidden layer is essentially a layer of neurons between the input and output, for example:

    (xn is the input num and hn is the neurons)
    x1->(h1 to h6)
    x2->(h1 to h6)
    ..
    x5->(h1 to h6)
    then
    h1->output
    h2->output
    ..
    h6->output

    In each these hidden nuerons, it computes using ReLU something like:
        h1 = ReLU(w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + b)
    The hidden nuerons are essentially determining the patterns like "Am I diving into a pipe" etc.

    Then the final output neuron blends these to see if it should flap or not
    -----------------------------------------------------------
    - 1 output (probability to flap)

2) Then we normalzie the input values to a consistent range (to avoid overflow)

3) We feed these inputs into the neural network

4) It computes the probability to flap

5) If the probability is greater than 50%, the bird flaps

