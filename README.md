<h1 align="center">Denoising AutoEncoder</h1>

<hr>

<h2>Project Overview</h2>

<p><strong>Language Used:</strong> Python</p>
<p><strong>Libraries Used:</strong> PyTorch (torch), TorchVision (torchvision), Matplotlib (matplotlib)</p>
<p><strong>Dataset Used:</strong> Fashion-MNIST (contains 60,000 grayscale images)</p>

<hr>

<h2>Workflow</h2>

<h3>1️⃣ Defining the Model Class</h3>

<p>
The <strong>DenoisingAutoencoder</strong> class inherits from <code>nn.Module</code>, the base class for neural network models in PyTorch.
It provides built-in methods such as <code>train()</code> and <code>eval()</code>.
</p>

<p>
The <code>nn.Sequential()</code> container allows chaining multiple layers where the output of one layer becomes the input to the next.
Instead of aggressive compression, the 28 × 28 Fashion-MNIST image undergoes <strong>gradual compression</strong>, making learning smoother and more stable.
</p>

<ul>
  <li><strong>Encoder:</strong> Transforms high-dimensional input into a lower-dimensional bottleneck representation.</li>
  <li><strong>Decoder:</strong> Reconstructs the compressed representation back to original dimensions.</li>
</ul>

<p>
<strong>Activation Functions Used:</strong>
</p>

<ul>
  <li><strong>ReLU:</strong> Applies <code>max(0, x)</code> to introduce sparsity and improve feature learning efficiency.</li>
  <li><strong>Tanh:</strong> Maps output values to range [-1, 1], ensuring zero-centered symmetric distribution.</li>
</ul>

<p>
The <code>forward()</code> function:
</p>

<ul>
  <li>Flattens 2D images into 1D vectors using <code>view()</code></li>
  <li>Passes input → encoder → activation → decoder</li>
  <li>Reshapes output back for comparison with original image</li>
</ul>

<p>
Batch size is automatically inferred using <code>-1</code> (batch size = 128).
</p>

<hr>

<h3>2️⃣ Loading Data for Training & Testing</h3>

<ul>
  <li>Separate datasets ensure proper generalization.</li>
  <li><code>ToTensor()</code> scales pixel values from [0,255] to [0,1].</li>
  <li><code>Normalize(0.5, 0.5)</code> shifts range to [-1,1] to match Tanh output.</li>
  <li><code>DataLoader</code> enables efficient batch processing.</li>
  <li>Training data is shuffled to prevent order-based overfitting.</li>
</ul>

<p>
Normalization formula:
<br>
<code>(pixel − mean) / std</code>
</p>

<hr>

<h3>3️⃣ Loss Function & Optimization</h3>

<p><strong>Loss Function:</strong> Mean Squared Error (MSE)</p>
<p>
Measures average squared difference between reconstructed and original images.
</p>

<p><strong>Optimizer:</strong> Adam</p>

<ul>
  <li><strong>Learning Rate (lr):</strong> Controls update magnitude.</li>
  <li><strong>Epochs:</strong> One full dataset pass.</li>
</ul>

<p>
Too high learning rate → Overshooting minima <br>
Too low learning rate → Slow convergence <br>
Too many epochs → Overfitting <br>
Too few epochs → Underfitting
</p>

<hr>

<h3>4️⃣ Training Loop</h3>

<ul>
  <li><code>model.train()</code> enables training mode.</li>
  <li>Random noise tensor added to input image.</li>
  <li>Noise scaled by factor <strong>0.3</strong> (optimal balance).</li>
  <li><code>clamp()</code> ensures noisy input remains within [-1,1].</li>
  <li><code>zero_grad()</code> resets gradients.</li>
  <li>Forward pass → Loss calculation → Backpropagation.</li>
  <li><code>optimizer.step()</code> updates weights.</li>
</ul>

<p>
This is an example of <strong>self-supervised learning</strong>:
The model receives a noisy image as input and learns to reconstruct the clean image.
</p>

<p>
For faster computation, <code>torch.device()</code> can be used to shift training to GPU if available.
</p>

<hr>

<h3>5️⃣ Testing Loop</h3>

<ul>
  <li><code>model.eval()</code> disables training-specific behavior.</li>
  <li><code>torch.no_grad()</code> prevents gradient computation.</li>
  <li>Noise added to test image.</li>
  <li>Model outputs denoised image.</li>
  <li><code>unsqueeze()</code> adds batch dimension.</li>
</ul>

<hr>

<h3>6️⃣ Displaying Images</h3>

<ul>
  <li><code>figure()</code> sets display size.</li>
  <li><code>subplot()</code> arranges original, noisy, and denoised images.</li>
  <li><code>squeeze()</code> removes extra batch/channel dimensions.</li>
  <li><code>imshow()</code> displays 2D grayscale image.</li>
  <li><code>axis("off")</code> removes axes for clean visualization.</li>
</ul>

<hr>

<h2>Applications</h2>

<ul>
  <li>Medical Imaging Enhancement</li>
  <li>Voice Recognition Systems</li>
  <li>Audio Communication Improvement</li>
  <li>Audio Restoration</li>
  <li>Surveillance Camera Noise Reduction</li>
</ul>
