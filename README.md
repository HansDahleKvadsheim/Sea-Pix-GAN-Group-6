# Sea-Pix-GAN reproduction (Group 6)
This repository is a reproduction of the model designed in the paper titled "Sea-Pix-GAN: Underwater image enhancement using adversarial neural network", the paper itself can be found on [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1047320323002717). We have included the Replication_Blog.pdf that documenting our process.  
The aquatic environment of underwater imaging significantly alters the produced images producing hazy and blue-green tinted results.
Many state-of-the-art solutions to underwater image enhancement exist, though their performance varies for the different types of image scenarios.
In their work, they propose a model that aims to solve this problem using image-to-image translation in a Generative Adversarial Network (GAN) structure.
Their model is trained to apply color, content, and style transfer on the [EUVP](https://irvlab.cs.umn.edu/resources/euvp-dataset) underwater images dataset.
Finally, they evaluate their model against a barrage of existing solutions with the [PSNR](https://www.sciencedirect.com/topics/engineering/signal-to-noise-ratio), [SSIM](https://www.sciencedirect.com/topics/engineering/similarity-index), and [UIQM](https://www.sciencedirect.com/topics/computer-science/quality-measure) metrics.

## Installation
To get a local copy up and running follow these steps:

1. **Clone the repository**:\
    Open a terminal and navigate to the location that you want to clone the project in then run the following command:
    ```bash
    git clone https://github.com/HansDahleKvadsheim/Sea-Pix-GAN-Group-6.git
    ```
   
2. **Install Python**\
    The code is written in Python, differing versions were used by our team.
    Python can be downloaded [here](https://www.python.org/downloads/).

3. **Install Conda**\
    To train Deep Neural Networks the code uses several libraries such as PyTorch and CUDA.
    Installing these libraries is easies using [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

4. **Opening interactive python notebooks (ipynb)**\
    All the code is in this singular interactive python notebook file `sea-pix-GAN.ipynb`, to open such a file you need a special editor.
    The following options were used during development, [Jupyter](https://jupyter.org/), [Google Colab](https://colab.google/), and [VS Code](https://code.visualstudio.com/).

5. **Download the dataset**\
    To run the code you will need to download the EUVP dataset [here](https://drive.google.com/drive/folders/1ZEql33CajGfHHzPe1vFxUFCMcP0YbZb3), importantly you only need to download the `Paired` folder.
    After you download the dataset you will need to manually merge the three subfolders, during development we merged the dataset into one folder with subfolders `input` and `ground_truth` from the original `trainA` and `trainB` folders respectively.

## Running
To run our reproduction first open the notebook file `sea-pix-GAN.ipynb` in your editor of choice.
Then before running, you need to define the path to where you stored the dataset in the `Running dataset construction code` code cell.
After this change, you can safely run all cells to import the packages (if they don't work insert a cell above and import them using `pip` if installed).
Running all the cells will handle everything itself, there is some extra information and special deviations which we describe below.

### Overwriting CPU vs GPU
By default, the code will check whether there is a GPU available and if so use that to run the training as it is faster.
When limited GPU memory is available this approach will crash, for this reason, a manual override cell is provided at the top.
Uncomment the `# device = th.device("cpu")` line and run that cell and your device will be set to the CPU, note that this is significantly slower than the GPU.

### Test runs
To quickly test the code on a small subset of the data we provide an option in the hyperparameters that limits the dataset to 192 images.
Simply set the `"test_run": False,` to `True` to activate this test run setup, additionally, you can change the test run size defined in `"test_run_size": 192,`.

### Model data storage and run continues
Training a model generally takes several hours, to split the workload we store all important data for a model so that a run can be stopped and continued later.
Near the hyperparameters the model data directory is defined, **before training each new model change this path!**
In this directory, the generator and discriminator model weights are stored as well as the dataset partitions and the loss statistics.
The data is stored automatically when running all the cells, and similarly, if there is data already stored it will load it in automatically overriding the current dataset, loss statistics, and model initializations.

## Authors
- Henrik Brinch van Meenen
- Thijs Penning
- Hans Kvadsheim
- Yuanfu Pan
