To run the example sampling code

`python3 lsst_y1_3x2pt_sampling.py configs/nn_emu.yaml 4 0`

The command line arguments are: 
- The config file 
- The iteration of emulation. Here we are using the pre-trained emulator wiht the 4th iteration
- Whether we are using tempering or not.

### Config File

You can set the following with the config file:
- `data`: Change the data vector, covariance and the scale cut mask with this option
- `sampling`: Number of walkers, MCMC steps, etc. 
