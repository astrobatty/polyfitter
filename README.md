[![Image](https://img.shields.io/badge/tutorials-%E2%9C%93-blue.svg)](https://github.com/astrobatty/polyfitter/tree/master/examples)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/astrobatty/polyfitter/blob/master/examples/run_polyfit.ipynb)
[![DOI](https://zenodo.org/badge/359579409.svg)](https://zenodo.org/badge/latestdoi/359579409)
<!--- [![Image](https://img.shields.io/badge/arXiv-1909.00446-blue.svg)](https://arxiv.org/abs/1909.00446) -->

# polyfitter - Polynomial chain fitting and classification based on light curve morphology for binary stars

The mathematical background of polynomial chain fitting (polyfit) is published in [Prša et al.,2008,ApJ,687,542](https://ui.adsabs.harvard.edu/abs/2008ApJ...687..542P/abstract).
The paper that related to this code can be found [here](https://ui.adsabs.harvard.edu/abs/maybeoneday).
This code is built upon the original [polyfit](https://github.com/aprsa/polyfit).

## Installation

This code depends on the GNU Scientific Library [(GSL)](https://www.gnu.org/software/gsl/) version>=2.6, which is available from [here](http://www.linuxfromscratch.org/blfs/view/svn/general/gsl.html). Before installation, make sure it is properly installed, e.g. running the followings, which should return the location of the library and the installed version of GSL:
```bash
pkg-config --cflags --libs gsl && gsl-config --version
```


To install the package clone this git and go the directory:
```bash
git clone https://github.com/astrobatty/polyfitter.git

cd ./polyfitter
```

As `polyfitter` is dependent on certain python package versions, the easiest way to install it is through creating a conda environment:
```bash
conda create -n polyfit python=3.7 scikit-learn=0.23.2 cython=0.29.20

conda activate polyfit

python setup.py install
```

If the code is not used, the environment can be deactivated:
```bash
conda deactivate
```

If you want to access this environment from jupyter you can do the followings:
```bash
conda install -c anaconda ipykernel

python -m ipykernel install --user --name=polyfit
```

Then after restarting your jupyter you'll be able to select this kernel.

## Example interactive usage

To fit a polynomial chain to the light curve of OGLE-BLG-ECL-040474:
```python
from polyfitter import Polyfitter

import numpy as np

# Parameters from OGLE database
ID = 'OGLE-BLG-ECL-040474'
P  = 1.8995918
t0 = 7000.90650

# Load light curve from OGLE database
# This is in magnitude scale
path_to_ogle = 'http://ogledb.astrouw.edu.pl/~ogle/OCVS/data/I/'+ID[-2:]+'/'+ID+'.dat'
lc = np.loadtxt(path_to_ogle).T

# For clarity
time = lc[0]
mag  = lc[1]
err  = lc[2]

# Create Polyfitter instance by setting the brightness scale of your data
# Set scale to "mag" or "flux"
pf = Polyfitter(scale='mag')

# Run polynomial chain fitting
t0new, phase, polyfit, messages = pf.get_polyfit(time,mag,err,P,t0)
```

Plotting our results gives:
```python
import matplotlib.pyplot as plt

plt.errorbar((time-t0new)/P%1,mag,err,fmt='k.')
plt.errorbar((time-t0new)/P%1-1,mag,err,fmt='k.')
plt.plot(phase,polyfit,c='r',zorder=10)
plt.plot(phase+1,polyfit,c='r',zorder=10)
plt.xlabel('Phase')
plt.ylabel('Magnitude')
plt.xlim(-0.5,1)
plt.gca().invert_yaxis()
plt.show()
```
![example fit](https://raw.githubusercontent.com/astrobatty/polyfitter/master/docs/OGLE-BLG-ECL-040474.jpg)

And the morphology classification:
```python
morp_array = pf.c
print('Morphology type =' , morp_array[0] )
```

You can find a Google Colab friendly tutorial [in the examples](https://github.com/astrobatty/polyfitter/tree/master/examples/run_polyfit.ipynb).

## Running as python script

The difference from using the code interactively is that you have to put your code under a main function. See [here](https://github.com/astrobatty/polyfitter/tree/master/examples/run_polyfit_as_script.py) how to do it.

## Available options
- Polyfitter class instance:
  - `scale` The scale of the input data that will be used with this instance. Must be "mag" or "flux".
  - `debug` If `True` each fit will be displayed with auxiliary messages.

- Getting polyfit:
  - `verbose` If `0` the fits will be done silently. Default is `1`.
  - `vertices` Number of equidistant vertices in the computed fit. Default is `1000`, which is mandatory to run the classification afterwards.
  - `maxiters` Maximum number of iterations. Default is `4000`.
  - `timeout` The time in seconds after a fit will be terminated. Default is `100`.

## Contributing
Feel free to open PR / Issue.

## Citing
If you find this code useful, please cite [X](https://ui.adsabs.harvard.edu/abs/maybeoneday), and [Prša et al.,2008,ApJ,687,542](https://ui.adsabs.harvard.edu/abs/2008ApJ...687..542P/abstract). Here are the BibTeX sources:
```
@ARTICLE{2008ApJ...687..542P,
       author = {{Pr{\v{s}}a}, A. and {Guinan}, E.~F. and {Devinney}, E.~J. and {DeGeorge}, M. and {Bradstreet}, D.~H. and {Giammarco}, J.~M. and {Alcock}, C.~R. and {Engle}, S.~G.},
        title = "{Artificial Intelligence Approach to the Determination of Physical Properties of Eclipsing Binaries. I. The EBAI Project}",
      journal = {\apj},
     keywords = {methods: data analysis, methods: numerical, binaries: eclipsing, stars: fundamental parameters, Astrophysics},
         year = 2008,
        month = nov,
       volume = {687},
       number = {1},
        pages = {542-565},
          doi = {10.1086/591783},
archivePrefix = {arXiv},
       eprint = {0807.1724},
 primaryClass = {astro-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2008ApJ...687..542P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Acknowledgements
This project was made possible by the funding provided by the Lendület Program of the Hungarian Academy of Sciences, project No. LP2018-7.
