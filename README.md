# TFHE-py

Python implementation of the [Fully Homomorphic Encryption (FHE)](https://en.wikipedia.org/wiki/Homomorphic_encryption#Fully_homomorphic_encryption) scheme [TFHE: Fast Fully Homomorphic Encryption over the Torus](https://eprint.iacr.org/2018/421.pdf).

You can see example usages in the [Jupyter Notebook](./main.ipynb) and the [tests](./tests/) directory.

The starting point of this implementation was the code written by [NuCypher](https://www.nucypher.com) for their [NuFHE](https://github.com/nucypher/nufhe) library. More specifically the code in commit [17f3b62](https://github.com/nucypher/nufhe/commit/17f3b6200425a42b84ff844928550e9add684280) was used which itself seems to be a port of their Julia version [nucypher/TFHE.jl](https://github.com/nucypher/TFHE.jl) which looks like a port (see [this commit](https://github.com/nucypher/TFHE.jl/commit/bf33742310a369c6da133593cbbefd75374bbefb)) of the original [tfhe/tfhe](https://github.com/tfhe/tfhe) codebase that's written in C / C++.

## Setup

1. `git clone <url>`
2. `asdf install`
3. `pipenv install -e .`
4. `pipenv shell`
5. `python tests/e2e_half_adder_test.py`
6. `pipenv run test`
7. `pipenv run coverage`
8. `pipenv scripts`

_Optional_: Update the properties marked with `TODO:` in the [`.vscode/settings.json`](./.vscode/settings.json) file. To get the correct paths run `which <package>` within a Pipenv shell.

## Useful Commands

```sh
asdf install

pipenv install -e .
pipenv install [-d] <name>[~=<version>]
pipenv shell
pipenv scripts
pipenv run <command>

python <path>

flake8 <path>

pylint <path> --recursive true

mypy <path>

pytest [-s] [-v] [-k <pattern>] [<path>]

coverage html
coverage report -m

py-spy record -o profile.svg --pid <pid>
py-spy record -o profile.svg -- python <path>
py-spy top -- python <path>
```

## Useful Resources

### (T)FHE

- [tfhe/tfhe](https://github.com/tfhe/tfhe)
- [nucypher/nufhe](https://github.com/nucypher/nufhe)
- [zama-ai/tfhe-rs](https://github.com/zama-ai/tfhe-rs)
- [nucypher/TFHE.jl](https://github.com/nucypher/TFHE.jl)
- [thedonutfactory/go-tfhe](https://github.com/thedonutfactory/go-tfhe)
- [thedonutfactory/rs_tfhe](https://github.com/thedonutfactory/rs_tfhe)
- [virtualsecureplatform/pyFHE](https://github.com/virtualsecureplatform/pyFHE)
- [openfheorg/openfhe-development](https://github.com/openfheorg/openfhe-development)
- [TFHE: Fast Fully Homomorphic Encryption over the Torus](https://eprint.iacr.org/2018/421)
- [Guide to Fully Homomorphic Encryption over the [Discretized] Torus](https://eprint.iacr.org/2021/1402)
- [SoK: Fully Homomorphic Encryption over the [Discretized] Torus](https://tches.iacr.org/index.php/TCHES/article/view/9836)
- [TFHE Deep Dive - Part I - Ciphertext types](https://www.zama.ai/post/tfhe-deep-dive-part-1)
- [TFHE Deep Dive - Part II - Encodings and linear leveled operations](https://www.zama.ai/post/tfhe-deep-dive-part-2)
- [TFHE Deep Dive - Part III - Key switching and leveled multiplications](https://www.zama.ai/post/tfhe-deep-dive-part-3)
- [TFHE Deep Dive - Part IV - Programmable Bootstrapping](https://www.zama.ai/post/tfhe-deep-dive-part-4)
- [Introduction to practical FHE and the TFHE scheme - Ilaria Chillotti, Simons Institute 2020](https://www.youtube.com/watch?v=FFox2S4uqEo)
- [TFHE Deep Dive - Ilaria Chillotti, FHE.org](https://www.youtube.com/watch?v=LZuEr4jpyUw)
- [003 TFHE Deep Dive (by Ilaria Chillotti)](https://www.youtube.com/watch?v=npoHSR6-oRw)
- [Part 1 Introduction to FHE and the TFHE scheme - Ilaria Chillotti, ICMS](https://www.youtube.com/watch?v=e_76kZ9j2-M)
- [Part 2 Introduction to FHE and the TFHE Scheme - Ilaria Chillotti, ICMS](https://www.youtube.com/watch?v=o7_WNbVuZqQ)
- [Introduction to FHE (Fully Homomorphic Encryption) - Pascal Paillier, FHE.org Meetup](https://www.youtube.com/watch?v=aruz58RarVA)

### Python

- [Real Python](https://realpython.com)
- [Python Cheatsheet](https://www.pythoncheatsheet.org)
- [Learn X in Y minutes](https://learnxinyminutes.com/docs/python)
- [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python)
- [gto76/python-cheatsheet](https://github.com/gto76/python-cheatsheet)
- [Writing Python like it's Rust](https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html)
