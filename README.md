# [Neanderthal ATLAS](http://neanderthal.uncomplicate.org) - Low-level Java (JNI) bindings for [ATLAS](httl://http://math-atlas.sourceforge.net/) BLAS and LAPACK.

Neanderthal ATLAS is a low-level Java library for calling native BLAS and LAPACK procedures provided by ATLAS library. Used by Clojure Neanderthal matrix and linear algebra library.

There is a lot of documentation and tutorials at [Neanderthal Website](http://neanderthal.uncomplicate.org). Please make sure to check it out.

## Project Goals

* Enable a low-overhead calling of fast, machine optimized BLAS and LAPACK procedures for fast matrix and linear algebra computations.
* Provide a stable base for higher-level Java matrix libraries.
* Seamless and automatic dependency - no manual work for setting up the library is needed (*ATLAS must be present on the system i.e. installed independently*)
* Offer fast native performance with no overhead on top of native libraries.

## Installation & Requirements

For these bindings to work, you need ATLAS on your system. Although ATLAS might be distributed as a pre-compiled binary by your system (for example, on  Ubuntu or Debian Linux you might install it through apt-get) **the only way to get full performance is to let ATLAS build script configure itself for your machine and install from the source**, as recommended in the official documentation. The build procedure is automatic, you just have to strictly follow the [**OFFICIAL INSTALLATION GUIDE**](http://math-atlas.sourceforge.net/atlas_install/). The difference in performance might even be an order of magnitude.

## Documentation & Examples

See the documentation of Clojure-s [Neanderthal library](http://neanderthal.uncomplicate.org)

## Project Maturity

BLAS and ATLAS have a decades-long history, so Neanderthal ATLAS bindings should be stable.
However, the Neanderthal library is still in development, so I expect to add additional bindings, fix bugs, etc.
While the project is in the 0.X.Y version it is considered in development, so the priority is adding new features and enhancing the existing code as much as possible, rather than backward compatibility.
Once it reaches version 1.0.0 it will be considered stable, and more consideration will be directed towards supporting backward compatibility.

## License

Copyright © 2014-2016 Dragan Djuric

Distributed under the Eclipse Public License either version 1.0 or (at your option) any later version.
