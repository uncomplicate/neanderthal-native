# [Neanderthal ATLAS](http://neanderthal.uncomplicate.org) - Low-level Java (JNI) bindings for [ATLAS](httl://http://math-atlas.sourceforge.net/) BLAS and LAPACK.
[![Build Status](https://secure.travis-ci.org/uncomplicate/fluokitten.png)](https://travis-ci.org/uncomplicate/neanderthal-atlas)

Neanderthal ATLAS is a low-level Java library for calling native BLAS and LAPACK procedures provided by ATLAS library. Used by Clojure Neanderthal matrix and linear algebra library.

There is a lot of documentation and tutorials at [Neanderthal Website](http://neanderthal.uncomplicate.org). Please make sure to check it out.

## Project Goals

* Enable a low-overhead calling of fast, machine optimized BLAS and LAPACK procedures for fast matrix and linear algebra computations.
* Provide a stable base for higher-level Java matrix libraries.
* Seamless and automatic dependency - no manual work for setting up the library is needed (*ATLAS must be present on the system i.e. installed independently*)
* Offer fast native performance with no overhead on top of native libraries.

## Installation & Requirements

For these bindings to work, you need ATLAS on your system. Although ATLAS might be distributed as a pre-compiled binary by your system (for example, on  Ubuntu or Debian Linux you might install it through apt-get) **the only way to get full performance is to let ATLAS build script configure and install from the source**, as recommended in the official documentation. The build procedure is fullu automatic, you just have to strictly follow the [**OFFICIAL INSTALLATION GUIDE**](http://math-atlas.sourceforge.net/atlas_install/). The difference in performance might even be an order of magnitude.

Add the following dependencies to your `project.clj` file:

```clojure
[uncomplicate/neanderthal-atlas "0.1.0"] ;; Java (JNI) part
[uncomplicate/neanderthal-atlas "0.1.0" :classifier "amd64-Linux-gpp-jni"] ;; native part, depends on your operating system.
```

Fluokitten artifacts are distributed through Clojars, so they will be downloaded by leiningen by default. If you are using other tools for dependency management, you can download Fluokitten from Clojars.org, or build it from the source by running `lein jar`.

## Usage

Import `uncomplicate.neanderthal.CBLAS` and call the appropriate BLAS procedures.

Usually, you would not use it directly, but through a higher-level library such as [Neanderthal](http://neanderthal.uncomplicate.org).

## [Get Involved](http://neanderthal.uncomplicate.org/articles/community.html)

I welcome anyone who is willing to contribute, no mather the level of experience. Here are some ways in which you can help:
* If you are a native English speaker, i would really appreciate if you can help with correcting the English on the Neanderthal site and in the documentation.
* Contribute articles and tutorials.
* Do code review of the Neanderthal code and suggest improvements.
* If you find bugs, report them via [Neanderthal ATLAS issue tracker](https://github.com/uncomplicate/neanderthal-atlas/issues).

## Documentation & Examples

See the documentation of Clojure-s [Neanderthal library](http://neanderthal.uncomplicate.org)

## Project Maturity

BLAS and ATLAS have a decades-long history, so Neanderthal ATLAS bindings should be stable.
However, the Neanderthal library is still in development, so I expect to add additional bindings and change some bugs.
While the project is in the 0.X.Y version it is considered in development, so the priority is adding new features and enhancing the existing code as much as possible, rather than backward compatibility.
Once it reaches version 1.0.0 it will be considered stable, and more consideration will be directed towards supporting backward compatibility.

## License

Copyright © 2015 Dragan Djuric

Distributed under the Eclipse Public License, the same as Clojure
