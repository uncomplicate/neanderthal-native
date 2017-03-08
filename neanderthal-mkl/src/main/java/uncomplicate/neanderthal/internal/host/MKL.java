//   Copyright (c) Dragan Djuric. All rights reserved.
//   The use and distribution terms for this software are covered by the
//   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
//   which can be found in the file LICENSE at the root of this distribution.
//   By using this software in any fashion, you are agreeing to be bound by
//   the terms of this license.
//   You must not remove this notice, or any other, from this software.

package uncomplicate.neanderthal.internal.host;

import java.nio.Buffer;

public class MKL {

    static {
        NarSystem.loadLibrary();
    }


    /* ======================================================
     *
     * MKL Functions
     *
     * =====================================================
     */

    // --------- set ---------------------------------------

    public static native void sset(int N, float alpha, Buffer X, int offsetX, int incX);

    public static native void dset(int N, double alpha, Buffer X, int offsetX, int incX);

    // ---------- axpby ------------------------------------

    public static native void saxpby(int N,
                                     float alpha, Buffer X, int offsetX, int incX,
                                     float beta, Buffer Y, int offsetY, int incY);

    public static native void daxpby(int N,
                                     double alpha, Buffer X, int offsetX, int incX,
                                     double beta, Buffer Y, int offsetY, int incY);

    public static native void somatadd(int ordering, int transa, int transb, int M, int N,
                                       float alpha, Buffer A, int offsetA, int lda,
                                       float beta, Buffer B, int offsetB, int ldb,
                                       Buffer C, int offsetC, int ldc);

    public static native void domatadd(int ordering, int transa, int transb, int M, int N,
                                       double alpha, Buffer A, int offsetA, int lda,
                                       double beta, Buffer B, int offsetB, int ldb,
                                       Buffer C, int offsetC, int ldc);

    public static native void somatcopy(int ordering, int trans, int M, int N,
                                        float alpha, Buffer A, int offsetA, int lda,
                                        Buffer B, int offsetB, int ldb);

    public static native void domatcopy(int ordering, int trans, int M, int N,
                                        double alpha, Buffer A, int offsetA, int lda,
                                        Buffer B, int offsetB, int ldb);

    public static native void simatcopy(int ordering, int trans, int M, int N,
                                        float alpha, Buffer AB, int offsetAB, int lda, int ldb);

    public static native void dimatcopy(int ordering, int trans, int M, int N,
                                        double alpha, Buffer AB, int offsetAB, int lda, int ldb);

}
