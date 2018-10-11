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

    // ========================= Mathematical functions ========================================

    public static native void vsSqr (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdSqr (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsMul (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                     Buffer Y, int offsetY);
    public static native void vdMul (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                     Buffer Y, int offsetY);

    public static native void vsDiv (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                     Buffer Y, int offsetY);
    public static native void vdDiv (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                     Buffer Y, int offsetY);

    public static native void vsInv (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdInv (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsAbs (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdAbs (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsLinearFrac (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                            float scaleA, float shiftA, float scaleB, float shiftB,
                                            Buffer Y, int offsetY);
    public static native void vdLinearFrac (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                            double scaleA, double shiftA, double scaleB, double shiftB,
                                            Buffer Y, int offsetY);

    public static native void vsFmod (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                      Buffer Y, int offsetY);
    public static native void vdFmod (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                      Buffer Y, int offsetY);

    public static native void vsRemainder (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                           Buffer Y, int offsetY);
    public static native void vdRemainder (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                           Buffer Y, int offsetY);

    public static native void vsSqrt (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdSqrt (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsInvSqrt (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdInvSqrt (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsCbrt (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdCbrt (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsInvCbrt (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdInvCbrt (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsPow2o3 (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdPow2o3 (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsPow3o2 (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdPow3o2 (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsPow (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                     Buffer Y, int offsetY);
    public static native void vdPow (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                     Buffer Y, int offsetY);

    public static native void vsPowx (int N, Buffer A, int offsetA, float b, Buffer Y, int offsetY);
    public static native void vdPowx (int N, Buffer A, int offsetA, double b, Buffer Y, int offsetY);

    public static native void vsHypot (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                       Buffer Y, int offsetY);
    public static native void vdHypot (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                       Buffer Y, int offsetY);

    public static native void vsExp (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdExp (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsExpm1 (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdExpm1 (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsLn (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdLn (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsLog10 (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdLog10 (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsCos (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdCos (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsSin (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdSin (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsSinCos (int N, Buffer A, int offsetA, Buffer Y, int offsetY,
                                        Buffer Z, int offsetZ);
    public static native void vdSinCos (int N, Buffer A, int offsetA, Buffer Y, int offsetY,
                                        Buffer Z, int offsetZ);

    public static native void vsTan (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdTan (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsAcos (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdAcos (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsAsin (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdAsin (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsAtan (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdAtan (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsAtan2 (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                       Buffer Y, int offsetY);
    public static native void vdAtan2 (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                       Buffer Y, int offsetY);

    public static native void vsCosh (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdCosh (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsSinh (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdSinh (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsTanh (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdTanh (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsAcosh (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdAcosh (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsAsinh (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdAsinh (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsAtanh (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdAtanh (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsErf (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdErf (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsErfc (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdErfc (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsErfInv (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdErfInv (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsErfcInv (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdErfcInv (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsCdfNorm (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdCdfNorm (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsCdfNormInv (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdCdfNormInv (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsGamma (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdGamma (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsLGamma (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdLGamma (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsExpInt1 (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdExpInt1 (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsFloor (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdFloor (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsCeil (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdCeil (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsTrunc (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdTrunc (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsRound (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdRound (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsModf (int N, Buffer A, int offsetA, Buffer Y, int offsetY,
                                      Buffer Z, int offsetZ);
    public static native void vdModf (int N, Buffer A, int offsetA, Buffer Y, int offsetY,
                                      Buffer Z, int offsetZ);

    public static native void vsFrac (int N, Buffer A, int offsetA, Buffer Y, int offsetY);
    public static native void vdFrac (int N, Buffer A, int offsetA, Buffer Y, int offsetY);

    public static native void vsFmax (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                      Buffer Y, int offsetY);
    public static native void vdFmax (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                      Buffer Y, int offsetY);

    public static native void vsFmin (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                      Buffer Y, int offsetY);
    public static native void vdFmin (int N, Buffer A, int offsetA, Buffer B, int offsetB,
                                      Buffer Y, int offsetY);

    // ========================= RNG functions ========================================

    public static native int vslNewStreamARS5 (int seed, Buffer stream);
    public static native int vslDeleteStream (Buffer stream);

    public static native int vsRngGaussian (Buffer stream, int n, Buffer res, Buffer params);
    public static native int vdRngGaussian (Buffer stream, int n, Buffer res, Buffer params);

    public static native int vsRngUniform (Buffer stream, int n, Buffer res, Buffer params);
    public static native int vdRngUniform (Buffer stream, int n, Buffer res, Buffer params);

}
