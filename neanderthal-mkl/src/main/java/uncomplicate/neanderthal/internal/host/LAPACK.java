//   Copyright (c) Dragan Djuric. All rights reserved.
//   The use and distribution terms for this software are covered by the
//   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
//   which can be found in the file LICENSE at the root of this distribution.
//   By using this software in any fashion, you are agreeing to be bound by
//   the terms of this license.
//   You must not remove this notice, or any other, from this software.

package uncomplicate.neanderthal.internal.host;

import java.nio.Buffer;

public class LAPACK {

    static {
        NarSystem.loadLibrary();
    }

    /* ======================================================
     *
     * LAPACK Functions
     *
     * =====================================================
     */

    /*
     * -----------------------------------------------------------------
     * Auxiliary Routines
     * -----------------------------------------------------------------
     */

    public static native float slange (int Order, int norm, int M, int N, Buffer A, int offsetA, int lda);

    public static native double dlange (int Order, int norm, int M, int N, Buffer A, int offsetA, int lda);

    public static native float slansy (int Order, int norm, int uplo, int N, Buffer A, int offsetA, int lda);

    public static native double dlansy (int Order, int norm, int uplo, int N, Buffer A, int offsetA, int lda);

    public static native float slantr (int Order, int norm, int uplo, int diag, int M, int N,
                                       Buffer A, int offsetA, int lda);

    public static native double dlantr (int Order, int norm, int uplo, int diag, int M, int N,
                                        Buffer A, int offsetA, int lda);

    public static native float slangb (int norm, int n, int kl, int ku,
                                       Buffer A, int offsetA, int lda, Buffer work);

    public static native double dlangb (int norm, int n, int kl, int ku,
                                        Buffer A, int offsetA, int lda, Buffer work);

    public static native float slansb (int norm, int uplo, int n, int k,
                                       Buffer A, int offsetA, int lda, Buffer work);

    public static native double dlansb (int norm, int uplo, int n, int k,
                                        Buffer A, int offsetA, int lda, Buffer work);

    public static native float slantb (int norm, int uplo, int diag, int n, int k,
                                       Buffer A, int offsetA, int lda, Buffer work);

    public static native double dlantb (int norm, int uplo, int diag, int n, int k,
                                        Buffer A, int offsetA, int lda, Buffer work);

    public static native float slansp (int norm, int uplo, int n,
                                       Buffer A, int offsetA, Buffer work);

    public static native double dlansp (int norm, int uplo, int n,
                                        Buffer A, int offsetA, Buffer work);

    public static native float slantp (int norm, int uplo, int diag, int n,
                                       Buffer A, int offsetA, Buffer work);

    public static native double dlantp (int norm, int uplo, int diag, int n,
                                        Buffer A, int offsetA, Buffer work);

    public static native float slangt (int norm, int n, Buffer D, int offsetD);

    public static native double dlangt (int norm, int n, Buffer D, int offsetD);

    public static native float slanst (int norm, int n, Buffer D, int offsetD);

    public static native double dlanst (int norm, int n, Buffer D, int offsetD);

    public static native int slacpy (int Order, int uplo, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int dlacpy (int Order, int uplo, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int slaset (int Order, int uplo, int M, int N, float alpha, float beta,
                                     Buffer A, int offsetA, int lda);

    public static native int dlaset (int Order, int uplo, int M, int N, double alpha, double beta,
                                     Buffer A, int offsetA, int lda);

    public static native int slascl (int Order, int type, int k1, int ku, float cfrom, float cto,
                                     int M, int N, Buffer A, int offsetA, int lda);

    public static native int dlascl (int Order, int type, int k1, int ku, double cfrom, double cto,
                                     int M, int N, Buffer A, int offsetA, int lda);

    public static native void slascl2 (int M, int N,
                                      Buffer D, int offsetD, Buffer X, int offsetX, int strideX);

    public static native void dlascl2 (int M, int N,
                                      Buffer D, int offsetD, Buffer X, int offsetX, int strideX);

    public static native int slaswp (int Order, int N, Buffer A, int offsetA, int lda, int k1, int k2,
                                     Buffer ipiv, int offsetIpiv, int incIpiv);

    public static native int dlaswp (int Order, int N, Buffer A, int offsetA, int lda, int k1, int k2,
                                     Buffer ipiv, int offsetIpiv, int incIpiv);

    public static native int slapmr (int Order, boolean forward, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer k, int offsetK);

    public static native int dlapmr (int Order, boolean forward, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer k, int offsetK);

    public static native int slapmt (int Order, boolean forward, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer k, int offsetK);

    public static native int dlapmt (int Order, boolean forward, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer k, int offsetK);

    public static native int ssyconv (int Order, int uplo, int way, int N,
                                      Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                      Buffer e, int offsetE);

    public static native int dsyconv (int Order, int uplo, int way, int N,
                                      Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                      Buffer e, int offsetE);

    public static native int slasrt (int id, int N, Buffer X, int offsetX);

    public static native int dlasrt (int id, int N, Buffer X, int offsetX);

    public static native void slagtm (int trans, int n, int nrhs,
                                      float alpha, Buffer D, int offsetD, Buffer B, int offsetB, int ldb,
                                      float beta,  Buffer C, int offsetC, int ldc);

    public static native void dlagtm (int trans, int n, int nrhs,
                                      double alpha, Buffer D, int offsetD, Buffer B, int offsetB, int ldb,
                                      double beta,  Buffer C, int offsetC, int ldc);

    public static native void slastm (int trans, int n, int nrhs,
                                      float alpha, Buffer D, int offsetD, Buffer B, int offsetB, int ldb,
                                      float beta,  Buffer C, int offsetC, int ldc);

    public static native void dlastm (int trans, int n, int nrhs,
                                      double alpha, Buffer D, int offsetD, Buffer B, int offsetB, int ldb,
                                      double beta,  Buffer C, int offsetC, int ldc);

    //==============================================================================================
    /*
     * -----------------------------------------------------------------
     * Linear Equation Routines
     * -----------------------------------------------------------------
     */

    /*
     * -----------------------------------------------------------------
     * Triangularization - TRF
     * -----------------------------------------------------------------
     */

    public static native int sgetrf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv);

    public static native int dgetrf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv);

    public static native int sgbtrf (int Order, int M, int N, int kl, int ku,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv);

    public static native int dgbtrf (int Order, int M, int N, int kl, int ku,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv);

    public static native int dgttrf (int N, Buffer D, int offsetD, Buffer ipiv, int offsetIpiv);

    public static native int sgttrf (int N, Buffer D, int offsetD, Buffer ipiv, int offsetIpiv);

    public static native int ddttrfb (int N, Buffer D, int offsetD);

    public static native int sdttrfb (int N, Buffer D, int offsetD);

    public static native int spotrf (int Order, int uplo, int N, Buffer A, int offsetA, int lda);

    public static native int dpotrf (int Order, int uplo, int N, Buffer A, int offsetA, int lda);

    public static native int spptrf (int Order, int uplo, int N, Buffer A, int offsetA);

    public static native int dpptrf (int Order, int uplo, int N, Buffer A, int offsetA);

    public static native int spbtrf (int Order, int uplo, int N, int kd, Buffer A, int offsetA, int lda);

    public static native int dpbtrf (int Order, int uplo, int N, int kd, Buffer A, int offsetA, int lda);

    public static native int dpttrf (int N, Buffer D, int offsetD);

    public static native int spttrf (int N, Buffer D, int offsetD);

    public static native int ssytrf (int Order, int uplo, int N,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv);

    public static native int dsytrf (int Order, int uplo, int N,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv);

    public static native int ssptrf (int Order, int uplo, int N,
                                     Buffer A, int offsetA, Buffer ipiv, int offsetIpiv);

    public static native int dsptrf (int Order, int uplo, int N,
                                     Buffer A, int offsetA, Buffer ipiv, int offsetIpiv);

    /*
     * -----------------------------------------------------------------
     * Solving the system - TRS
     * -----------------------------------------------------------------
     */

    public static native int sgetrs (int Order, int trans, int N, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                     Buffer B, int offsetB, int ldb);

    public static native int dgetrs (int Order, int trans, int N, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                     Buffer B, int offsetB, int ldb);

    public static native int sgbtrs (int Order, int trans, int N, int kl, int ku, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                     Buffer B, int offsetB, int ldb);

    public static native int dgbtrs (int Order, int trans, int N, int kl, int ku, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                     Buffer B, int offsetB, int ldb);

    public static native int sgttrs (int Order, int trans, int N, int nrhs,
                                     Buffer D, int offsetD, Buffer ipiv, int offsetIpiv,
                                     Buffer B, int offsetB, int ldb);

    public static native int dgttrs (int Order, int trans, int N, int nrhs,
                                     Buffer D, int offsetD, Buffer ipiv, int offsetIpiv,
                                     Buffer B, int offsetB, int ldb);

    public static native int sdttrsb (int trans, int N, int nrhs,
                                      Buffer D, int offsetD, Buffer B, int offsetB, int ldb);

    public static native int ddttrsb (int trans, int N, int nrhs,
                                      Buffer D, int offsetD, Buffer B, int offsetB, int ldb);

    public static native int spotrs (int Order, int uplo, int N, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int dpotrs (int Order, int uplo, int N, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int spptrs (int Order, int uplo, int N, int nrhs,
                                     Buffer A, int offsetA, Buffer B, int offsetB, int ldb);

    public static native int dpptrs (int Order, int uplo, int N, int nrhs,
                                     Buffer A, int offsetA, Buffer B, int offsetB, int ldb);

    public static native int spbtrs (int Order, int uplo, int N, int kd, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int dpbtrs (int Order, int uplo, int N, int kd, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int spttrs (int Order, int N, int nrhs,
                                     Buffer D, int offsetD, Buffer B, int offsetB, int ldb);

    public static native int dpttrs (int Order, int N, int nrhs,
                                     Buffer D, int offsetD, Buffer B, int offsetB, int ldb);

    public static native int ssytrs (int Order, int uplo, int N, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                     Buffer B, int offsetB, int ldb);

    public static native int dsytrs (int Order, int uplo, int N, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                     Buffer B, int offsetB, int ldb);

    public static native int ssptrs (int Order, int uplo, int N, int nrhs,
                                     Buffer A, int offsetA, Buffer ipiv, int offsetIpiv,
                                     Buffer B, int offsetB, int ldb);

    public static native int dsptrs (int Order, int uplo, int N, int nrhs,
                                     Buffer A, int offsetA, Buffer ipiv, int offsetIpiv,
                                     Buffer B, int offsetB, int ldb);

    public static native int strtrs (int Order, int uplo, int trans, int diag, int N, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int dtrtrs (int Order, int uplo, int trans, int diag, int N, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int stptrs (int Order, int uplo, int trans, int diag, int N, int nrhs,
                                     Buffer A, int offsetA, Buffer B, int offsetB, int ldb);

    public static native int dtptrs (int Order, int uplo, int trans, int diag, int N, int nrhs,
                                     Buffer A, int offsetA, Buffer B, int offsetB, int ldb);

    public static native int stbtrs (int Order, int uplo, int trans, int diag, int N, int kd, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int dtbtrs (int Order, int uplo, int trans, int diag, int N, int kd, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    /*
     * -----------------------------------------------------------------
     * Estimating Condition Number - CON
     * -----------------------------------------------------------------
     */

    public static native int sgecon (int Order, int norm, int N,
                                     Buffer A, int offsetA, int lda, float anorm, Buffer rcond);

    public static native int dgecon (int Order, int norm, int N,
                                     Buffer A, int offsetA, int lda, double anorm, Buffer rcond);

    public static native int sgbcon (int Order, int norm, int N, int kl, int ku,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                     float anorm, Buffer rcond);

    public static native int dgbcon (int Order, int norm, int N, int kl, int ku,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                     double anorm, Buffer rcond);

    public static native int sgtcon (int norm, int N, Buffer D, int offsetD,
                                     Buffer ipiv, int offsetIpiv, float anorm, Buffer rcond);

    public static native int dgtcon (int norm, int N, Buffer D, int offsetD,
                                     Buffer ipiv, int offsetIpiv, double anorm, Buffer rcond);

    public static native int spocon (int Order, int uplo, int N,
                                     Buffer A, int offsetA, int lda, float anorm, Buffer rcond);

    public static native int dpocon (int Order, int uplo, int N,
                                     Buffer A, int offsetA, int lda, double anorm, Buffer rcond);

    public static native int sppcon (int Order, int uplo, int N,
                                     Buffer A, int offsetA, float anorm, Buffer rcond);

    public static native int dppcon (int Order, int uplo, int N,
                                     Buffer A, int offsetA, double anorm, Buffer rcond);

    public static native int spbcon (int Order, int uplo, int N, int kd,
                                     Buffer A, int offsetA, int lda, float anorm, Buffer rcond);

    public static native int dpbcon (int Order, int uplo, int N, int kd,
                                     Buffer A, int offsetA, int lda, double anorm, Buffer rcond);

    public static native int sptcon (int N, Buffer D, int offsetD, float anorm, Buffer rcond);

    public static native int dptcon (int N, Buffer D, int offsetD, double anorm, Buffer rcond);

    public static native int ssycon (int Order, int uplo, int N, Buffer A, int offsetA, int lda,
                                     Buffer ipiv, int offsetIpiv, float anorm, Buffer rcond);

    public static native int dsycon (int Order, int uplo, int N, Buffer A, int offsetA, int lda,
                                     Buffer ipiv, int offsetIpiv, double anorm, Buffer rcond);

    public static native int sspcon (int Order, int uplo, int N, Buffer A, int offsetA,
                                     Buffer ipiv, int offsetIpiv, float anorm, Buffer rcond);

    public static native int dspcon (int Order, int uplo, int N, Buffer A, int offsetA,
                                     Buffer ipiv, int offsetIpiv, double anorm, Buffer rcond);

    public static native int strcon (int Order, int norm, int uplo, int diag, int N,
                                     Buffer A, int offsetA, int lda, Buffer rcond);

    public static native int dtrcon (int Order, int norm, int uplo, int diag, int N,
                                     Buffer A, int offsetA, int lda, Buffer rcond);

    public static native int stpcon (int Order, int norm, int uplo, int diag, int N,
                                     Buffer A, int offsetA, Buffer rcond);

    public static native int dtpcon (int Order, int norm, int uplo, int diag, int N,
                                     Buffer A, int offsetA, Buffer rcond);

    public static native int stbcon (int Order, int norm, int uplo, int diag, int N, int kd,
                                     Buffer A, int offsetA, int lda, Buffer rcond);

    public static native int dtbcon (int Order, int norm, int uplo, int diag, int N, int kd,
                                     Buffer A, int offsetA, int lda, Buffer rcond);

    /*
     * -----------------------------------------------------------------
     * Inverse matrix - TRI
     * -----------------------------------------------------------------
     */

    public static native int sgetri (int Order, int N,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv);

    public static native int dgetri (int Order, int N,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv);

    public static native int spotri (int Order, int uplo, int N, Buffer A, int offsetA, int lda);

    public static native int dpotri (int Order, int uplo, int N, Buffer A, int offsetA, int lda);

    public static native int spptri (int Order, int uplo, int N, Buffer A, int offsetA);

    public static native int dpptri (int Order, int uplo, int N, Buffer A, int offsetA);

    public static native int ssytri (int Order, int uplo, int N,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv);

    public static native int dsytri (int Order, int uplo, int N,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv);

    public static native int ssptri (int Order, int uplo, int N,
                                     Buffer A, int offsetA, Buffer ipiv, int offsetIpiv);

    public static native int dsptri (int Order, int uplo, int N,
                                     Buffer A, int offsetA, Buffer ipiv, int offsetIpiv);

    public static native int strtri (int Order, int uplo, int diag, int N, Buffer A, int offsetA, int lda);

    public static native int dtrtri (int Order, int uplo, int diag, int N, Buffer A, int offsetA, int lda);

    public static native int stptri (int Order, int uplo, int diag, int N, Buffer A, int offsetA);

    public static native int dtptri (int Order, int uplo, int diag, int N, Buffer A, int offsetA);

    /*
     * -----------------------------------------------------------------
     * Solve system of linear equations - SV
     * -----------------------------------------------------------------
     */

    public static native int sgesv (int Order, int N, int nrhs, Buffer A, int offsetA, int lda,
                                    Buffer ipiv, int offsetIpiv, Buffer B, int offsetB, int ldb);

    public static native int dgesv (int Order, int N, int nrhs, Buffer A, int offsetA, int lda,
                                    Buffer ipiv, int offsetIpiv, Buffer B, int offsetB, int ldb);

    public static native int sgbsv (int Order, int N, int kl, int ku, int nrhs,
                                    Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                    Buffer B, int offsetB, int ldb);

    public static native int dgbsv (int Order, int N, int kl, int ku, int nrhs,
                                    Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                    Buffer B, int offsetB, int ldb);

    public static native int sgtsv (int Order, int N, int nrhs, Buffer D, int offsetD,
                                    Buffer B, int offsetB, int ldb);

    public static native int dgtsv (int Order, int N, int nrhs, Buffer D, int offsetD,
                                    Buffer B, int offsetB, int ldb);

    public static native int sdtsv (int N, int nrhs, Buffer D, int offsetD,
                                    Buffer B, int offsetB, int ldb);

    public static native int ddtsv (int N, int nrhs, Buffer D, int offsetD,
                                    Buffer B, int offsetB, int ldb);

    public static native int sposv (int Order, int uplo, int N, int nrhs,
                                    Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int dposv (int Order, int uplo, int N, int nrhs,
                                    Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int sppsv (int Order, int uplo, int N, int nrhs,
                                    Buffer A, int offsetA, Buffer B, int offsetB, int ldb);

    public static native int dppsv (int Order, int uplo, int N, int nrhs,
                                    Buffer A, int offsetA, Buffer B, int offsetB, int ldb);

    public static native int spbsv (int Order, int uplo, int N, int kd, int nrhs,
                                    Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int dpbsv (int Order, int uplo, int N, int kd, int nrhs,
                                    Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int sptsv (int Order, int N, int nrhs, Buffer D, int offsetD,
                                    Buffer B, int offsetB, int ldb);

    public static native int dptsv (int Order, int N, int nrhs, Buffer D, int offsetD,
                                    Buffer B, int offsetB, int ldb);

    public static native int ssysv (int Order, int uplo, int N, int nrhs,
                                    Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                    Buffer B, int offsetB, int ldb);

    public static native int dsysv (int Order, int uplo, int N, int nrhs,
                                    Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                    Buffer B, int offsetB, int ldb);

    public static native int sspsv (int Order, int uplo, int N, int nrhs,
                                    Buffer A, int offsetA, Buffer ipiv, int offsetIpiv,
                                    Buffer B, int offsetB, int ldb);

    public static native int dspsv (int Order, int uplo, int N, int nrhs,
                                    Buffer A, int offsetA, Buffer ipiv, int offsetIpiv,
                                    Buffer B, int offsetB, int ldb);

    //=============================================================================================
    /*
     * -----------------------------------------------------------------
     * Orthogonal Factorization Routines
     * -----------------------------------------------------------------
     */
    // QRF

    public static native int sgeqrf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dgeqrf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int sorgqr(int Order, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dorgqr(int Order, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int sormqr(int Order, int side, int trans, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau,
                                    Buffer B, int offsetB, int ldb);

    public static native int dormqr(int Order, int side, int trans, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau,
                                    Buffer B, int offsetB, int ldb);

    // QP3
    public static native int sgeqp3 (int Order, int M, int N, Buffer A, int offsetA, int lda,
                                     Buffer jpiv, int offsetJpiv, Buffer tau, int offsetTau);

    public static native int dgeqp3 (int Order, int M, int N, Buffer A, int offsetA, int lda,
                                     Buffer jpiv, int offsetJpiv, Buffer tau, int offsetTau);

    // QRFP
    public static native int sgeqrfp (int Order, int M, int N,
                                      Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dgeqrfp (int Order, int M, int N,
                                      Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);


    // RQF
    public static native int sgerqf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dgerqf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int sorgrq(int Order, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dorgrq(int Order, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int sormrq(int Order, int side, int trans, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau,
                                    Buffer B, int offsetB, int ldb);

    public static native int dormrq(int Order, int side, int trans, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau,
                                    Buffer B, int offsetB, int ldb);

    // LQF
    public static native int sgelqf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dgelqf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int sorglq(int Order, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dorglq(int Order, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int sormlq(int Order, int side, int trans, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau,
                                    Buffer B, int offsetB, int ldb);

    public static native int dormlq(int Order, int side, int trans, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau,
                                    Buffer B, int offsetB, int ldb);

    // QLF
    public static native int sgeqlf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dgeqlf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int sorgql(int Order, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dorgql(int Order, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int sormql(int Order, int side, int trans, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau,
                                    Buffer B, int offsetB, int ldb);

    public static native int dormql(int Order, int side, int trans, int M, int N, int K,
                                    Buffer A, int offsetA, int lda, Buffer tau, int offsetTau,
                                    Buffer B, int offsetB, int ldb);


    /*
     * -----------------------------------------------------------------
     * Linear Least Squares Routines
     * -----------------------------------------------------------------
     */

    public static native int sgels (int Order, int trans, int M, int N, int nrhs,
                                    Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int dgels (int Order, int trans, int M, int N, int nrhs,
                                    Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb);

    public static native int sgglse (int Order, int M, int N, int p,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb,
                                     Buffer C, int offsetC, Buffer D, int offsetD, Buffer X, int offsetX);

    public static native int dgglse (int Order, int M, int N, int p,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb,
                                     Buffer C, int offsetC, Buffer D, int offsetD, Buffer X, int offsetX);

    public static native int sggglm (int Order, int M, int N, int p,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb,
                                     Buffer D, int offsetD, Buffer X, int offsetX, Buffer Y, int offsetY);

    public static native int dggglm (int Order, int M, int N, int p,
                                     Buffer A, int offsetA, int lda, Buffer B, int offsetB, int ldb,
                                     Buffer D, int offsetD, Buffer X, int offsetX, Buffer Y, int offsetY);

    /*
     * -----------------------------------------------------------------
     * Non-Symmetric Eigenvalue Problem Routines
     * -----------------------------------------------------------------
     */

    public static native int sgeev (int Order, int jobvl, int jobvr, int N, Buffer a, int offsetA, int lda,
                                    Buffer WR, int offsetWR, Buffer WI, int offsetWI,
                                    Buffer VL, int offsetVL, int ldvl, Buffer VR, int offsetVR, int ldvr);

    public static native int dgeev (int Order, int jobvl, int jobvr, int N, Buffer a, int offsetA, int lda,
                                    Buffer WR, int offsetWR, Buffer WI, int offsetWI,
                                    Buffer VL, int offsetVL, int ldvl, Buffer VR, int offsetVR, int ldvr);

    public static native int sgees (int Order, int jobvs, int N, Buffer a, int offsetA, int lda,
                                    Buffer WR, int offsetWR, Buffer WI, int offsetWI,
                                    Buffer VS, int offsetVS, int ldvs);

    public static native int dgees (int Order, int jobvs, int N, Buffer a, int offsetA, int lda,
                                    Buffer WR, int offsetWR, Buffer WI, int offsetWI,
                                    Buffer VS, int offsetVS, int ldvs);

    /*
     * -----------------------------------------------------------------
     * Singular Value Decomposition Routines
     * -----------------------------------------------------------------
     */

    public static native int sgebrd (int Order, int M, int N, Buffer a, int offsetA, int lda,
                                     Buffer D, int offsetD,
                                     Buffer tauq, int offsetTauq, Buffer taup, int offsetTaup);

    public static native int dgebrd (int Order, int M, int N, Buffer a, int offsetA, int lda,
                                     Buffer D, int offsetD,
                                     Buffer tauq, int offsetTauq, Buffer taup, int offsetTaup);

    public static native int sorgbr (int Order, int vect, int M, int N, int K,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dorgbr (int Order, int vect, int M, int N, int K,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int sbdsqr (int Order, int uplo, int N, int ncvt, int nru, int ncc,
                                     Buffer D, int offsetD,
                                     Buffer VT, int offsetVT, int ldvt, Buffer U, int offsetU, int ldu,
                                     Buffer C, int offsetC, int ldc);

    public static native int dbdsqr (int Order, int uplo, int N, int ncvt, int nru, int ncc,
                                     Buffer D, int offsetD,
                                     Buffer VT, int offsetVT, int ldvt, Buffer U, int offsetU, int ldu,
                                     Buffer C, int offsetC, int ldc);

    public static native int sgesvd (int Order, int jobu, int jobvt, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer S, int offsetS,
                                     Buffer U, int offsetU, int ldu, Buffer VT, int offsetVT, int ldvt,
                                     Buffer superb, int offsetSuperb);

    public static native int dgesvd (int Order, int jobu, int jobvt, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer S, int offsetS,
                                     Buffer U, int offsetU, int ldu, Buffer VT, int offsetVT, int ldvt,
                                     Buffer superb, int offsetSuperb);

    public static native int sgesdd (int Order, int jobz, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer S, int offsetS,
                                     Buffer U, int offsetU, int ldu, Buffer VT, int offsetVT, int ldvt);

    public static native int dgesdd (int Order, int jobz, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer S, int offsetS,
                                     Buffer U, int offsetU, int ldu, Buffer VT, int offsetVT, int ldvt);

}
