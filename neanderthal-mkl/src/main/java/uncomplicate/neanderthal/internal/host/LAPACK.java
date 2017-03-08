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

    public static native int slaswp (int Order, int N, Buffer A, int offsetA, int lda, int k1, int k2,
                                     Buffer ipiv, int offsetIpiv, int incIpiv);

    public static native int dlaswp (int Order, int N, Buffer A, int offsetA, int lda, int k1, int k2,
                                     Buffer ipiv, int offsetIpiv, int incIpiv);

    public static native int slasrt (int id, int N, Buffer X, int offsetX);

    public static native int dlasrt (int id, int N, Buffer X, int offsetX);

    /*
     * -----------------------------------------------------------------
     * Linear Equation Routines
     * -----------------------------------------------------------------
     */

    public static native int sgetrf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv);

    public static native int dgetrf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv);

    public static native int sgetrs (int Order, int trans, int N, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                     Buffer B, int offsetB, int ldb);

    public static native int dgetrs (int Order, int trans, int N, int nrhs,
                                     Buffer A, int offsetA, int lda, Buffer ipiv, int offsetIpiv,
                                     Buffer B, int offsetB, int ldb);

    public static native int sgesv (int Order, int N, int nrhs,
                                    Buffer A, int offsetA, int lda,
                                    Buffer ipiv, int offsetIpiv,
                                    Buffer B, int offsetB, int ldb);

    public static native int dgesv (int Order, int N, int nrhs,
                                    Buffer A, int offsetA, int lda,
                                    Buffer ipiv, int offsetIpiv,
                                    Buffer B, int offsetB, int ldb);

    /*
     * -----------------------------------------------------------------
     * Orthogonal Factorization Routines
     * -----------------------------------------------------------------
     */

    public static native int sgeqrf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dgeqrf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int sgeqrfp (int Order, int M, int N,
                                      Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dgeqrfp (int Order, int M, int N,
                                      Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int sgerqf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dgerqf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int sgelqf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dgelqf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int sgeqlf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    public static native int dgeqlf (int Order, int M, int N,
                                     Buffer A, int offsetA, int lda, Buffer tau, int offsetTau);

    /*
     * -----------------------------------------------------------------
     * Linear Least Squares Routines
     * -----------------------------------------------------------------
     */

    public static native int sgels (int Order, int trans, int M, int N, int nrhs,
                                    Buffer A, int offsetA, int lda, Buffer b, int offsetB, int ldb);

    public static native int dgels (int Order, int trans, int M, int N, int nrhs,
                                    Buffer A, int offsetA, int lda, Buffer b, int offsetB, int ldb);

    /*
     * -----------------------------------------------------------------
     * Non-Symmetric Eigenvalue Problem Routines
     * -----------------------------------------------------------------
     */

    public static native int sgeev (int Order, int jobvl, int jobvr, int N, Buffer a, int offsetA, int lda,
                                    Buffer WR, int offsetWR, Buffer WL, int offsetWL,
                                    Buffer VL, int offsetVL, int ldvl, Buffer VR, int offsetVR, int ldvr);

    public static native int dgeev (int Order, int jobvl, int jobvr, int N, Buffer a, int offsetA, int lda,
                                    Buffer WR, int offsetWR, Buffer WL, int offsetWL,
                                    Buffer VL, int offsetVL, int ldvl, Buffer VR, int offsetVR, int ldvr);

    /*
     * -----------------------------------------------------------------
     * Singular Value Decomposition Routines
     * -----------------------------------------------------------------
     */

    public static native int sgebrd (int Order, int M, int N, Buffer a, int offsetA, int lda,
                                     Buffer D, int offsetD, Buffer E, int offsetE,
                                     Buffer tauq, int offsetTauq, Buffer taup, int offsetTaup);

    public static native int dgebrd (int Order, int M, int N, Buffer a, int offsetA, int lda,
                                     Buffer D, int offsetD, Buffer E, int offsetE,
                                     Buffer tauq, int offsetTauq, Buffer taup, int offsetTaup);

    public static native int sbdsqr (int Order, int uplo, int N, int ncvt, int nru, int ncc,
                                     Buffer D, int offsetD, Buffer E, int offsetE,
                                     Buffer VT, int offsetVT, int ldvt, Buffer U, int offsetU, int ldu,
                                     Buffer C, int offsetC, int ldc);

    public static native int dbdsqr (int Order, int uplo, int N, int ncvt, int nru, int ncc,
                                     Buffer D, int offsetD, Buffer E, int offsetE,
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

}
