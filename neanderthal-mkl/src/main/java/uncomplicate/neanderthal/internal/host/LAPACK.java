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

    public static native int sgetrf(int Order, int N, Buffer A, int offsetA, int lda, Buffer ipiv);

    public static native int dgetrf(int Order, int N, Buffer A, int offsetA, int lda, Buffer ipiv);

    public static native int sgesv(int Order, int N, int nrhs,
                                   Buffer A, int offsetA, int lda,
                                   Buffer ipiv,
                                   Buffer B, int offsetB, int ldb);

    public static native int dgesv(int Order, int N, int nrhs,
                                   Buffer A, int offsetA, int lda,
                                   Buffer ipiv,
                                   Buffer B, int offsetB, int ldb);
}
