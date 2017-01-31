//   Copyright (c) Dragan Djuric. All rights reserved.
//   The use and distribution terms for this software are covered by the
//   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
//   which can be found in the file LICENSE at the root of this distribution.
//   By using this software in any fashion, you are agreeing to be bound by
//   the terms of this license.
//   You must not remove this notice, or any other, from this software.

#include <stdlib.h>
#include <jni.h>
#include "cblas.h"
#include "uncomplicate_neanderthal_CBLAS.h"


JavaVM *javavm;

JNIEXPORT jint JNICALL JNI_OnLoad (JavaVM *jvm, void *reserved) {
    javavm=jvm;
    return JNI_VERSION_1_2;
}

int cblas_errprn(int ierr, int info, char *form, ...) {
    JNIEnv *env;
    (*javavm)->AttachCurrentThread(javavm, (void **)&env, NULL);
    jclass iaexception = (*env)->FindClass(env, "java/lang/IllegalArgumentException");

    va_list argptr;
    va_start(argptr, form);

    int len = vsnprintf(NULL, 0, form, argptr) + 1;
    char *message = malloc((unsigned)len);
    vsprintf(message, form, argptr);

    (*env)->ThrowNew(env, iaexception, message);
    va_end(argptr);
    if (ierr < info)
        return(ierr);
    else return(info);
};

void cblas_xerbla(int p, const char *rout, const char *form, ...) {
    // Override exit(-1) of the original cblas_xerbla.
    // In ATLAS, form is empty, so we have to use cblass_errprn
    // to get the error information.
    return;
};

/*
 * ======================================================
 * Level 1 BLAS functions
 * ======================================================
 */

/*
 * ------------------------------------------------------
 * DOT
 * ------------------------------------------------------
 */

JNIEXPORT jfloat JNICALL Java_uncomplicate_neanderthal_CBLAS_sdsdot
(JNIEnv *env, jclass clazz, jint N, jfloat alpha,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    return cblas_sdsdot(N, alpha, cX + offsetX, incX, cY + offsetY, incY);
};


JNIEXPORT jdouble JNICALL Java_uncomplicate_neanderthal_CBLAS_dsdot
(JNIEnv *env, jclass clazz, jint N,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    return cblas_dsdot(N, cX + offsetX, incX, cY + offsetY, incY);
};

JNIEXPORT jdouble JNICALL Java_uncomplicate_neanderthal_CBLAS_ddot
(JNIEnv *env, jclass clazz, jint N,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    return cblas_ddot(N, cX + offsetX, incX, cY + offsetY, incY);
};

JNIEXPORT jfloat JNICALL Java_uncomplicate_neanderthal_CBLAS_sdot
(JNIEnv *env, jclass clazz, jint N,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    return cblas_sdot(N, cX + offsetX, incX, cY + offsetY, incY);
};

/*
 * ------------------------------------------------------
 * NRM2
 * ------------------------------------------------------
 */

JNIEXPORT jfloat JNICALL Java_uncomplicate_neanderthal_CBLAS_snrm2
(JNIEnv *env, jclass clazz, jint N, jobject X, jint offsetX, jint incX) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    return cblas_snrm2(N, cX + offsetX, incX);
};

JNIEXPORT jdouble JNICALL Java_uncomplicate_neanderthal_CBLAS_dnrm2
(JNIEnv *env, jclass clazz, jint N, jobject X, jint offsetX, jint incX) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    return cblas_dnrm2(N, cX + offsetX, incX);
};

/*
 * ------------------------------------------------------
 * ASUM
 * ------------------------------------------------------
 */

JNIEXPORT jfloat JNICALL Java_uncomplicate_neanderthal_CBLAS_sasum
(JNIEnv *env, jclass clazz, jint N, jobject X, jint offsetX, jint incX) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    return cblas_sasum(N, cX + offsetX, incX);
};

JNIEXPORT jdouble JNICALL Java_uncomplicate_neanderthal_CBLAS_dasum
(JNIEnv *env, jclass clazz, jint N, jobject X, jint offsetX, jint incX) {
    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    return cblas_dasum(N, cX + offsetX, incX);
};


/*
 * ------------------------------------------------------
 * BLAS PLUS: SUM
 * ------------------------------------------------------
 */

JNIEXPORT jfloat JNICALL Java_uncomplicate_neanderthal_CBLAS_ssum
(JNIEnv *env, jclass clazz, jint N,
 jobject X, jint offsetX, jint incX) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X) + offsetX;

    const int stride = incX;
    const int step = 16;
    const int tail = N % step;
    const int n = N - tail;

    float *end = cX + n * incX;

    float acc = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float acc8 = 0.0f;
    float acc9 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;
    float acc14 = 0.0f;
    float acc15 = 0.0f;

    while (cX != end) {
        acc0 += cX[0];
        acc1 += cX[stride];
        acc2 += cX[2*stride];
        acc3 += cX[3*stride];
        acc4 += cX[4*stride];
        acc5 += cX[5*stride];
        acc6 += cX[6*stride];
        acc7 += cX[7*stride];
        acc8 += cX[8*stride];
        acc9 += cX[9*stride];
        acc10 += cX[10*stride];
        acc11 += cX[11*stride];
        acc12 += cX[12*stride];
        acc13 += cX[13*stride];
        acc14 += cX[14*stride];
        acc15 += cX[15*stride];
        cX += 16 * stride;
    }

    for (int i = 0; i < tail; i++) {
        acc += end[i * stride];
    }

    return acc + acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7
        + acc8 + acc9 + acc10 + acc11 + acc12 + acc13 + acc14 + acc15;
};

JNIEXPORT jdouble JNICALL Java_uncomplicate_neanderthal_CBLAS_dsum
(JNIEnv *env, jclass clazz, jint N, jobject X, jint offsetX, jint incX) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X) + offsetX;

    const int stride = incX;
    const int step = 16;
    const int tail = N % step;
    const int n = N - tail;

    double *end = cX + n * incX;

    double acc = 0.0f;
    double acc0 = 0.0f;
    double acc1 = 0.0f;
    double acc2 = 0.0f;
    double acc3 = 0.0f;
    double acc4 = 0.0f;
    double acc5 = 0.0f;
    double acc6 = 0.0f;
    double acc7 = 0.0f;
    double acc8 = 0.0f;
    double acc9 = 0.0f;
    double acc10 = 0.0f;
    double acc11 = 0.0f;
    double acc12 = 0.0f;
    double acc13 = 0.0f;
    double acc14 = 0.0f;
    double acc15 = 0.0f;

    while (cX != end) {
        acc0 += cX[0];
        acc1 += cX[stride];
        acc2 += cX[2*stride];
        acc3 += cX[3*stride];
        acc4 += cX[4*stride];
        acc5 += cX[5*stride];
        acc6 += cX[6*stride];
        acc7 += cX[7*stride];
        acc8 += cX[8*stride];
        acc9 += cX[9*stride];
        acc10 += cX[10*stride];
        acc11 += cX[11*stride];
        acc12 += cX[12*stride];
        acc13 += cX[13*stride];
        acc14 += cX[14*stride];
        acc15 += cX[15*stride];
        cX += 16 * stride;
    }

    for (int i = 0; i < tail; i++) {
        acc += end[i * stride];
    }

    return acc + acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7
        + acc8 + acc9 + acc10 + acc11 + acc12 + acc13 + acc14 + acc15;
};


/*
 * ------------------------------------------------------
 * IAMAX
 * ------------------------------------------------------
 */

JNIEXPORT jint JNICALL Java_uncomplicate_neanderthal_CBLAS_isamax
(JNIEnv *env, jclass clazz, jint N, jobject X, jint offsetX, jint incX) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    return cblas_isamax(N, cX + offsetX, incX);
};

JNIEXPORT jint JNICALL Java_uncomplicate_neanderthal_CBLAS_idamax
(JNIEnv *env, jclass clazz, jint N, jobject X, jint offsetX, jint incX) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    return cblas_idamax(N, cX + offsetX, incX);
};

/*
 * ======================================================
 * Level 1 BLAS procedures
 * ======================================================
 */

/*
 * ------------------------------------------------------
 * ROT
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_srot
(JNIEnv *env, jclass clazz, jint N,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY,
 jobject cs, jint offset_cs, jint inc_cs) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    float *c_cs = (float *) (*env)->GetDirectBufferAddress(env, cs) + offset_cs;
    cblas_srot(N, cX + offsetX, incX, cY + offsetY, incY, c_cs[0], c_cs[inc_cs]);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_drot
(JNIEnv *env, jclass clazz, jint N,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY,
 jobject cs, jint offset_cs, jint inc_cs) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    double *c_cs = (double *) (*env)->GetDirectBufferAddress(env, cs) + offset_cs;
    cblas_drot(N, cX + offsetX, incX, cY + offsetY, incY, c_cs[0], c_cs[inc_cs]);
};

/*
 * ------------------------------------------------------
 * ROTG
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_srotg
(JNIEnv *env, jclass clazz,
 jobject ab, jint offset_ab, jint inc_ab,
 jobject cs, jint offset_cs, jint inc_cs) {

    float *c_ab = (float *) (*env)->GetDirectBufferAddress(env, ab) + offset_ab;
    float *c_cs = (float *) (*env)->GetDirectBufferAddress(env, cs) + offset_cs;
    cblas_srotg(c_ab, c_ab + inc_ab, c_cs, c_cs + inc_cs);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_drotg
(JNIEnv *env, jclass clazz,
 jobject ab, jint offset_ab, jint inc_ab,
 jobject cs, jint offset_cs, jint inc_cs) {

    double *c_ab = (double *) (*env)->GetDirectBufferAddress(env, ab) + offset_ab;
    double *c_cs = (double *) (*env)->GetDirectBufferAddress(env, cs) + offset_cs;
    cblas_drotg(c_ab, c_ab + inc_ab, c_cs, c_cs + inc_cs);
};

/*
 * ------------------------------------------------------
 * ROTM
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_srotm
(JNIEnv *env, jclass clazz, jint N,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY,
 jobject param) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    float *c_param = (float *) (*env)->GetDirectBufferAddress(env, param);
    cblas_srotm(N, cX + offsetX, incX, cY + offsetY, incY, c_param);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_drotm
(JNIEnv *env, jclass clazz, jint N,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY,
 jobject param) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    double *c_param = (double *) (*env)->GetDirectBufferAddress(env, param);
    cblas_drotm(N, cX + offsetX, incX, cY + offsetY, incY, c_param);
};

/*
 * ------------------------------------------------------
 * ROTMG
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_srotmg
(JNIEnv *env, jclass clazz,
 jobject d1d2xy, jint offset_d1d2xy, jint inc_d1d2xy,
 jobject param) {

    float *c_d1d2xy = (float *) (*env)->GetDirectBufferAddress(env, d1d2xy) + offset_d1d2xy;
    float *c_param = (float *) (*env)->GetDirectBufferAddress(env, param);
    cblas_srotmg(c_d1d2xy, c_d1d2xy + inc_d1d2xy, c_d1d2xy + 2 * inc_d1d2xy, c_d1d2xy[3*inc_d1d2xy], c_param);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_drotmg
(JNIEnv *env, jclass clazz, jobject d1d2xy, jint offset_d1d2xy, jint inc_d1d2xy,
 jobject param) {

    double *c_d1d2xy = (double *) (*env)->GetDirectBufferAddress(env, d1d2xy) + offset_d1d2xy;
    double *c_param = (double *) (*env)->GetDirectBufferAddress(env, param);
    cblas_drotmg(c_d1d2xy, c_d1d2xy + inc_d1d2xy, c_d1d2xy + 2 * inc_d1d2xy, c_d1d2xy[3*inc_d1d2xy], c_param);
};

/*
 * ------------------------------------------------------
 * SWAP
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_sswap
(JNIEnv *env, jclass clazz, jint N,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_sswap(N, cX + offsetX, incX, cY + offsetY, incY);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dswap
(JNIEnv *env, jclass clazz, jint N,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_dswap(N, cX + offsetX, incX, cY + offsetY, incY);

};

/*
 * ------------------------------------------------------
 * SCAL
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_sscal
(JNIEnv *env, jclass clazz,
 jint N, jfloat alpha,
 jobject X, jint offsetX, jint incX) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    cblas_sscal(N, alpha, cX + offsetX, incX);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dscal
(JNIEnv *env, jclass clazz,
 jint N, jdouble alpha,
 jobject X, jint offsetX, jint incX) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    cblas_dscal(N, alpha, cX + offsetX, incX);
};

/*
 * ------------------------------------------------------
 * COPY
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_scopy
(JNIEnv *env, jclass clazz, jint N,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_scopy(N, cX + offsetX, incX, cY + offsetY, incY);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dcopy
(JNIEnv *env, jclass clazz, jint N,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_dcopy(N, cX + offsetX, incX, cY + offsetY, incY);
};

/*
 * ------------------------------------------------------
 * AXPY
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_saxpy
(JNIEnv *env, jclass clazz,
 jint N, jfloat alpha,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_saxpy(N, alpha, cX + offsetX, incX, cY + offsetY, incY);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_daxpy
(JNIEnv *env, jclass clazz,
 jint N, jdouble alpha,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_daxpy(N, alpha, cX + offsetX, incX, cY + offsetY, incY);
};

/*
 * ======================================================
 * Level 2 BLAS procedures
 * ======================================================
 */

/*
 * ------------------------------------------------------
 * GEMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_sgemv
(JNIEnv * env, jclass clazz,
 jint Order, jint TransA,
 jint M, jint N,
 jfloat alpha,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX,
 jfloat beta,
 jobject Y, jint offsetY, jint incY) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_sgemv(Order, TransA, M, N, alpha, cA + offsetA, lda,
                cX + offsetX, incX, beta, cY + offsetY, incY);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dgemv
(JNIEnv * env, jclass clazz,
 jint Order, jint TransA,
 jint M, jint N,
 jdouble alpha,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX,
 jdouble beta,
 jobject Y, jint offsetY, jint incY) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_dgemv(Order, TransA, M, N, alpha, cA + offsetA, lda,
                cX + offsetX, incX, beta, cY + offsetY, incY);
};

/*
 * ------------------------------------------------------
 * GBMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_sgbmv
(JNIEnv *env, jclass clazz,
 jint Order, jint TransA,
 jint M, jint N,
 jint KL, jint KU,
 jfloat alpha,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX,
 jfloat beta,
 jobject Y, jint offsetY, jint incY) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_sgbmv(Order, TransA, M, N, KL, KU, alpha, cA + offsetA, lda,
                cX + offsetX, incX, beta, cY + offsetY, incY);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dgbmv
(JNIEnv *env, jclass clazz,
 jint Order, jint TransA,
 jint M, jint N,
 jint KL, jint KU,
 jdouble alpha,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX,
 jdouble beta,
 jobject Y, jint offsetY, jint incY) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_dgbmv(Order, TransA, M, N, KL, KU, alpha, cA + offsetA, lda,
                cX + offsetX, incX, beta, cY + offsetY, incY);
};

/*
 * ------------------------------------------------------
 * SYMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_ssymv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N,
 jfloat alpha,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX,
 jfloat beta,
 jobject Y, jint offsetY, jint incY) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_ssymv(Order, Uplo, N, alpha, cA + offsetA, lda,
                cX + offsetX, incX, beta, cY + offsetY, incY);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dsymv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N,
 jdouble alpha,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX,
 jdouble beta,
 jobject Y, jint offsetY, jint incY) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_dsymv(Order, Uplo, N, alpha, cA + offsetA, lda,
                cX + offsetX, incX, beta, cY + offsetY, incY);
};

/*
 * ------------------------------------------------------
 * SBMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_ssbmv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N, jint K,
 jfloat alpha,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX,
 jfloat beta,
 jobject Y, jint offsetY, jint incY) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_ssbmv(Order, Uplo, N, K, alpha, cA, lda,
                cX + offsetX, incX, beta, cY + offsetY, incY);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dsbmv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N, jint K,
 jdouble alpha,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX,
 jdouble beta,
 jobject Y, jint offsetY, jint incY) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_dsbmv(Order, Uplo, N, K, alpha, cA, lda,
                cX + offsetX, incX, beta, cY + offsetY, incY);
};

/*
 * ------------------------------------------------------
 * SPMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_sspmv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N,
 jfloat alpha,
 jobject Ap,
 jobject X, jint offsetX, jint incX,
 jfloat beta,
 jobject Y, jint offsetY, jint incY) {

    float *cAp = (float *) (*env)->GetDirectBufferAddress(env, Ap);
    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_sspmv(Order, Uplo, N, alpha, cAp,
                cX + offsetX, incX, beta, cY + offsetY, incY);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dspmv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N,
 jdouble alpha,
 jobject Ap,
 jobject X, jint offsetX, jint incX,
 jdouble beta,
 jobject Y, jint offsetY, jint incY) {

    double *cAp = (double *) (*env)->GetDirectBufferAddress(env, Ap);
    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    cblas_dspmv(Order, Uplo, N, alpha, cAp,
                cX + offsetX, incX, beta, cY + offsetY, incY);
};

/*
 * ------------------------------------------------------
 * TRMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_strmv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint TransA, jint Diag,
 jint N,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    cblas_strmv(Order, Uplo, TransA, Diag, N, cA + offsetA, lda, cX + offsetX, incX);
};


JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dtrmv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint TransA, jint Diag,
 jint N,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    cblas_dtrmv(Order, Uplo, TransA, Diag, N, cA + offsetA, lda, cX + offsetX, incX);
};

/*
 * ------------------------------------------------------
 * TBMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_stbmv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint TransA, jint Diag,
 jint N, jint K,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    cblas_strmv(Order, Uplo, TransA, Diag, N, cA + offsetA, lda, cX + offsetX, incX);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dtbmv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint TransA, jint Diag,
 jint N, jint K,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    cblas_dtrmv(Order, Uplo, TransA, Diag, N, cA + offsetA, lda, cX + offsetX, incX);
};

/*
 * ------------------------------------------------------
 * TPMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_stpmv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint TransA, jint Diag,
 jint N,
 jobject Ap,
 jobject X, jint offsetX, jint incX) {

    float *cAp = (float *) (*env)->GetDirectBufferAddress(env, Ap);
    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    cblas_stpmv(Order, Uplo, TransA, Diag, N, cAp, cX + offsetX, incX);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dtpmv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint TransA, jint Diag,
 jint N,
 jobject Ap,
 jobject X, jint offsetX, jint incX) {

    double *cAp = (double *) (*env)->GetDirectBufferAddress(env, Ap);
    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    cblas_dtpmv(Order, Uplo, TransA, Diag, N, cAp, cX + offsetX, incX);
};

/*
 * ------------------------------------------------------
 * TRSV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_strsv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint TransA, jint Diag,
 jint N,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    cblas_strsv(Order, Uplo, TransA, Diag, N, cA + offsetA, lda, cX + offsetX, incX);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dtrsv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint TransA, jint Diag,
 jint N,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    cblas_dtrsv(Order, Uplo, TransA, Diag, N, cA + offsetA, lda, cX + offsetX, incX);
};

/*
 * ------------------------------------------------------
 * TBSV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_stbsv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint TransA, jint Diag,
 jint N, jint K,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    cblas_stbsv(Order, Uplo, TransA, Diag, N, K, cA + offsetA, lda, cX + offsetX, incX);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dtbsv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint TransA, jint Diag,
 jint N, jint K,
 jobject A, jint offsetA, jint lda,
 jobject X, jint offsetX, jint incX) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    cblas_dtbsv(Order, Uplo, TransA, Diag, N, K, cA + offsetA, lda, cX + offsetX, incX);
};

/*
 * ------------------------------------------------------
 * TPSV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_stpsv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint TransA, jint Diag,
 jint N,
 jobject Ap,
 jobject X, jint offsetX, jint incX) {

    float *cAp = (float *) (*env)->GetDirectBufferAddress(env, Ap);
    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    cblas_stpsv(Order, Uplo, TransA, Diag, N, cAp, cX + offsetX, incX);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dtpsv
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint TransA, jint Diag,
 jint N,
 jobject Ap,
 jobject X, jint offsetX, jint incX) {

    double *cAp = (double *) (*env)->GetDirectBufferAddress(env, Ap);
    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    cblas_dtpsv(Order, Uplo, TransA, Diag, N, cAp, cX + offsetX, incX);
};

/*
 * ------------------------------------------------------
 * GER
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_sger
(JNIEnv *env, jclass clazz,
 jint Order,
 jint M, jint N,
 jfloat alpha,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY,
 jobject A, jint offsetA, jint lda) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    cblas_sger(Order, M, N, alpha, cX + offsetX, incX, cY + offsetY, incY, cA + offsetA, lda);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dger
(JNIEnv *env, jclass clazz,
 jint Order,
 jint M, jint N,
 jdouble alpha,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY,
 jobject A, jint offsetA, jint lda) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    cblas_dger(Order, M, N, alpha, cX + offsetX, incX, cY + offsetY, incY, cA + offsetA, lda);
};

/*
 * ------------------------------------------------------
 * SYR
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_ssyr
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N,
 jfloat alpha,
 jobject X, jint offsetX, jint incX,
 jobject A, jint offsetA, jint lda) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    cblas_ssyr(Order, Uplo, N, alpha, cX + offsetX, incX, cA + offsetA, lda);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dsyr
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N,
 jdouble alpha,
 jobject X, jint offsetX, jint incX,
 jobject A, jint offsetA, jint lda) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    cblas_dsyr(Order, Uplo, N, alpha, cX + offsetX, incX, cA + offsetA, lda);
}

/*
 * ------------------------------------------------------
 * SPR
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_sspr
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N,
 jfloat alpha,
 jobject X, jint offsetX, jint incX,
 jobject Ap) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cAp = (float *) (*env)->GetDirectBufferAddress(env, Ap);
    cblas_sspr(Order, Uplo, N, alpha, cX + offsetX, incX, cAp);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dspr
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N,
 jdouble alpha,
 jobject X, jint offsetX, jint incX,
 jobject Ap) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cAp = (double *) (*env)->GetDirectBufferAddress(env, Ap);
    cblas_dspr(Order, Uplo, N, alpha, cX + offsetX, incX, cAp);
};

/*
 * ------------------------------------------------------
 * SYR2
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_ssyr2
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N,
 jfloat alpha,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY,
 jobject A, jint offsetA, jint lda) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    cblas_ssyr2(Order, Uplo, N, alpha, cX + offsetX, incX, cY + offsetY, incY, cA + offsetA, lda);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dsyr2
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N,
 jdouble alpha,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY,
 jobject A, jint offsetA, jint lda) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    cblas_dsyr2(Order, Uplo, N, alpha, cX + offsetX, incX, cY + offsetY, incY, cA + offsetA, lda);
};

/*
 * ------------------------------------------------------
 * SPR2
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_sspr2
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N,
 jfloat alpha,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY,
 jobject Ap) {

    float *cX = (float *) (*env)->GetDirectBufferAddress(env, X);
    float *cY = (float *) (*env)->GetDirectBufferAddress(env, Y);
    float *cAp = (float *) (*env)->GetDirectBufferAddress(env, Ap);
    cblas_sspr2(Order, Uplo, N, alpha, cX + offsetX, incX, cY + offsetY, incY, cAp);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dspr2
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo,
 jint N,
 jdouble alpha,
 jobject X, jint offsetX, jint incX,
 jobject Y, jint offsetY, jint incY,
 jobject Ap) {

    double *cX = (double *) (*env)->GetDirectBufferAddress(env, X);
    double *cY = (double *) (*env)->GetDirectBufferAddress(env, Y);
    double *cAp = (double *) (*env)->GetDirectBufferAddress(env, Ap);
    cblas_dspr2(Order, Uplo, N, alpha, cX + offsetX, incX, cY + offsetY, incY, cAp);
};

/*
 * ======================================================
 * Level 3 BLAS procedures
 * ======================================================
 */


/*
 * ------------------------------------------------------
 * GEMM
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_sgemm
(JNIEnv *env, jclass clazz,
 jint Order, jint TransA, jint TransB,
 jint M, jint N, jint K,
 jfloat alpha,
 jobject A, jint offsetA, jint lda,
 jobject B, jint offsetB, jint ldb,
 jfloat beta,
 jobject C, jint offsetC, jint ldc) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cB = (float *) (*env)->GetDirectBufferAddress(env, B);
    float *cC = (float *) (*env)->GetDirectBufferAddress(env, C);
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha,
                cA + offsetA, lda, cB + offsetB, ldb, beta, cC + offsetC, ldc);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dgemm
(JNIEnv *env, jclass clazz,
 jint Order, jint TransA, jint TransB,
 jint M, jint N, jint K,
 jdouble alpha,
 jobject A, jint offsetA, jint lda,
 jobject B, jint offsetB, jint ldb,
 jdouble beta,
 jobject C, jint offsetC, jint ldc) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cB = (double *) (*env)->GetDirectBufferAddress(env, B);
    double *cC = (double *) (*env)->GetDirectBufferAddress(env, C);
    cblas_dgemm(Order, TransA, TransB, M, N, K, alpha,
                cA + offsetA, lda, cB + offsetB, ldb, beta, cC + offsetC, ldc);
};

/*
 * ------------------------------------------------------
 * SYMM
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_ssymm
(JNIEnv *env, jclass clazz,
 jint Order, jint Side, jint Uplo,
 jint M, jint N,
 jfloat alpha,
 jobject A, jint offsetA, jint lda,
 jobject B, jint offsetB, jint ldb,
 jfloat beta,
 jobject C, jint offsetC, jint ldc) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cB = (float *) (*env)->GetDirectBufferAddress(env, B);
    float *cC = (float *) (*env)->GetDirectBufferAddress(env, C);
    cblas_ssymm(Order, Side, Uplo, M, N, alpha,
                cA + offsetA, lda, cB + offsetB, ldb, beta, cC + offsetC, ldc);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dsymm
(JNIEnv *env, jclass clazz,
 jint Order, jint Side, jint Uplo,
 jint M, jint N,
 jdouble alpha,
 jobject A, jint offsetA, jint lda,
 jobject B, jint offsetB, jint ldb,
 jdouble beta,
 jobject C, jint offsetC, jint ldc) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cB = (double *) (*env)->GetDirectBufferAddress(env, B);
    double *cC = (double *) (*env)->GetDirectBufferAddress(env, C);
    cblas_dsymm(Order, Side, Uplo, M, N, alpha,
                cA + offsetA, lda, cB + offsetB, ldb, beta, cC + offsetC, ldc);
};

/*
 * ------------------------------------------------------
 * SYRK
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_ssyrk
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint Trans,
 jint N, jint K,
 jfloat alpha,
 jobject A, jint offsetA, jint lda,
 jfloat beta,
 jobject C, jint offsetC, jint ldc) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cC = (float *) (*env)->GetDirectBufferAddress(env, C);
    cblas_ssyrk(Order, Uplo, Trans, N, K, alpha, cA + offsetA, lda, beta, cC + offsetC, ldc);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dsyrk
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint Trans,
 jint N, jint K,
 jdouble alpha,
 jobject A, jint offsetA, jint lda,
 jfloat beta,
 jobject C, jint offsetC, jint ldc) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cC = (double *) (*env)->GetDirectBufferAddress(env, C);
    cblas_dsyrk(Order, Uplo, Trans, N, K, alpha, cA + offsetA, lda, beta, cC + offsetC, ldc);
};

/*
 * ------------------------------------------------------
 * SYR2K
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_ssyr2k
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint Trans,
 jint N, jint K,
 jfloat alpha,
 jobject A, jint offsetA, jint lda,
 jobject B, jint offsetB, jint ldb,
 jfloat beta,
 jobject C, jint offsetC, jint ldc) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cB = (float *) (*env)->GetDirectBufferAddress(env, B);
    float *cC = (float *) (*env)->GetDirectBufferAddress(env, C);
    cblas_ssyr2k(Order, Uplo, Trans, N, K, alpha,
                 cA + offsetA, lda, cB + offsetB, ldb, beta, cC + offsetC, ldc);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dsyr2k
(JNIEnv *env, jclass clazz,
 jint Order, jint Uplo, jint Trans,
 jint N, jint K,
 jdouble alpha,
 jobject A, jint offsetA, jint lda,
 jobject B, jint offsetB, jint ldb,
 jdouble beta,
 jobject C, jint offsetC, jint ldc) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cB = (double *) (*env)->GetDirectBufferAddress(env, B);
    double *cC = (double *) (*env)->GetDirectBufferAddress(env, C);
    cblas_dsyr2k(Order, Uplo, Trans, N, K, alpha,
                 cA + offsetA, lda, cB + offsetB, ldb, beta, cC + offsetC, ldc);
};

/*
 * ------------------------------------------------------
 * TRMM
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_strmm
(JNIEnv *env, jclass clazz,
 jint Order, jint Side,
 jint Uplo, jint TransA, jint Diag,
 jint M, jint N,
 jfloat alpha,
 jobject A, jint offsetA, jint lda,
 jobject B, jint offsetB, jint ldb) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cB = (float *) (*env)->GetDirectBufferAddress(env, B);
    cblas_strmm(Order, Side, Uplo, TransA, Diag, M, N, alpha,
                cA + offsetA, lda, cB + offsetB, ldb);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dtrmm
(JNIEnv *env, jclass clazz,
 jint Order, jint Side,
 jint Uplo, jint TransA, jint Diag,
 jint M, jint N,
 jdouble alpha,
 jobject A, jint offsetA, jint lda,
 jobject B, jint offsetB, jint ldb) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cB = (double *) (*env)->GetDirectBufferAddress(env, B);
    cblas_dtrmm(Order, Side, Uplo, TransA, Diag, M, N, alpha,
                cA + offsetA, lda, cB + offsetB, ldb);
};

/*
 * ------------------------------------------------------
 * TRSM
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_strsm
(JNIEnv *env, jclass clazz,
 jint Order, jint Side,
 jint Uplo, jint TransA, jint Diag,
 jint M, jint N,
 jfloat alpha,
 jobject A, jint offsetA, jint lda,
 jobject B, jint offsetB, jint ldb) {

    float *cA = (float *) (*env)->GetDirectBufferAddress(env, A);
    float *cB = (float *) (*env)->GetDirectBufferAddress(env, B);
    cblas_strsm(Order, Side, Uplo, TransA, Diag, M, N, alpha,
                cA + offsetA, lda, cB + offsetB, ldb);
};

JNIEXPORT void JNICALL Java_uncomplicate_neanderthal_CBLAS_dtrsm
(JNIEnv *env, jclass clazz,
 jint Order, jint Side,
 jint Uplo, jint TransA, jint Diag,
 jint M, jint N,
 jdouble alpha,
 jobject A, jint offsetA, jint lda,
 jobject B, jint offsetB, jint ldb) {

    double *cA = (double *) (*env)->GetDirectBufferAddress(env, A);
    double *cB = (double *) (*env)->GetDirectBufferAddress(env, B);
    cblas_dtrsm(Order, Side, Uplo, TransA, Diag, M, N, alpha,
                cA + offsetA, lda, cB + offsetB, ldb);
};
