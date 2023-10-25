/*********************************************************************************/
/*                                                                               */
/*  vp2intersect.h                                                               */
/*                                                                               */
/*  Include file for emulation of AVX512 VP2INTERSECT instructions.              */
/*  This is not a strict drop-in replacement for VP2INTERSECT instructions.      */
/*  Instead, the functions in this code return only the first output mask        */
/*  of the corresponding instruction in VP2INTERSECT, which is suficient to      */
/*  to compute intersections of long vectors of integers, or to compute the      */
/*  size of the intersection of long vectors of integers.                        */
/*                                                                               */
/*  April 11, 2022                                                               */
/*                                                                               */
/*  Copyright 2022                                                               */
/*  Guillermo Diez Canas                                                         */
/*  guille@berkeley.edu                                                      */
/*                                                                               */
/*  This program may be freely redistributed under the condition that the        */
/*  copyright notices (including this entire header) are not removed, and        */
/*  no compensation is received.  Private, research, and institutional           */
/*  use is free.  You may distribute modified versions of this code UNDER        */
/*  THE CONDITION THAT THIS CODE AND ANY MODIFICATIONS MADE TO IT IN THE         */
/*  SAME FILE REMAIN UNDER COPYRIGHT OF THE ORIGINAL AUTHOR, BOTH SOURCE         */
/*  AND OBJECT CODE ARE MADE FREELY AVAILABLE WITHOUT CHARGE, AND CLEAR          */
/*  NOTICE IS GIVEN OF THE MODIFICATIONS.  Distribution of this code as          */
/*  part of a commercial system is permissible ONLY BY DIRECT ARRANGEMENT        */
/*  WITH THE AUTHOR.  (If you are not directly supplying this code to a          */
/*  customer, and you are instead telling them how they can obtain it for        */
/*  free, then you are not required to make any arrangement with me.)            */
/*                                                                               */
/*********************************************************************************/
/*                                                                               */
/*  Each function is equivalent to the correspondingly named AVX512              */
/*  VP2INTERSECT instruction, but only returns the first output mask.            */
/*                                                                               */
/*********************************************************************************/
#ifndef __VP2INTERSECT_H__
#define __VP2INTERSECT_H__
#ifdef __SSE2__
#include <cstdint>
#include <immintrin.h>


inline uint32_t vp2i_rol32(const uint32_t x, const int n)				{ return (x << n) | (x >> (32 - n)); }
inline uint16_t vp2i_rol16(const uint16_t x, const int n)				{ return (x << n) | (x >> (16 - n)); }
inline uint8_t vp2i_rol8(const uint8_t x, const int n)					{ return (x << n) | (x >> (8 - n)); }
inline uint8_t vp2i_rol4(const uint8_t x, const int n)					{ return (x << n) | (x >> (4 - n)); }

inline uint32_t vp2i_ror32(const uint32_t x, const int n)				{ return (x >> n) | (x << (32 - n)); }
inline uint16_t vp2i_ror16(const uint16_t x, const int n)				{ return (x >> n) | (x << (16 - n)); }
inline uint8_t vp2i_ror8(const uint8_t x, const int n)					{ return (x >> n) | (x << (8 - n)); }
inline uint8_t vp2i_ror4(const uint8_t x, const int n)					{ return (x >> n) | (x << (4 - n)); }


#ifdef __AVX512F__
inline uint32_t _mm512_2intersect_epi16_mask(__m512i a, __m512i b)
{
	__m512i a1 = _mm512_alignr_epi32(a, a,  4);
	__m512i a2 = _mm512_alignr_epi32(a, a,  8);
	__m512i a3 = _mm512_alignr_epi32(a, a, 12);

	__m512i b1 = _mm512_shuffle_epi32(b, _MM_PERM_ADCB);
	__m512i b2 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
	__m512i b3 = _mm512_shuffle_epi32(b, _MM_PERM_CBAD);

#ifdef __AVX512VBMI2__
	__m512i b01 = _mm512_shrdi_epi32(b , b , 16);
	__m512i b11 = _mm512_shrdi_epi32(b1, b1, 16);
	__m512i b21 = _mm512_shrdi_epi32(b2, b2, 16);
	__m512i b31 = _mm512_shrdi_epi32(b3, b3, 16);
#else
	__m512i b01 = _mm512_or_si512(_mm512_srli_epi32(b, 16), _mm512_slli_epi32(b, 16));
	__m512i b11 = _mm512_shuffle_epi32(b01, _MM_PERM_ADCB);
	__m512i b21 = _mm512_shuffle_epi32(b01, _MM_PERM_BADC);
	__m512i b31 = _mm512_shuffle_epi32(b01, _MM_PERM_CBAD);
#endif
	auto nm00 = _mm512_cmpneq_epi16_mask(a , b);
	auto nm01 = _mm512_cmpneq_epi16_mask(a1, b);
	auto nm02 = _mm512_cmpneq_epi16_mask(a2, b);
	auto nm03 = _mm512_cmpneq_epi16_mask(a3, b);

	auto nm10 = _mm512_mask_cmpneq_epi16_mask(nm00, a , b01);
	auto nm11 = _mm512_mask_cmpneq_epi16_mask(nm01, a1, b01);
	auto nm12 = _mm512_mask_cmpneq_epi16_mask(nm02, a2, b01);
	auto nm13 = _mm512_mask_cmpneq_epi16_mask(nm03, a3, b01);

	auto nm20 = _mm512_mask_cmpneq_epi16_mask(nm10, a , b1);
	auto nm21 = _mm512_mask_cmpneq_epi16_mask(nm11, a1, b1);
	auto nm22 = _mm512_mask_cmpneq_epi16_mask(nm12, a2, b1);
	auto nm23 = _mm512_mask_cmpneq_epi16_mask(nm13, a3, b1);

	auto nm30 = _mm512_mask_cmpneq_epi16_mask(nm20, a , b11);
	auto nm31 = _mm512_mask_cmpneq_epi16_mask(nm21, a1, b11);
	auto nm32 = _mm512_mask_cmpneq_epi16_mask(nm22, a2, b11);
	auto nm33 = _mm512_mask_cmpneq_epi16_mask(nm23, a3, b11);

	auto nm40 = _mm512_mask_cmpneq_epi16_mask(nm30, a , b2);
	auto nm41 = _mm512_mask_cmpneq_epi16_mask(nm31, a1, b2);
	auto nm42 = _mm512_mask_cmpneq_epi16_mask(nm32, a2, b2);
	auto nm43 = _mm512_mask_cmpneq_epi16_mask(nm33, a3, b2);

	auto nm50 = _mm512_mask_cmpneq_epi16_mask(nm40, a , b21);
	auto nm51 = _mm512_mask_cmpneq_epi16_mask(nm41, a1, b21);
	auto nm52 = _mm512_mask_cmpneq_epi16_mask(nm42, a2, b21);
	auto nm53 = _mm512_mask_cmpneq_epi16_mask(nm43, a3, b21);

	auto nm60 = _mm512_mask_cmpneq_epi16_mask(nm50, a , b3);
	auto nm61 = _mm512_mask_cmpneq_epi16_mask(nm51, a1, b3);
	auto nm62 = _mm512_mask_cmpneq_epi16_mask(nm52, a2, b3);
	auto nm63 = _mm512_mask_cmpneq_epi16_mask(nm53, a3, b3);

	auto nm70 = _mm512_mask_cmpneq_epi16_mask(nm60, a , b31);
	auto nm71 = _mm512_mask_cmpneq_epi16_mask(nm61, a1, b31);
	auto nm72 = _mm512_mask_cmpneq_epi16_mask(nm62, a2, b31);
	auto nm73 = _mm512_mask_cmpneq_epi16_mask(nm63, a3, b31);

	return ~(uint32_t)(nm70 & vp2i_rol32(nm71, 8) & vp2i_rol32(nm72, 16) & vp2i_ror32(nm73, 8));
}

inline uint16_t _mm512_2intersect_epi32_mask(__m512i a, __m512i b)
{
	__m512i a1 = _mm512_alignr_epi32(a, a, 4);
	__m512i b1 = _mm512_shuffle_epi32(b, _MM_PERM_ADCB);
	auto nm00 = _mm512_cmpneq_epi32_mask(a, b);

	__m512i a2 = _mm512_alignr_epi32(a, a, 8);
	__m512i a3 = _mm512_alignr_epi32(a, a, 12);
	auto nm01 = _mm512_cmpneq_epi32_mask(a1, b);
	auto nm02 = _mm512_cmpneq_epi32_mask(a2, b);

	auto nm03 = _mm512_cmpneq_epi32_mask(a3, b);
	auto nm10 = _mm512_mask_cmpneq_epi32_mask(nm00, a , b1);
	auto nm11 = _mm512_mask_cmpneq_epi32_mask(nm01, a1, b1);

	__m512i b2 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
	auto nm12 = _mm512_mask_cmpneq_epi32_mask(nm02, a2, b1);
	auto nm13 = _mm512_mask_cmpneq_epi32_mask(nm03, a3, b1);
	auto nm20 = _mm512_mask_cmpneq_epi32_mask(nm10, a , b2);

	__m512i b3 = _mm512_shuffle_epi32(b, _MM_PERM_CBAD);
	auto nm21 = _mm512_mask_cmpneq_epi32_mask(nm11, a1, b2);
	auto nm22 = _mm512_mask_cmpneq_epi32_mask(nm12, a2, b2);
	auto nm23 = _mm512_mask_cmpneq_epi32_mask(nm13, a3, b2);

	auto nm0 = _mm512_mask_cmpneq_epi32_mask(nm20, a , b3);
	auto nm1 = _mm512_mask_cmpneq_epi32_mask(nm21, a1, b3);
	auto nm2 = _mm512_mask_cmpneq_epi32_mask(nm22, a2, b3);
	auto nm3 = _mm512_mask_cmpneq_epi32_mask(nm23, a3, b3);

	return ~(uint16_t)(nm0 & vp2i_rol16(nm1, 4) & vp2i_rol16(nm2, 8) & vp2i_ror16(nm3, 4));
}

inline uint16_t _mm512_2intersect_epi32_mask(__m512i a, int32_t * const b)
{
	auto m00 = _mm512_cmpneq_epi32_mask(a, _mm512_set1_epi32(b[0]));
	auto m01 = _mm512_cmpneq_epi32_mask(a, _mm512_set1_epi32(b[1]));
	auto m02 = _mm512_cmpneq_epi32_mask(a, _mm512_set1_epi32(b[2]));

	auto m03 = _mm512_mask_cmpneq_epi32_mask(m00, a, _mm512_set1_epi32(b[ 3]));
	auto m04 = _mm512_mask_cmpneq_epi32_mask(m01, a, _mm512_set1_epi32(b[ 4]));
	auto m05 = _mm512_mask_cmpneq_epi32_mask(m02, a, _mm512_set1_epi32(b[ 5]));
	auto m06 = _mm512_mask_cmpneq_epi32_mask(m03, a, _mm512_set1_epi32(b[ 6]));
	auto m07 = _mm512_mask_cmpneq_epi32_mask(m04, a, _mm512_set1_epi32(b[ 7]));
	auto m08 = _mm512_mask_cmpneq_epi32_mask(m05, a, _mm512_set1_epi32(b[ 8]));
	auto m09 = _mm512_mask_cmpneq_epi32_mask(m06, a, _mm512_set1_epi32(b[ 9]));
	auto m10 = _mm512_mask_cmpneq_epi32_mask(m07, a, _mm512_set1_epi32(b[10]));
	auto m11 = _mm512_mask_cmpneq_epi32_mask(m08, a, _mm512_set1_epi32(b[11]));
	auto m12 = _mm512_mask_cmpneq_epi32_mask(m09, a, _mm512_set1_epi32(b[12]));
	auto m13 = _mm512_mask_cmpneq_epi32_mask(m10, a, _mm512_set1_epi32(b[13]));
	auto m14 = _mm512_mask_cmpneq_epi32_mask(m11, a, _mm512_set1_epi32(b[14]));
	auto m15 = _mm512_mask_cmpneq_epi32_mask(m12, a, _mm512_set1_epi32(b[15]));

	return ~(uint16_t)(m13 & m14 & m15);
}

inline uint8_t _mm512_2intersect_epi64_mask(__m512i a, __m512i b)
{
	__m512i a1 = _mm512_alignr_epi32(a, a, 4);
	__m512i b1 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
	__m512i a2 = _mm512_alignr_epi32(a, a, 8);
	auto m0 = _mm512_mask_cmpneq_epi64_mask(_mm512_cmpneq_epi64_mask(a , b), a , b1);
	__m512i a3 = _mm512_alignr_epi32(a, a, 12);
	auto m1 = _mm512_mask_cmpneq_epi64_mask(_mm512_cmpneq_epi64_mask(a1, b), a1, b1);
	auto m2 = _mm512_mask_cmpneq_epi64_mask(_mm512_cmpneq_epi64_mask(a2, b), a2, b1);
	auto m3 = _mm512_mask_cmpneq_epi64_mask(_mm512_cmpneq_epi64_mask(a3, b), a3, b1);

	return ~(uint8_t)(m0 & vp2i_rol8(m1, 2) & vp2i_rol8(m2, 4) & vp2i_ror8(m3, 2));
}
#endif





#if defined(__AVX512F__) && defined(__AVX512VL__)
inline uint16_t _mm256_2intersect_epi16_mask(__m256i a, __m256i b)
{
	__m256i a1 = _mm256_alignr_epi32(a, a, 4);
	__m256i b1 = _mm256_shuffle_epi32(b, _MM_PERM_ADCB);
	__m256i b2 = _mm256_shuffle_epi32(b, _MM_PERM_BADC);
	__m256i b3 = _mm256_shuffle_epi32(b, _MM_PERM_CBAD);
#ifdef __AVX512VBMI2__
	__m256i b01 = _mm256_shrdi_epi32(b , b , 16);
	__m256i b11 = _mm256_shrdi_epi32(b1, b1, 16);
	__m256i b21 = _mm256_shrdi_epi32(b2, b2, 16);
	__m256i b31 = _mm256_shrdi_epi32(b3, b3, 16);
#else
	__m256i b01 = _mm256_or_si256(_mm256_srli_epi32(b, 16), _mm256_slli_epi32(b, 16));
	__m256i b11 = _mm256_shuffle_epi32(b01, _MM_PERM_ADCB);
	__m256i b21 = _mm256_shuffle_epi32(b01, _MM_PERM_BADC);
	__m256i b31 = _mm256_shuffle_epi32(b01, _MM_PERM_CBAD);
#endif
	auto nm00 = _mm256_cmpneq_epi16_mask(a , b);
	auto nm01 = _mm256_cmpneq_epi16_mask(a1, b);

	auto nm10 = _mm256_mask_cmpneq_epi16_mask(nm00, a , b01);
	auto nm11 = _mm256_mask_cmpneq_epi16_mask(nm01, a1, b01);

	auto nm20 = _mm256_mask_cmpneq_epi16_mask(nm10, a , b1);
	auto nm21 = _mm256_mask_cmpneq_epi16_mask(nm11, a1, b1);

	auto nm30 = _mm256_mask_cmpneq_epi16_mask(nm20, a , b11);
	auto nm31 = _mm256_mask_cmpneq_epi16_mask(nm21, a1, b11);

	auto nm40 = _mm256_mask_cmpneq_epi16_mask(nm30, a , b2);
	auto nm41 = _mm256_mask_cmpneq_epi16_mask(nm31, a1, b2);

	auto nm50 = _mm256_mask_cmpneq_epi16_mask(nm40, a , b21);
	auto nm51 = _mm256_mask_cmpneq_epi16_mask(nm41, a1, b21);

	auto nm60 = _mm256_mask_cmpneq_epi16_mask(nm50, a , b3);
	auto nm61 = _mm256_mask_cmpneq_epi16_mask(nm51, a1, b3);

	auto nm70 = _mm256_mask_cmpneq_epi16_mask(nm60, a , b31);
	auto nm71 = _mm256_mask_cmpneq_epi16_mask(nm61, a1, b31);

	return ~(uint16_t)(nm70 & vp2i_rol16(nm71, 8));
}

inline uint8_t _mm256_2intersect_epi32_mask(__m256i a, __m256i b)
{
	__m256i a1 = _mm256_alignr_epi32(a, a, 4);
	__m256i b1 = _mm256_shuffle_epi32(b, _MM_PERM_ADCB);
	__m256i b2 = _mm256_shuffle_epi32(b, _MM_PERM_BADC);
	__m256i b3 = _mm256_shuffle_epi32(b, _MM_PERM_CBAD);
	auto nm00 = _mm256_cmpneq_epi32_mask(a, b);
	auto nm01 = _mm256_cmpneq_epi32_mask(a1, b);
	auto nm10 = _mm256_mask_cmpneq_epi32_mask(nm00, a , b1);
	auto nm11 = _mm256_mask_cmpneq_epi32_mask(nm01, a1, b1);
	auto nm20 = _mm256_mask_cmpneq_epi32_mask(nm10, a , b2);
	auto nm21 = _mm256_mask_cmpneq_epi32_mask(nm11, a1, b2);
	auto nm0 = _mm256_mask_cmpneq_epi32_mask(nm20, a , b3);
	auto nm1 = _mm256_mask_cmpneq_epi32_mask(nm21, a1, b3);
	return ~(uint8_t)(nm0 & vp2i_rol8(nm1, 4));
}

inline uint8_t _mm256_2intersect_epi64_mask(__m256i a, __m256i b)
{
	__m256i a1 = _mm256_alignr_epi32(a, a, 4);
	__m256i b1 = _mm256_shuffle_epi32(b, _MM_PERM_BADC);
	auto m0 = _mm256_mask_cmpneq_epi64_mask(_mm256_cmpneq_epi64_mask(a , b), a , b1);
	auto m1 = _mm256_mask_cmpneq_epi64_mask(_mm256_cmpneq_epi64_mask(a1, b), a1, b1);
	return 0xf ^ (m0 & vp2i_rol4(m1, 2));
}


#elif defined(__AVX2__)
inline __m256i _mm256_2intersect_epi16_mask(__m256i a, __m256i b)
{
	__m256i a1 = _mm256_permute2x128_si256(a, a, 1);
	__m256i b1 = _mm256_shuffle_epi32(b, _MM_PERM_ADCB);
	__m256i b2 = _mm256_shuffle_epi32(b, _MM_PERM_BADC);
	__m256i b3 = _mm256_shuffle_epi32(b, _MM_PERM_CBAD);
	__m256i b01 = _mm256_or_si256(_mm256_srli_epi32(b, 16), _mm256_slli_epi32(b, 16));
	__m256i b11 = _mm256_shuffle_epi32(b01, _MM_PERM_ADCB);
	__m256i b21 = _mm256_shuffle_epi32(b01, _MM_PERM_BADC);
	__m256i b31 = _mm256_shuffle_epi32(b01, _MM_PERM_CBAD);

	__m256i l1l = _mm256_or_si256(_mm256_cmpeq_epi16(a , b ), _mm256_cmpeq_epi16(a , b01));
	__m256i l1h = _mm256_or_si256(_mm256_cmpeq_epi16(a , b1), _mm256_cmpeq_epi16(a , b11));
	__m256i l2l = _mm256_or_si256(_mm256_cmpeq_epi16(a , b2), _mm256_cmpeq_epi16(a , b21));
	__m256i l2h = _mm256_or_si256(_mm256_cmpeq_epi16(a , b3), _mm256_cmpeq_epi16(a , b31));
	__m256i h1l = _mm256_or_si256(_mm256_cmpeq_epi16(a1, b ), _mm256_cmpeq_epi16(a1, b01));
	__m256i h1h = _mm256_or_si256(_mm256_cmpeq_epi16(a1, b1), _mm256_cmpeq_epi16(a1, b11));
	__m256i h2l = _mm256_or_si256(_mm256_cmpeq_epi16(a1, b2), _mm256_cmpeq_epi16(a1, b21));
	__m256i h2h = _mm256_or_si256(_mm256_cmpeq_epi16(a1, b3), _mm256_cmpeq_epi16(a1, b31));
	__m256i l1 = _mm256_or_si256(l1l, l1h);
	__m256i l2 = _mm256_or_si256(l2l, l2h);
	__m256i h1 = _mm256_or_si256(h1l, h1h);
	__m256i h2 = _mm256_or_si256(h2l, h2h);
	__m256i l = _mm256_or_si256(l1, l2);
	__m256i h = _mm256_or_si256(h1, h2);
	return _mm256_or_si256(l, _mm256_permute2x128_si256(h, h, 1));
}

inline __m256i _mm256_2intersect_epi32_mask(__m256i a, __m256i b)
{
	__m256i a1 = _mm256_permute2x128_si256(a, a, 1);
	__m256i b1 = _mm256_shuffle_epi32(b, _MM_PERM_ADCB);
	__m256i b2 = _mm256_shuffle_epi32(b, _MM_PERM_BADC);
	__m256i b3 = _mm256_shuffle_epi32(b, _MM_PERM_CBAD);

	__m256i ll = _mm256_or_si256(_mm256_cmpeq_epi32(a , b ), _mm256_cmpeq_epi32(a , b1));
	__m256i lh = _mm256_or_si256(_mm256_cmpeq_epi32(a , b2), _mm256_cmpeq_epi32(a , b3));
	__m256i hl = _mm256_or_si256(_mm256_cmpeq_epi32(a1, b ), _mm256_cmpeq_epi32(a1, b1));
	__m256i hh = _mm256_or_si256(_mm256_cmpeq_epi32(a1, b2), _mm256_cmpeq_epi32(a1, b3));
	__m256i l = _mm256_or_si256(ll, lh);
	__m256i h = _mm256_or_si256(hl, hh);
	return _mm256_or_si256(l, _mm256_permute2x128_si256(h, h, 1));
}

inline __m256i _mm256_2intersect_epi64_mask(__m256i a, __m256i b)
{
	__m256i a1 = _mm256_permute2x128_si256(a, a, 1);
	__m256i b1 = _mm256_shuffle_epi32(b, _MM_PERM_BADC);
	__m256i l = _mm256_or_si256(_mm256_cmpeq_epi64(a , b), _mm256_cmpeq_epi64(a , b1));
	__m256i h = _mm256_or_si256(_mm256_cmpeq_epi64(a1, b), _mm256_cmpeq_epi64(a1, b1));
	return _mm256_or_si256(l, _mm256_permute2x128_si256(h, h, 1));
}
#endif





inline auto _mm_2intersect_epi16_mask(__m128i a, __m128i b)
{
	__m128i b1 = _mm_shuffle_epi32(b, _MM_PERM_ADCB);
	__m128i b2 = _mm_shuffle_epi32(b, _MM_PERM_BADC);
	__m128i b3 = _mm_shuffle_epi32(b, _MM_PERM_CBAD);
#if defined(__AVX512VL__) && defined(__AVX512VBMI2__)
	__m128i b01 = _mm_shrdi_epi32(b , b , 16);
	__m128i b11 = _mm_shrdi_epi32(b1, b1, 16);
	__m128i b21 = _mm_shrdi_epi32(b2, b2, 16);
	__m128i b31 = _mm_shrdi_epi32(b3, b3, 16);
#else
	__m128i b01 = _mm_or_si128(_mm_srli_epi32(b, 16), _mm_slli_epi32(b, 16));
	__m128i b11 = _mm_shuffle_epi32(b01, _MM_PERM_ADCB);
	__m128i b21 = _mm_shuffle_epi32(b01, _MM_PERM_BADC);
	__m128i b31 = _mm_shuffle_epi32(b01, _MM_PERM_CBAD);
#endif
#if defined(__AVX512F__) && defined(__AVX512VL__)
	auto nm0 = _mm_cmpneq_epi16_mask(a, b);
	auto nm1 = _mm_cmpneq_epi16_mask(a, b01);
	auto nm2 = _mm_mask_cmpneq_epi16_mask(nm0, a, b1);
	auto nm3 = _mm_mask_cmpneq_epi16_mask(nm1, a, b11);
	auto nm4 = _mm_mask_cmpneq_epi16_mask(nm2, a, b2);
	auto nm5 = _mm_mask_cmpneq_epi16_mask(nm3, a, b21);
	auto nm6 = _mm_mask_cmpneq_epi16_mask(nm4, a, b3);
	auto nm7 = _mm_mask_cmpneq_epi16_mask(nm5, a, b31);
	return (uint8_t)~(uint8_t)(nm6 & nm7);
#else
	__m128i l = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi16(a, b  ), _mm_cmpeq_epi16(a, b1 )), _mm_or_si128(_mm_cmpeq_epi16(a, b2 ), _mm_cmpeq_epi16(a, b3 )));
	__m128i h = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi16(a, b01), _mm_cmpeq_epi16(a, b11)), _mm_or_si128(_mm_cmpeq_epi16(a, b21), _mm_cmpeq_epi16(a, b31)));
	return _mm_or_si128(l, h);
#endif
}


inline auto _mm_2intersect_epi32_mask(__m128i a, __m128i b)
{
	__m128i b1 = _mm_shuffle_epi32(b, _MM_PERM_ADCB);
	__m128i b2 = _mm_shuffle_epi32(b, _MM_PERM_BADC);
	__m128i b3 = _mm_shuffle_epi32(b, _MM_PERM_CBAD);
#if defined(__AVX512F__) && defined(__AVX512VL__)
	auto nm00 = _mm_cmpneq_epi32_mask(a, b);
	auto nm10 = _mm_mask_cmpneq_epi32_mask(nm00, a, b1);
	auto nm20 = _mm_mask_cmpneq_epi32_mask(nm10, a, b2);
	auto nm0  = _mm_mask_cmpneq_epi32_mask(nm20, a, b3);
	return 0xf ^ nm0;
#else
	return _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi32(a, b), _mm_cmpeq_epi32(a, b1)), _mm_or_si128(_mm_cmpeq_epi32(a, b2), _mm_cmpeq_epi32(a, b3)));
#endif
}


inline auto _mm_2intersect_epi64_mask(__m128i a, __m128i b)
{
	__m128i b1 = _mm_shuffle_epi32(b, _MM_PERM_BADC);
#if defined(__AVX512F__) && defined(__AVX512VL__)
	auto m0 = _mm_cmpeq_epi64_mask(a, b);
	auto m1 = _mm_cmpeq_epi64_mask(a, b1);
	return m0 | m1;
#elif defined(__SSE4_1__)
	return _mm_or_si128(_mm_cmpeq_epi64(a, b), _mm_cmpeq_epi64(a, b1));
#else
	__m128i tl = _mm_cmpeq_epi32(a, b ), l = _mm_and_si128(tl, _mm_shuffle_epi32(tl, _MM_PERM_CDAB));
	__m128i th = _mm_cmpeq_epi32(a, b1), h = _mm_and_si128(th, _mm_shuffle_epi32(th, _MM_PERM_CDAB));
	return _mm_or_si128(l, h);
#endif
}


#endif// __SSE2__
#endif// __VP2INTERSECT_H__
