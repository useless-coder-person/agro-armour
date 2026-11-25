package org.agroarmour.vision

import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

/** Letterbox geometry. Mirrors ultralytics LetterBox(auto=False, center=True) and ml/export/make_golden.py. */
data class LetterboxInfo(val ratio: Double, val padLeft: Int, val padTop: Int, val newW: Int, val newH: Int, val size: Int)

/**
 * Pure-Kotlin preprocessing on ARGB_8888 pixel arrays (Bitmap.getPixels layout), so it runs in JVM unit tests.
 * Resize reproduces OpenCV's INTER_LINEAR fixed-point path for 8-bit images (what Ultralytics uses).
 */
object Preprocess {
    const val PAD = 114
    private const val COEF_BITS = 11
    private const val COEF_SCALE = 1 shl COEF_BITS

    fun letterboxParams(w: Int, h: Int, size: Int = 640): LetterboxInfo {
        val r = min(size.toDouble() / h, size.toDouble() / w)
        val nw = max(round(w * r).toInt(), 1)  // kotlin.math.round = ties-to-even, same as Python round()
        val nh = max(round(h * r).toInt(), 1)
        val dw = (size - nw) / 2.0
        val dh = (size - nh) / 2.0
        return LetterboxInfo(r, round(dw - 0.1).toInt(), round(dh - 0.1).toInt(), nw, nh, size)
    }

    /** ARGB pixels -> letterboxed RGB, HWC, size*size*3 values 0..255. */
    fun letterboxRgb(argb: IntArray, w: Int, h: Int, size: Int = 640): Pair<IntArray, LetterboxInfo> {
        val lb = letterboxParams(w, h, size)
        val resized = if (w == lb.newW && h == lb.newH) argb else resizeLinear(argb, w, h, lb.newW, lb.newH)
        val out = IntArray(size * size * 3) { PAD }
        for (y in 0 until lb.newH) {
            val dst = ((y + lb.padTop) * size + lb.padLeft) * 3
            for (x in 0 until lb.newW) {
                val p = resized[y * lb.newW + x]
                val o = dst + x * 3
                out[o] = (p shr 16) and 0xFF
                out[o + 1] = (p shr 8) and 0xFF
                out[o + 2] = p and 0xFF
            }
        }
        return out to lb
    }

    /** HWC 0..255 -> NCHW float [0,1]. */
    fun toNchw(rgb: IntArray, size: Int): FloatArray {
        val plane = size * size
        val out = FloatArray(3 * plane)
        for (i in 0 until plane) {
            out[i] = rgb[i * 3] / 255f
            out[plane + i] = rgb[i * 3 + 1] / 255f
            out[2 * plane + i] = rgb[i * 3 + 2] / 255f
        }
        return out
    }

    private class Taps(val idx0: IntArray, val idx1: IntArray, val c0: IntArray, val c1: IntArray)

    private fun taps(dst: Int, src: Int): Taps {
        val scale = src.toDouble() / dst
        val i0 = IntArray(dst); val i1 = IntArray(dst); val c0 = IntArray(dst); val c1 = IntArray(dst)
        for (d in 0 until dst) {
            var f = ((d + 0.5) * scale - 0.5).toFloat()  // OpenCV computes this in float
            var s = floor(f).toInt()
            f -= s
            if (s < 0) { s = 0; f = 0f }
            if (s >= src - 1) { s = src - 1; f = 0f }
            val a0 = Math.rint(((1f - f) * COEF_SCALE).toDouble()).toInt()  // saturate_cast<short> = round half even
            i0[d] = s; i1[d] = min(s + 1, src - 1); c0[d] = a0; c1[d] = COEF_SCALE - a0
        }
        return Taps(i0, i1, c0, c1)
    }

    /** OpenCV INTER_LINEAR for 8-bit, per channel: horizontal int pass, then the 8U vertical fixed-point cast. */
    fun resizeLinear(argb: IntArray, w: Int, h: Int, nw: Int, nh: Int): IntArray {
        val tx = taps(nw, w)
        val ty = taps(nh, h)
        val out = IntArray(nw * nh)
        val row0 = IntArray(nw * 3)
        val row1 = IntArray(nw * 3)
        fun hpass(sy: Int, row: IntArray) {
            val base = sy * w
            for (x in 0 until nw) {
                val p = argb[base + tx.idx0[x]]
                val q = argb[base + tx.idx1[x]]
                val a = tx.c0[x]; val b = tx.c1[x]
                row[x * 3] = ((p shr 16) and 0xFF) * a + ((q shr 16) and 0xFF) * b
                row[x * 3 + 1] = ((p shr 8) and 0xFF) * a + ((q shr 8) and 0xFF) * b
                row[x * 3 + 2] = (p and 0xFF) * a + (q and 0xFF) * b
            }
        }
        for (y in 0 until nh) {
            hpass(ty.idx0[y], row0)
            hpass(ty.idx1[y], row1)
            val b0 = ty.c0[y]; val b1 = ty.c1[y]
            for (x in 0 until nw) {
                var px = 0xFF shl 24
                for (c in 0 until 3) {
                    val v = (((b0 * (row0[x * 3 + c] shr 4)) shr 16) + ((b1 * (row1[x * 3 + c] shr 4)) shr 16) + 2) shr 2
                    px = px or (v.coerceIn(0, 255) shl (16 - 8 * c))
                }
                out[y * nw + x] = px
            }
        }
        return out
    }
}
