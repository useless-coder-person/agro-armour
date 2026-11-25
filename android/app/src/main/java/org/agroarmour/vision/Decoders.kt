package org.agroarmour.vision

import kotlin.math.max
import kotlin.math.min

data class Detection(val cls: Int, val score: Float, val x1: Double, val y1: Double, val x2: Double, val y2: Double)

/**
 * Detector output decoders. Which one applies is read from io_spec.json ("decoder": "raw" | "e2e").
 * Defaults must match ml/export/make_golden.py (CONF, IOU, MAX_DET).
 */
object Decoders {
    const val CONF = 0.25f
    const val IOU = 0.5f
    const val MAX_DET = 50

    /** Raw one-to-many head, layout (1, 4+nc, A): cx, cy, w, h in input pixels, then per-class scores. */
    fun decodeRaw(
        out: FloatArray, nc: Int, lb: LetterboxInfo, srcW: Int, srcH: Int,
        conf: Float = CONF, iou: Float = IOU, maxDet: Int = MAX_DET,
    ): List<Detection> {
        val a = out.size / (4 + nc)
        require(a * (4 + nc) == out.size) { "output size ${out.size} is not (4+$nc)*A" }
        val cand = ArrayList<FloatArray>()  // x1 y1 x2 y2 score cls
        for (i in 0 until a) {
            var best = 0
            var bs = out[4 * a + i]
            for (c in 1 until nc) {
                val s = out[(4 + c) * a + i]
                if (s > bs) { bs = s; best = c }  // strict > keeps the first max, like numpy argmax
            }
            if (bs < conf) continue
            val cx = out[i]; val cy = out[a + i]; val w = out[2 * a + i]; val h = out[3 * a + i]
            cand.add(floatArrayOf(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, bs, best.toFloat()))
        }
        return finish(nms(cand, iou, maxDet), lb, srcW, srcH)
    }

    /** End-to-end head, layout (1, N, 6): x1 y1 x2 y2 score cls in input pixels, already NMS-free. */
    fun decodeE2e(out: FloatArray, lb: LetterboxInfo, srcW: Int, srcH: Int, conf: Float = CONF, maxDet: Int = MAX_DET): List<Detection> {
        val rows = (0 until out.size / 6).map { out.copyOfRange(it * 6, it * 6 + 6) }
            .filter { it[4] >= conf }.sortedByDescending { it[4] }.take(maxDet)
        return finish(rows, lb, srcW, srcH)
    }

    /** Class-wise greedy NMS; candidates sorted by score descending, stable. */
    fun nms(cand: List<FloatArray>, iou: Float, maxDet: Int): List<FloatArray> {
        val sorted = cand.withIndex().sortedWith(compareBy({ -it.value[4] }, { it.index })).map { it.value }
        val kept = ArrayList<FloatArray>()
        for (b in sorted) {
            if (kept.any { it[5] == b[5] && iou(it, b) > iou }) continue
            kept.add(b)
            if (kept.size == maxDet) break
        }
        return kept
    }

    fun iou(a: FloatArray, b: FloatArray): Float {
        val iw = max(0f, min(a[2], b[2]) - max(a[0], b[0]))
        val ih = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
        val inter = iw * ih
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9f)
    }

    private fun finish(rows: List<FloatArray>, lb: LetterboxInfo, w: Int, h: Int) = rows.map {
        fun x(v: Float) = ((v - lb.padLeft) / lb.ratio).coerceIn(0.0, w.toDouble())
        fun y(v: Float) = ((v - lb.padTop) / lb.ratio).coerceIn(0.0, h.toDouble())
        Detection(it[5].toInt(), it[4], x(it[0]), y(it[1]), x(it[2]), y(it[3]))
    }
}
