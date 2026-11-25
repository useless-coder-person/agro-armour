package org.agroarmour.vision

import android.graphics.Bitmap
import org.json.JSONObject
import java.io.Closeable
import java.io.File

/** Per-pixel classes over the letterboxed input (0 background incl. other leaves, 1 main-leaf tissue, 2 lesion). */
class LesionMap(val ratio: Double?, val cls: ByteArray, val size: Int, val lb: LetterboxInfo)

/**
 * Learned lesion segmentation (ml/train/lesion_net.py); replaces the hue rules.
 * Same letterbox and [0,1] scaling as the pest detector; the model normalises internally.
 */
class LesionSegmenter(dir: File, threads: Int = 0) : Closeable {
    private val size = JSONObject(File(dir, "io_spec.json").readText()).getJSONArray("input").getInt(2)
    private val model = VisionModel(File(dir, "model.pte").path, threads)

    fun run(bmp: Bitmap): LesionMap {
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h).also { bmp.getPixels(it, 0, w, 0, 0, w, h) }
        val (rgb, lb) = Preprocess.letterboxRgb(px, w, h, size)
        val (logits, _) = model.run(Preprocess.toNchw(rgb, size), longArrayOf(1, 3, size.toLong(), size.toLong()))
        return LesionMath.map(logits, size, lb)
    }

    override fun close() = model.close()
}

/** Pure logic, unit-tested. */
object LesionMath {
    /** Below this share of the photo the main leaf is too small to judge; severity then comes from the field check. */
    const val MIN_LEAF = 0.03

    fun map(logits: FloatArray, size: Int, lb: LetterboxInfo): LesionMap {
        val plane = size * size
        val cls = ByteArray(plane)
        var leaf = 0; var lesion = 0; var n = 0
        for (y in lb.padTop until lb.padTop + lb.newH) for (x in lb.padLeft until lb.padLeft + lb.newW) {
            val i = y * size + x
            val a = logits[i]; val b = logits[plane + i]; val c = logits[2 * plane + i]
            val k = if (c > a && c > b) 2 else if (b > a) 1 else 0
            cls[i] = k.toByte(); n++
            if (k == 1) leaf++ else if (k == 2) lesion++
        }
        val ratio = if (leaf + lesion < MIN_LEAF * n) null else lesion.toDouble() / (leaf + lesion)
        return LesionMap(ratio, cls, size, lb)
    }
}
