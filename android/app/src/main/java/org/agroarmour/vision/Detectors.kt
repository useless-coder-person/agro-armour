package org.agroarmour.vision

import android.graphics.Bitmap
import org.json.JSONObject
import java.io.Closeable
import java.io.File

/** Pest detector: letterbox 640 -> .pte -> decoder named in io_spec.json. */
class PestDetector(dir: File, threads: Int = 0) : Closeable {
    private val spec = JSONObject(File(dir, "io_spec.json").readText())
    private val nc = spec.getInt("nc")
    private val decoder = spec.getString("decoder")
    private val size = spec.getJSONArray("input").getInt(2)
    private val model = VisionModel(File(dir, "model.pte").path, threads)

    fun detect(bmp: Bitmap): List<Detection> {
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h).also { bmp.getPixels(it, 0, w, 0, 0, w, h) }
        val (rgb, lb) = Preprocess.letterboxRgb(px, w, h, size)
        val (out, _) = model.run(Preprocess.toNchw(rgb, size), longArrayOf(1, 3, size.toLong(), size.toLong()))
        // Cut-off below the lowest MEDIUM band; Fusion applies the per-group bands.
        return if (decoder == "raw") Decoders.decodeRaw(out, nc, lb, w, h, conf = 0.1f)
        else Decoders.decodeE2e(out, lb, w, h, conf = 0.1f)
    }

    override fun close() = model.close()
}

/** Leaf-disease classifier; the .pte ends in softmax, so the output is already probabilities. */
class DiseaseClassifier(dir: File, threads: Int = 0) : Closeable {
    private val size = JSONObject(File(dir, "io_spec.json").readText()).getJSONArray("input").getInt(2)
    private val model = VisionModel(File(dir, "model.pte").path, threads)

    fun classify(bmp: Bitmap): FloatArray {
        val crop = ClassifierPreprocess.resizeCenterCrop(bmp, size)
        val px = IntArray(size * size).also { crop.getPixels(it, 0, size, 0, 0, size, size) }
        val hwc = IntArray(px.size * 3) { (px[it / 3] shr (16 - 8 * (it % 3))) and 0xFF }
        return model.run(Preprocess.toNchw(hwc, size), longArrayOf(1, 3, size.toLong(), size.toLong())).first
    }

    override fun close() = model.close()
}

object ClassifierPreprocess {
    /**
     * Ultralytics classify_transforms: Resize(short side = size, bilinear, antialiased on PIL) -> CenterCrop -> /255,
     * mean 0, std 1. Android's bilinear filter does not antialias, so shrink by halves first (cheap box filter),
     * then a final bilinear step. Checked by top-1 agreement on device, not pixel parity.
     */
    fun resizeCenterCrop(src: Bitmap, size: Int): Bitmap {
        var b = src
        while (minOf(b.width, b.height) >= 2 * size) b = Bitmap.createScaledBitmap(b, b.width / 2, b.height / 2, true)
        val s = size.toFloat() / minOf(b.width, b.height)
        val rw = maxOf(size, Math.round(b.width * s)); val rh = maxOf(size, Math.round(b.height * s))
        b = Bitmap.createScaledBitmap(b, rw, rh, true)
        return Bitmap.createBitmap(b, (rw - size) / 2, (rh - size) / 2, size, size)
    }
}
