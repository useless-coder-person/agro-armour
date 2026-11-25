package org.agroarmour.vision

import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.Closeable

/** Thin wrapper over the ExecuTorch Module API (pinned AAR 1.5.1) so API drift touches one file. */
class VisionModel(path: String, numThreads: Int = 0) : Closeable {
    private val module = Module.load(path, Module.LOAD_MODE_MMAP, numThreads)

    /** Single float32 NCHW input -> first output as a flat float array plus its shape. */
    fun run(input: FloatArray, shape: LongArray): Pair<FloatArray, LongArray> {
        val out = module.forward(EValue.from(Tensor.fromBlob(input, shape)))[0].toTensor()
        return out.dataAsFloatArray to out.shape()
    }

    override fun close() = module.destroy()
}
