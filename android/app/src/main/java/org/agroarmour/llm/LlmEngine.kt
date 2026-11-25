package org.agroarmour.llm

import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import org.json.JSONObject
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import org.pytorch.executorch.extension.llm.LlmModuleConfig
import java.io.Closeable
import java.io.File

data class LoadStats(val loadMs: Long)
data class GenStats(val ttftMs: Long, val promptTokens: Int, val genTokens: Int, val tokPerSec: Double, val raw: String)

/** The ExecuTorch LLM API is experimental; it is touched only in ExecuTorchLlmEngine. */
interface LlmEngine : Closeable {
    fun load(model: File, tokenizer: File): LoadStats   // blocking: call on the LLM dispatcher
    fun generate(prompt: String, maxNewTokens: Int, temperature: Float = 0.2f): Flow<String>  // cancel = stop()
    fun stats(): GenStats?
    fun unload()
    override fun close() = unload()
}

class ExecuTorchLlmEngine(private val numBos: Int = 0) : LlmEngine {
    private var module: LlmModule? = null
    private var last: GenStats? = null
    private var primed: String? = null  // system prompt already in the KV cache; the next generate() skips it

    /**
     * Constructs the runner without loading the weights. The constructor resets ExecuTorch's global threadpool to
     * (performant cores - 1); XNNPACK captures whatever pool exists when [load] runs. AppVm.ensureLlm sets the shared
     * thread count between the two calls.
     */
    fun create(model: File, tokenizer: File) {
        unload()
        module = LlmModule(
            LlmModuleConfig.create().modulePath(model.path).tokenizerPath(tokenizer.path)
                .modelType(LlmModuleConfig.MODEL_TYPE_TEXT).loadMode(LlmModuleConfig.LOAD_MODE_MMAP).build()
        )
    }

    override fun load(model: File, tokenizer: File): LoadStats {
        if (module == null) create(model, tokenizer)
        val t0 = System.nanoTime()
        module!!.load()
        return LoadStats((System.nanoTime() - t0) / 1_000_000)
    }

    /** Prefills [system] into a fresh KV cache ahead of time (blocking; LLM dispatcher). No-op if it is already there. */
    fun prime(system: String) {
        val m = module ?: return
        if (primed == system) return
        m.resetContext()
        m.prefillPrompt(system)
        primed = system
    }

    override fun generate(prompt: String, maxNewTokens: Int, temperature: Float): Flow<String> = callbackFlow {
        val m = checkNotNull(module) { "LLM not loaded" }
        // Builder defaults (AAR 1.5.1 bytecode): echo=true, temperature=0.8, seqLen=-1 (model max). numBos is
        // configurable so the smoke test can check for a doubled <|begin_of_text|>.
        val cfg = LlmGenerationConfig.create().echo(false).temperature(temperature).numBos(numBos)
            .maxNewTokens(maxNewTokens).build()
        val cb = object : LlmCallback {
            // A failed send means the collector is gone (advice cancelled): stop decoding now, not after maxNewTokens.
            override fun onResult(result: String) { if (trySend(result).isFailure) m.stop() }
            override fun onStats(stats: String) { last = parseStats(stats) }
            override fun onError(code: Int, message: String) { close(IllegalStateException("LLM error $code: $message")) }
        }
        val sys = primed
        primed = null
        val text = when {
            sys != null && prompt.startsWith(sys) -> prompt.substring(sys.length)
            // The runner continues from the previous KV-cache position unless reset; a self-contained prompt starts over.
            else -> { m.resetContext(); prompt }
        }
        last = null
        m.generate(text, cfg, cb)  // blocks until done or stop()
        close()
        awaitClose { m.stop() }
    }

    override fun stats() = last
    override fun unload() { module?.close(); module = null; primed = null }

    companion object {
        /** Stats JSON keys as emitted by the ExecuTorch runner (times in ms since epoch). */
        fun parseStats(s: String): GenStats {
            val j = JSONObject(s)
            val start = j.optLong("inference_start_ms")
            val first = j.optLong("first_token_ms")
            val end = j.optLong("inference_end_ms")
            val gen = j.optInt("generated_tokens")
            val decodeMs = (end - first).coerceAtLeast(1)
            return GenStats(first - start, j.optInt("prompt_tokens"), gen, (gen - 1).coerceAtLeast(0) * 1000.0 / decodeMs, s)
        }
    }
}
