package org.agroarmour.i18n

import android.content.Context
import android.speech.tts.TextToSpeech
import com.google.android.gms.tasks.Task
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.common.model.RemoteModelManager
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.TranslateRemoteModel
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import org.agroarmour.llm.OutputValidator
import java.util.Locale
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** Languages shipped in Phase 1. */
val LANGS = listOf("en", "hi", "ta")

suspend fun <T> Task<T>.await(): T = suspendCancellableCoroutine { k ->
    addOnSuccessListener { k.resume(it) }
    addOnFailureListener { k.resumeWithException(it) }
}

/**
 * One ML Kit on-device model per language: EN->L for advice, L->EN for farmer questions.
 * Only the chosen language's pack stays installed.
 */
class TranslatorManager {
    private var toL: Translator? = null
    private var toEn: Translator? = null
    var lang = "en"; private set

    private fun mlkit(l: String) = when (l) { "hi" -> TranslateLanguage.HINDI; "ta" -> TranslateLanguage.TAMIL; else -> TranslateLanguage.ENGLISH }

    suspend fun isDownloaded(l: String) = l == "en" ||
        RemoteModelManager.getInstance().isModelDownloaded(TranslateRemoteModel.Builder(mlkit(l)).build()).await()

    /** Needs network once (provisioning). */
    suspend fun download(l: String) {
        if (l == "en") return
        Translation.getClient(opts("en", l)).use { it.downloadModelIfNeeded(DownloadConditions.Builder().build()).await() }
        val mgr = RemoteModelManager.getInstance()
        for (other in LANGS - setOf("en", l)) {
            val m = TranslateRemoteModel.Builder(mlkit(other)).build()
            if (mgr.isModelDownloaded(m).await()) mgr.deleteDownloadedModel(m).await()
        }
    }

    private fun opts(from: String, to: String) = TranslatorOptions.Builder().setSourceLanguage(mlkit(from)).setTargetLanguage(mlkit(to)).build()

    fun use(l: String) {
        if (l == lang && (l == "en" || toL != null)) return
        close()
        lang = l
        if (l != "en") { toL = Translation.getClient(opts("en", l)); toEn = Translation.getClient(opts(l, "en")) }
    }

    /** EN -> L with entity masking; falls back to unmasked translation if a token is lost or duplicated. */
    suspend fun toLocal(en: String, terms: Map<String, String> = emptyMap()): String {
        val t = toL ?: return en
        // full mask first; if MT drops or duplicates a token (Tamil: "Spray [2] of [3]" -> "[2] [3] ... [2]"), mask names only
        for (namesOnly in listOf(false, true)) {
            val (masked, ents) = EntityMasker.mask(en, terms, namesOnly)
            if (ents.isNotEmpty()) EntityMasker.unmask(t.translate(masked).await(), ents)?.let { return it }
        }
        return t.translate(en).await()
    }

    suspend fun toEnglish(local: String): String = toEn?.translate(local)?.await() ?: local

    fun close() { toL?.close(); toEn?.close(); toL = null; toEn = null; lang = "en" }
}

/**
 * Protects numbers-with-units and pesticide names from the translator with opaque tokens.
 * [terms] (English phrase -> glossary name) are swapped for the local name on unmask: pest and crop names never go
 * through MT (MT turned "planthopper" into Tamil for "planned").
 */
object EntityMasker {
    private val TOKEN = Regex("\\[\\d+]")  // "[n]": ML Kit Hindi turned "Q1" into "क्यू 1" mid-sentence (probe 2026-11-19)

    fun mask(s: String, terms: Map<String, String> = emptyMap(), namesOnly: Boolean = false): Pair<String, List<String>> {
        val ents = mutableListOf<String>()
        var out = s
        for ((en, local) in terms.entries.sortedByDescending { it.key.length }) {
            val p = if (en.endsWith("y")) Regex.escape(en.dropLast(1)) + "(y|ies)" else Regex.escape(en) + "(e?s)?"  // whitefly -> whiteflies
            out = Regex("\\b$p\\b", RegexOption.IGNORE_CASE).replace(out) { ents += local; "[${ents.size}]" }
        }
        if (namesOnly) return out to ents
        out = OutputValidator.NUM_UNIT.replace(out) { ents += it.value; "[${ents.size}]" }
        for (a in OutputValidator.ACTIVES) {
            out = Regex("\\b${Regex.escape(a)}\\b", RegexOption.IGNORE_CASE).replace(out) { ents += it.value; "[${ents.size}]" }
        }
        return out to ents
    }

    /** null if any token is missing, duplicated or unknown after translation. */
    fun unmask(s: String, ents: List<String>): String? {
        val found = TOKEN.findAll(s).map { it.value }.toList()
        if (found.sorted() != List(ents.size) { "[${it + 1}]" }.sorted()) return null
        return TOKEN.replace(s) { ents[it.value.trim('[', ']').toInt() - 1] }
    }
}

/** Offline TTS. Google's engine explicitly: the M34's default Samsung engine has no Tamil voice. */
class TtsManager(ctx: Context, onReady: () -> Unit) {
    private var ready = false
    private val tts: TextToSpeech = TextToSpeech(ctx, { ready = it == TextToSpeech.SUCCESS; onReady() }, "com.google.android.tts")
    var rate = 0.9f

    fun locale(l: String): Locale = if (l == "en") Locale("en", "IN") else Locale(l, "IN")

    /** True when an offline (installed, no network) voice exists for the language. */
    fun offlineVoice(l: String): Boolean {
        if (!ready) return false
        val loc = locale(l)
        return tts.voices.orEmpty().any {
            it.locale.language == loc.language && !it.isNetworkConnectionRequired &&
                TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED !in it.features
        }
    }

    fun speak(text: String, l: String, flush: Boolean = false) {
        if (!ready) return
        val loc = locale(l)
        tts.voices.orEmpty().filter { it.locale.language == loc.language && !it.isNetworkConnectionRequired &&
            TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED !in it.features }.maxByOrNull { it.quality }?.let { tts.voice = it }
            ?: tts.setLanguage(loc)
        tts.setSpeechRate(rate)
        text.chunked(TextToSpeech.getMaxSpeechInputLength() - 1).forEachIndexed { i, chunk ->
            tts.speak(chunk, if (flush && i == 0) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD, null, "u${System.nanoTime()}")
        }
    }

    fun stop() { tts.stop() }
    fun shutdown() { tts.shutdown() }
}
