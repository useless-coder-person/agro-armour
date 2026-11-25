package org.agroarmour

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.agroarmour.i18n.TranslatorManager
import org.agroarmour.i18n.TtsManager
import org.agroarmour.kb.Card
import org.agroarmour.kb.Kb
import org.agroarmour.llm.AdviceContext
import org.agroarmour.llm.ExecuTorchLlmEngine
import org.agroarmour.llm.OutputValidator
import org.agroarmour.llm.PromptBuilder
import org.agroarmour.llm.TemplateAdvisor
import org.agroarmour.perf.MemorySampler
import org.agroarmour.perf.Metrics
import org.agroarmour.perf.Tier
import org.agroarmour.vision.Candidate
import org.agroarmour.vision.Detection
import org.agroarmour.vision.DiseaseClassifier
import org.agroarmour.vision.Fusion
import org.agroarmour.vision.LesionMap
import org.agroarmour.vision.LesionSegmenter
import org.agroarmour.vision.Level
import org.agroarmour.vision.Outcome
import org.agroarmour.vision.PestDetector
import org.agroarmour.vision.PestThresholds
import org.agroarmour.vision.Severity
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.Calendar
import java.util.concurrent.Executors

const val TAG = "AgroArmour"
enum class Screen { SETUP, HOME, RESULT, DIAG }
data class Region(val code: String, val name: String, val lang: String)
data class ChatTurn(val qLocal: String, val qEn: String, var aEn: String = "", var aLocal: String = "")

class AppVm(app: Application) : AndroidViewModel(app) {
    private val ctx: Context get() = getApplication()
    private val prefs = app.getSharedPreferences("agroarmour", Context.MODE_PRIVATE)
    private val llmDispatcher = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val visionDispatcher = Executors.newSingleThreadExecutor().asCoroutineDispatcher()

    val tier = Tier.of(app)
    val kb: Kb
    val regions: List<Region>
    private val diseaseLabels: List<String>
    private val thresholds: PestThresholds
    private val modelDir = File(app.filesDir, "models")
    // App-created (so app-owned) dirs, so `adb push` into them works.
    val llamaDir = File(app.getExternalFilesDir(null), "models/llama").apply { mkdirs(); File(parentFile!!.parentFile, "bench").mkdirs() }
    val translator = TranslatorManager()
    val tts = TtsManager(app) { refreshReadiness() }

    var screen by mutableStateOf(if (prefs.getBoolean("setup_done", false)) Screen.HOME else Screen.SETUP)
    var lang by mutableStateOf(prefs.getString("lang", "en")!!); private set
    var stateCode by mutableStateOf(prefs.getString("state", "TN")!!); private set
    var crop by mutableStateOf<String?>(prefs.getString("crop", "rice")); private set
    var stage by mutableStateOf<String?>(null)
    var status by mutableStateOf("")

    // readiness
    var visionReady by mutableStateOf(false); private set
    var llamaReady by mutableStateOf<Boolean?>(null); private set
    var packReady by mutableStateOf(false); private set
    var voiceReady by mutableStateOf(false); private set
    var busy by mutableStateOf(false); private set

    // result
    var photo by mutableStateOf<Bitmap?>(null); private set
    var dets by mutableStateOf<List<Detection>>(emptyList()); private set
    var outcome by mutableStateOf<Outcome?>(null); private set
    var findings by mutableStateOf<List<Candidate>>(emptyList()); private set
    var selected by mutableStateOf(0)
    var incidence by mutableStateOf<Int?>(null); private set
    var lesionMap by mutableStateOf<LesionMap?>(null); private set  // learned lesion segmentation of the photo
    val lesionRatio get() = lesionMap?.ratio
    val adviceEn = mutableStateListOf<String>()
    val adviceLocal = mutableStateListOf<String>()
    var adviceMode by mutableStateOf("")  // "llm" | "template" | ""
    val chat = mutableStateListOf<ChatTurn>()

    // diagnostics
    var metrics by mutableStateOf(Metrics()); private set
    val validatorTotals = mutableMapOf("number" to 0, "chemical" to 0, "format" to 0, "retries" to 0, "template_fallbacks" to 0)

    private var engine: ExecuTorchLlmEngine? = null
    private var genJob: Job? = null
    private val sampler = MemorySampler()

    // Models stay loaded while the app runs. ExecuTorch has one global threadpool: the LlmModule
    // constructor resets it (to 1 thread on the Exynos 1280) and every Module load with another count resets it again,
    // leaving loaded models on a freed pool. So every model uses [threads], and the Llama constructor runs under
    // [modelLock] with vision closed, followed by the vision loads that set the pool to [threads] before Llama's load().
    val threads = threadsOverride.takeIf { it > 0 } ?: DEFAULT_THREADS
    private val modelLock = Mutex()
    private val llmLock = Mutex()
    private var pest: PestDetector? = null
    private var disease: DiseaseClassifier? = null
    private var lesion: LesionSegmenter? = null
    private val speakQueue = Channel<Pair<String, MutableList<String>?>>(Channel.UNLIMITED)

    init {
        fun asset(p: String) = app.assets.open(p).bufferedReader().readText()
        kb = Kb(app.assets.list("kb/cards").orEmpty().map { Card.parse(asset("kb/cards/$it")) }, glossary(lang, ::asset))
        regions = JSONArray(asset("kb/regions.json")).let { a -> List(a.length()) { a.getJSONObject(it).let { r -> Region(r.getString("code"), r.getString("name"), r.getString("lang")) } } }
        diseaseLabels = JSONArray(asset("models/disease/labels.json")).let { a -> List(a.length()) { a.getString(it) } }
        thresholds = PestThresholds(asset("models/pest/thresholds.json"))
        // Sentences are translated and spoken in order, one at a time, while the LLM keeps streaming.
        viewModelScope.launch {
            for ((en, sink) in speakQueue) {
                val local = runCatching { translator.toLocal(en, glossaryTerms()) }.getOrElse { Log.w(TAG, "translate", it); en }
                sink?.add(local)
                if (prefs.getBoolean("autospeak", true)) tts.speak(local, lang)
            }
        }
        viewModelScope.launch(Dispatchers.IO) { copyVisionModels(); refreshReadiness().join(); preload() }
        translator.use(lang)
    }

    private fun glossary(l: String, asset: (String) -> String) = runCatching {
        JSONObject(asset("kb/glossary/$l.json")).getJSONObject("names").let { o -> o.keys().asSequence().associateWith { o.getString(it) } }
    }.getOrDefault(emptyMap())

    var kbLocal = kb; private set

    private fun copyVisionModels() {
        for (m in listOf("pest", "disease", "lesion")) for (f in ctx.assets.list("models/$m").orEmpty()) {
            val dst = File(modelDir, "$m/$f").apply { parentFile!!.mkdirs() }
            val same = f == "model.pte" && dst.length() == ctx.assets.openFd("models/$m/$f").use { it.length }
            if (!same) ctx.assets.open("models/$m/$f").use { i -> dst.outputStream().use { i.copyTo(it) } }
        }
        visionReady = File(modelDir, "pest/model.pte").exists() && File(modelDir, "disease/model.pte").exists()
    }

    fun refreshReadiness(): Job = viewModelScope.launch {
        packReady = runCatching { translator.isDownloaded(lang) }.getOrDefault(false)
        voiceReady = tts.offlineVoice(lang)
        if (llamaReady != true) llamaReady = withContext(Dispatchers.IO) { verifyLlama() }
    }

    /** SHA-256 of the pushed Llama files against the known export; cached by size+mtime. */
    private fun verifyLlama(): Boolean {
        val m = File(llamaDir, "model.pte"); val t = File(llamaDir, "tokenizer.model")
        if (!m.exists() || !t.exists()) return false
        val stamp = "${m.length()}:${m.lastModified()}:${t.length()}:${t.lastModified()}"
        if (prefs.getString("llama_ok", null) == stamp) return true
        fun sha(f: File) = MessageDigest.getInstance("SHA-256").let { md ->
            f.inputStream().buffered(1 shl 20).use { s -> val b = ByteArray(1 shl 20); while (true) { val n = s.read(b); if (n < 0) break; md.update(b, 0, n) } }
            md.digest().joinToString("") { "%02x".format(it) }
        }
        val ok = sha(m) == BuildConfig.LLAMA_SHA256 && sha(t) == BuildConfig.TOKENIZER_SHA256
        if (ok) prefs.edit().putString("llama_ok", stamp).apply()
        return ok
    }

    val offlineReady get() = visionReady && llamaReady == true && packReady && (voiceReady || lang == "en")

    fun setLanguage(l: String) {
        lang = l; prefs.edit().putString("lang", l).apply()
        translator.use(l)
        kbLocal = Kb(emptyList(), glossary(l) { ctx.assets.open(it).bufferedReader().readText() })
        refreshReadiness()
    }
    fun setState(code: String) { stateCode = code; prefs.edit().putString("state", code).apply() }
    fun pickCrop(c: String?) { crop = c; prefs.edit().putString("crop", c).apply() }
    fun finishSetup() { prefs.edit().putBoolean("setup_done", true).apply(); screen = Screen.HOME }

    fun downloadPack() = viewModelScope.launch {
        status = "…"
        status = runCatching { translator.download(lang); translator.close(); translator.use(lang); "ok" }.getOrElse { "error: ${it.message}" }
        refreshReadiness()
    }

    /** English phrases the LLM uses for this finding and the crops -> glossary names (kept out of MT). */
    private val factCache = HashMap<String, String>()
    /** Card text in the farmer's language (masked so doses and names stay exact); English if no pack yet. */
    suspend fun localFact(en: String): String {
        if (lang == "en") return en
        val k = "$lang|$en"
        return factCache[k] ?: runCatching { translator.toLocal(en, glossaryTerms()) }
            .onSuccess { factCache[k] = it }.getOrElse { Log.w(TAG, "fact translate", it); en }
    }

    fun glossaryTerms(): Map<String, String> {
        if (lang == "en") return emptyMap()
        val m = mutableMapOf<String, String>()
        for (c in listOf("rice", "tomato", "potato", "maize")) kbLocal.glossary[c]?.let { m[c] = it }
        kbLocal.glossary["rice"]?.let { m["paddy"] = it }
        val f = current() ?: return m
        val name = (kb.card(f.key)?.nameEn ?: return m).substringBefore(" (").trim()
        val local = localName(f.key)
        m[name] = local
        name.substringBefore(" of ").takeIf { it != name }?.let { m[it] = local }               // "Late blight of tomato"
        name.split(" ").let { w -> if (w.size >= 2 && w[0].lowercase() in m) m[w.drop(1).joinToString(" ")] = local }  // "Rice planthopper"
        return m
    }

    fun localName(key: String) = kbLocal.localName(key, kb.card(key)?.nameEn ?: key.substringAfter("__").replace('_', ' '))

    // ---------------- vision ----------------

    private fun decode(uri: Uri): Bitmap = if (Build.VERSION.SDK_INT >= 28) {
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(ctx.contentResolver, uri)) { d, info, _ ->
            d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE   // ImageDecoder applies EXIF orientation itself
            val s = 1600.0 / maxOf(info.size.width, info.size.height)
            if (s < 1) d.setTargetSize((info.size.width * s).toInt(), (info.size.height * s).toInt())
        }
    } else ctx.contentResolver.openInputStream(uri)!!.use { android.graphics.BitmapFactory.decodeStream(it) }

    fun analyze(uri: Uri, onDone: (() -> Unit)? = null) = viewModelScope.launch {
        cancelAdvice()
        busy = true; screen = Screen.RESULT; outcome = null; findings = emptyList(); lesionMap = null; adviceEn.clear(); adviceLocal.clear(); chat.clear()
        incidence = null; selected = 0; adviceMode = ""; forcedLevel = null
        val m = Metrics()
        try {
            withContext(visionDispatcher) {
                val bmp = m.time("decode") { decode(uri) }
                photo = bmp
                m.time("capture_to_result") {
                    val (d, probs) = modelLock.withLock {
                        openVision()
                        m.time("pest") { pest!!.detect(bmp) } to m.time("disease") { disease!!.classify(bmp) }
                    }
                    val pests = Fusion.aggregatePests(d, thresholds, bmp.width, bmp.height)
                    val o = Fusion.fuse(pests, Fusion.cropPrior(probs, diseaseLabels, crop), diseaseLabels)
                    dets = d
                    if (o is Outcome.Findings && o.items.any { it.kind != "pest" }) lesionMap = m.time("lesion") { modelLock.withLock { lesion?.run(bmp) } }
                    lesionMap?.ratio?.let { m.values["lesion_ratio"] = "%.3f".format(it) }
                    outcome = o
                    findings = (o as? Outcome.Findings)?.items ?: emptyList()
                    m.values["outcome"] = o.toString().take(200)
                }
            }
        } catch (e: Throwable) {
            Log.e(TAG, "analyze", e); status = "error: ${e.message}"; outcome = Outcome.NotSure
        }
        metrics = m; busy = false
        if (findings.isNotEmpty()) startAdvice(onDone) else { log(m); onDone?.invoke() }  // no-finding outcomes get a record too
    }

    fun current(): Candidate? = findings.getOrNull(selected)
    fun card(): Card? = current()?.let { kb.card(it.key) }

    fun imageLevel(): Level { val c = current() ?: return Level.MEDIUM; val cd = card() ?: return Level.MEDIUM
        return Severity.image(cd.severityMode, c.pest, lesionRatio) }
    var forcedLevel: Level? = null  // LLM eval only (scripts/llm_eval_device.sh)
    fun level(): Level = forcedLevel ?: Severity.final(imageLevel(), incidence)

    /** LLM eval case through the real advice path: card key + severity, optionally a follow-up question. */
    fun evalCase(key: String, sev: String, question: String?, done: () -> Unit) {
        cancelAdvice()
        val kind = kb.card(key)?.kind ?: "disease"
        findings = listOf(Candidate(key, kind, 1f, org.agroarmour.vision.Band.HIGH)); outcome = Outcome.Findings(findings)
        selected = 0; incidence = null; chat.clear(); forcedLevel = Level.valueOf(sev); metrics = Metrics().also { it.values["eval"] = true }
        screen = Screen.RESULT
        // Follow-up cases run after the advisory, as on the phone; the advisory record is marked warmup (prompt_eval skips it).
        if (question == null) startAdvice(done) else { metrics.values["warmup"] = true; startAdvice { ask(question, done) } }
    }
    fun basis() = if (incidence == null) "from photo only" else "from photo and your field check"

    fun setIncidenceAnswer(n: Int?) { val before = level(); incidence = n; if (level() != before) startAdvice() }
    fun select(i: Int) { selected = i; chat.clear(); startAdvice() }  // follow-up history is per finding

    private fun adviceContext() = AdviceContext(
        card()?.crops?.let { cs -> crop?.takeIf { it in cs } ?: cs.first() } ?: crop ?: "the crop", level().name, basis(),
        regions.firstOrNull { it.code == stateCode }?.name ?: stateCode, stateCode,
        Kb.season(Calendar.getInstance().get(Calendar.MONTH) + 1), stage,
    )

    fun facts(): String? = card()?.let { PromptBuilder.facts(it, adviceContext()) }

    // ---------------- LLM ----------------

    private fun cancelAdvice() { genJob?.cancel(); tts.stop() }

    fun leaveResult() {
        cancelAdvice(); screen = Screen.HOME
        primeLater()
    }

    /** Prefill the system prompt while the farmer reads, so the next advisory or follow-up prefills only its user turn. */
    private fun primeLater() = viewModelScope.launch(llmDispatcher) { engine?.prime(PromptBuilder.SYSTEM_PART) }

    // Vision models: call under modelLock.
    private fun openVision() {
        if (pest == null) pest = PestDetector(File(modelDir, "pest"), threads)
        if (disease == null) disease = DiseaseClassifier(File(modelDir, "disease"), threads)
        if (lesion == null && File(modelDir, "lesion/model.pte").exists()) lesion = LesionSegmenter(File(modelDir, "lesion"), threads)
    }
    private fun closeVision() { pest?.close(); disease?.close(); lesion?.close(); pest = null; disease = null; lesion = null }

    /** App start: Llama + vision load in the background while the farmer sets up and takes the photo. */
    private fun preload() = viewModelScope.launch {
        val m = Metrics().also { it.values["preload"] = true }
        if (ensureLlm(m) == null) modelLock.withLock { withContext(visionDispatcher) { openVision() } }
        log(m)
    }

    /** onTrimMemory: free everything; the next photo loads again. */
    fun freeModels() = viewModelScope.launch {
        llmLock.withLock { withContext(llmDispatcher) { engine?.unload(); engine = null } }
        modelLock.withLock { withContext(visionDispatcher) { closeVision() } }
    }

    private suspend fun ensureLlm(m: Metrics): ExecuTorchLlmEngine? = llmLock.withLock {
        engine?.let { return@withLock it }
        if (llamaReady != true) llamaReady = withContext(Dispatchers.IO) { verifyLlama() }  // files may be pushed after start
        if (tier == Tier.C || llamaReady != true) return@withLock null
        withContext(llmDispatcher) {
            runCatching {
                val t0 = System.nanoTime()
                val model = File(llamaDir, "model.pte"); val tok = File(llamaDir, "tokenizer.model")
                val e = ExecuTorchLlmEngine()
                // Order matters (see [threads]): constructor resets the pool, vision loads set it to [threads], load() captures it.
                // Warm-up stays before load(): running it concurrently with load() hung preload at 4 threads (2026-11-23).
                modelLock.withLock { closeVision(); e.create(model, tok); openVision(); m.time("warmup") { warmUp() } }
                e.load(model, tok)
                m.values["llm_load_ms"] = (System.nanoTime() - t0) / 1_000_000
                m.time("prime") { e.prime(PromptBuilder.SYSTEM_PART) }
                e
            }.onFailure { Log.e(TAG, "llm load", it) }.getOrNull()
        }?.also { engine = it }
    }

    /** One forward per vision model so the first photo does not pay for page faults. */
    private fun warmUp() {
        val b = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        runCatching { openVision(); pest!!.detect(b); disease!!.classify(b); lesion?.run(b) }.onFailure { Log.w(TAG, "warm-up", it) }
    }

    /**
     * Streams sentences that pass the validator; retry once at temperature 0, then the template.
     */
    private suspend fun stream(prompt: String, facts: String, maxNew: Int, m: Metrics, minSentences: Int = 2, onSentence: (String) -> Unit): Boolean {
        val e = m.time("llm_wait") { ensureLlm(m) } ?: return false
        for (attempt in 0..1) {
            val v = OutputValidator(facts)
            val kept = mutableListOf<String>()
            val buf = StringBuilder()
            fun drain(final: Boolean) {
                val parts = buf.split(OutputValidator.SENTENCE_END)
                val done = if (final) parts else parts.dropLast(1)
                done.forEach { s -> if (kept.size < 8) v.check(s)?.let { kept += it; onSentence(it) } }
                buf.clear(); if (!final) buf.append(parts.last())
            }
            val sampler = sampler.start(viewModelScope)
            try {
                e.generate(prompt, maxNew, if (attempt == 0) 0.2f else 0f)
                    .flowOn(llmDispatcher).collect { tok -> buf.append(tok); drain(false) }
                drain(true)
            } finally { sampler.cancel() }
            val st = e.stats()
            m.values["peak_pss_mb"] = this.sampler.peakMb
            st?.let { m.values["ttft_ms"] = it.ttftMs; m.values["prompt_tokens"] = it.promptTokens; m.values["gen_tokens"] = it.genTokens; m.values["tok_per_s"] = "%.2f".format(it.tokPerSec) }
            v.actions.forEach { (k, n) -> validatorTotals.merge(k, n, Int::plus) }
            m.values["validator_attempt$attempt"] = JSONObject(v.actions as Map<*, *>).toString()
            if (kept.size >= minSentences) return true
            validatorTotals.merge("retries", 1, Int::plus)
        }
        return false
    }

    fun startAdvice(onDone: (() -> Unit)? = null) {
        cancelAdvice()
        val card = card() ?: run { onDone?.invoke(); return }
        val ctxA = adviceContext()
        val facts = PromptBuilder.facts(card, ctxA)
        adviceEn.clear(); adviceLocal.clear(); adviceMode = "llm"
        val m = metrics
        genJob = viewModelScope.launch {
            val ok = m.time("advice_total") {
                stream(PromptBuilder.advisory(facts), facts, 160, m) { s ->
                    if (adviceEn.isEmpty()) m.values["first_sentence_ms"] = m.sinceMs()  // from photo (or eval start) to screen
                    adviceEn += s; speakQueue.trySend(s to adviceLocal)
                }
            }
            primeLater()
            if (!ok) {
                validatorTotals.merge("template_fallbacks", 1, Int::plus)
                adviceEn.clear(); adviceLocal.clear(); adviceMode = "template"
                TemplateAdvisor.render(card, ctxA).forEach { adviceEn += it; speakQueue.trySend(it to adviceLocal) }
            }
            m.values["advice_mode"] = adviceMode
            m.values["advice_en"] = adviceEn.joinToString(" ")
            if (lang != "en") {  // translations arrive through the speak queue
                repeat(60) { if (adviceLocal.size >= adviceEn.size) return@repeat; delay(250) }
                m.values["advice_local"] = adviceLocal.joinToString(" ")
                m.values["translated_sentences"] = "${adviceLocal.size}/${adviceEn.size}"
            }
            m.values["card"] = card.id; m.values["severity"] = ctxA.severity
            metrics = m
            log(m)
            onDone?.invoke()
        }
    }

    fun ask(questionLocal: String, onDone: (() -> Unit)? = null) {
        val card = card() ?: return
        cancelAdvice()
        genJob = viewModelScope.launch {
            val qEn = runCatching { translator.toEnglish(questionLocal) }.getOrDefault(questionLocal)
            val turn = ChatTurn(questionLocal, qEn); chat += turn
            val facts = PromptBuilder.facts(card, adviceContext())
            val history = chat.dropLast(1).takeLast(if (tier == Tier.A) 2 else 1).map { it.qEn to it.aEn }
            val m = Metrics()
            val answer = mutableListOf<String>()
            val local = mutableListOf<String>()
            var declined = false  // the model declined: show exactly OFF_TOPIC_REPLY, drop the rest of its wording
            val ok = stream(PromptBuilder.followUp(facts, history, qEn), facts, 120, m, minSentences = 1) { s ->
                if (declined) return@stream
                val shown = if (PromptBuilder.REFUSAL.containsMatchIn(s)) PromptBuilder.OFF_TOPIC_REPLY.also { declined = true } else s
                answer += shown; speakQueue.trySend(shown to local)
            }
            primeLater()
            val idx = chat.indexOf(turn)
            delay(50)
            // Not ok = the validator rejected both attempts: the same plain reply.
            val aEn = if (!ok || declined) PromptBuilder.OFF_TOPIC_REPLY else answer.joinToString(" ")
            if (idx >= 0) chat[idx] = turn.copy(aEn = aEn)
            // local text arrives through the speak queue; poll briefly for it so the chat shows the translation
            repeat(40) { if (local.size >= answer.size) return@repeat; delay(250) }
            if (idx >= 0) chat[idx] = chat[idx].copy(aLocal = local.joinToString(" "))
            m.values["card"] = card.id; m.values["severity"] = level().name; m.values["question_en"] = qEn
            m.values["advice_mode"] = if (!ok) "refusal_template" else if (declined) "llm_declined" else "llm"
            m.values["answer_en"] = chat.getOrNull(idx)?.aEn ?: ""
            metrics.values["followup"] = m.json(); log(m)
            onDone?.invoke()
        }
    }

    private fun log(m: Metrics) {
        val j = m.json().put("device", Build.MODEL).put("soc", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE)
            .put("tier", tier.name).put("lang", lang).put("threads", threads).put("validator_totals", JSONObject(validatorTotals as Map<*, *>))
        Log.i("AgroArmourBench", j.toString())
        runCatching { File(ctx.getExternalFilesDir(null), "metrics.jsonl").appendText(j.toString() + "\n") }
    }

    /**
     * Debug (--ez prime_check true): one advisory at temperature 0 from the primed system prompt, then the same prompt
     * from a reset context. Equal text means prefillPrompt() really filled the KV cache.
     */
    fun primeCheck() = viewModelScope.launch {
        val e = ensureLlm(Metrics()) ?: return@launch
        val card = kb.card("G_PLANTHOPPER") ?: return@launch
        val p = PromptBuilder.advisory(PromptBuilder.facts(card, adviceContext().copy(severity = "HIGH")))
        suspend fun run() = StringBuilder().also { sb -> e.generate(p, 60, 0f).flowOn(llmDispatcher).collect { sb.append(it) } }.toString() to e.stats()
        withContext(llmDispatcher) { e.prime(PromptBuilder.SYSTEM_PART) }
        val (primed, s1) = run()   // consumes the prime
        val (fresh, s2) = run()    // not primed: reset + full prompt
        Log.i("AgroArmourBench", JSONObject().put("prime_check", primed == fresh).put("threads", threads)
            .put("ttft_primed", s1?.ttftMs).put("prompt_tokens_primed", s1?.promptTokens)
            .put("ttft_fresh", s2?.ttftMs).put("prompt_tokens_fresh", s2?.promptTokens)
            .put("primed", primed).put("fresh", fresh).toString())
    }

    override fun onCleared() { engine?.unload(); closeVision(); translator.close(); tts.shutdown() }

    companion object {
        const val DEFAULT_THREADS = 8  // best TTFT and first-sentence time in the 2026-11-23/24 sweeps
        var threadsOverride = 0        // per process, from --ei threads N
    }
}
