package org.agroarmour.vision

import org.json.JSONObject

enum class Band { LOW, MEDIUM, HIGH }

/** thresholds.json from ml/eval/eval_pest.py: per-group HIGH threshold (val precision >= 0.85), MEDIUM = frac * HIGH. */
class PestThresholds(json: String) {
    private val j = JSONObject(json)
    val classToGroup: List<String> = j.getJSONArray("class_to_group").let { a -> List(a.length()) { a.getString(it) } }
    private val mediumFrac = j.getDouble("medium_frac").toFloat()
    private val high = j.getJSONObject("groups").let { g -> g.keys().asSequence().associateWith { g.getJSONObject(it).getDouble("threshold").toFloat() } }

    fun band(group: String, score: Float): Band {
        val t = high.getValue(group)
        return if (score >= t) Band.HIGH else if (score >= mediumFrac * t) Band.MEDIUM else Band.LOW
    }
}

data class PestAgg(val group: String, val count: Int, val maxConf: Float, val areaFrac: Double, val band: Band, val fineClass: Int)
data class Candidate(val key: String, val kind: String, val conf: Float, val band: Band, val pest: PestAgg? = null)

sealed class Outcome {
    object NotPlant : Outcome()
    data class Findings(val items: List<Candidate>) : Outcome()
    data class Healthy(val label: String) : Outcome()
    object NotSure : Outcome()
}

/** Pure logic, unit-tested. */
object Fusion {
    // Classifier bands: fixed, not tuned on val.
    const val CLS_HIGH = 0.70f
    const val CLS_MEDIUM = 0.40f
    val PEST_DAMAGE = setOf("rice__dead_heart", "rice__hispa", "tomato__spider_mites")

    fun clsBand(p: Float) = if (p >= CLS_HIGH) Band.HIGH else if (p >= CLS_MEDIUM) Band.MEDIUM else Band.LOW

    /** Drop a box when a stronger box of another class covers the same insect (class-wise NMS keeps both). */
    fun crossClassSuppress(dets: List<Detection>, iou: Float = 0.7f): List<Detection> {
        fun fa(d: Detection) = floatArrayOf(d.x1.toFloat(), d.y1.toFloat(), d.x2.toFloat(), d.y2.toFloat())
        val sorted = dets.sortedByDescending { it.score }
        return sorted.filterIndexed { i, d -> sorted.take(i).none { it.cls != d.cls && Decoders.iou(fa(it), fa(d)) > iou } }
    }

    fun aggregatePests(dets: List<Detection>, th: PestThresholds, imgW: Int, imgH: Int): List<PestAgg> =
        crossClassSuppress(dets).groupBy { th.classToGroup[it.cls] }.map { (g, ds) ->
            val best = ds.maxBy { it.score }
            val area = ds.sumOf { (it.x2 - it.x1) * (it.y2 - it.y1) } / (imgW.toDouble() * imgH)
            val kept = ds.filter { th.band(g, it.score) != Band.LOW }
            PestAgg(g, kept.size, best.score, area, th.band(g, best.score), best.cls)
        }.filter { it.band != Band.LOW }.sortedByDescending { it.maxConf }

    /** Crop prior: zero other crops' classes (keep other__*), renormalise. crop == null: no prior. */
    fun cropPrior(probs: FloatArray, labels: List<String>, crop: String?): FloatArray {
        if (crop == null) return probs
        val q = FloatArray(probs.size) { if (labels[it].startsWith("${crop}__") || labels[it].startsWith("other__")) probs[it] else 0f }
        val s = q.sum()
        return if (s > 0f) FloatArray(q.size) { q[it] / s } else probs
    }

    fun fuse(pests: List<PestAgg>, probs: FloatArray, labels: List<String>): Outcome {
        val order = probs.indices.sortedByDescending { probs[it] }
        val top = order.take(2).map { i ->
            val l = labels[i]
            Candidate(l, if (l in PEST_DAMAGE) "pest_damage" else "disease", probs[i], clsBand(probs[i]))
        }
        val t1 = top[0]
        if (t1.key.startsWith("other__") && t1.band == Band.HIGH && pests.none { it.band == Band.HIGH }) return Outcome.NotPlant
        // Decided here, never asked: pest = every HIGH group, else the single best MEDIUM group;
        // disease = classifier top-1 if it is a condition at MEDIUM or above. Both, one or neither.
        val pestHits = pests.filter { it.band == Band.HIGH }.ifEmpty { pests.filter { it.band == Band.MEDIUM }.take(1) }
        val findings = pestHits.map { Candidate(it.group, "pest", it.maxConf, it.band, it) }.toMutableList()
        val isCondition = !t1.key.endsWith("__healthy") && !t1.key.startsWith("other__")
        if (isCondition && t1.band != Band.LOW) findings += t1
        if (findings.isNotEmpty()) return Outcome.Findings(findings.sortedByDescending { it.band == Band.HIGH })
        if (t1.key.endsWith("__healthy") && t1.band == Band.HIGH) return Outcome.Healthy(t1.key)
        return Outcome.NotSure
    }
}
