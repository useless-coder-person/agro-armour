package org.agroarmour.vision

enum class Level { LOW, MEDIUM, HIGH }

/**
 * Image score, optionally raised by the farmer's 10-plant incidence count. The lesion ratio comes from
 * the learned segmentation model (LesionSegmenter); there are no colour rules.
 */
object Severity {
    // Area rule needs >= 2 insects: one close-up insect filling the frame is not a heavy infestation.
    fun fromCount(n: Int, areaFrac: Double) = when {
        n > 15 || (n >= 2 && areaFrac > 0.15) -> Level.HIGH
        n <= 3 && areaFrac < 0.02 -> Level.LOW
        else -> Level.MEDIUM
    }

    fun fromLesion(r: Double) = when { r < 0.10 -> Level.LOW; r <= 0.30 -> Level.MEDIUM; else -> Level.HIGH }

    /** "Check 10 plants nearby. How many show this?" */
    fun fromIncidence(plants: Int) = when { plants <= 1 -> Level.LOW; plants <= 4 -> Level.MEDIUM; else -> Level.HIGH }

    /** mode from the KB card: count | presence_incidence | lesion_area. lesionRatio null = not computed. */
    fun image(mode: String, pest: PestAgg?, lesionRatio: Double?): Level = when (mode) {
        "count" -> if (pest != null) fromCount(pest.count, pest.areaFrac) else Level.MEDIUM
        "lesion_area" -> lesionRatio?.let { fromLesion(it) } ?: Level.MEDIUM
        else -> Level.MEDIUM
    }

    fun final(image: Level, incidence: Int?) = if (incidence == null) image else maxOf(image, fromIncidence(incidence))
}
