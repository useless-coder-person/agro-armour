package org.agroarmour.kb

import org.json.JSONArray
import org.json.JSONObject

data class Chemical(val active: String, val formulation: String, val dose: String, val unit: String) {
    fun sentence() = "$active $formulation at $dose $unit"
}

data class Actions(val cultural: List<String>, val biological: List<String>, val chemical: List<Chemical>)

data class Card(
    val id: String, val kind: String, val key: String, val alsoKeys: List<String>, val crops: List<String>,
    val nameEn: String, val scientific: String?, val relevance: String, val identify: List<String>, val damage: String,
    val severityMode: String, val etl: String?, val actions: Map<String, Actions>, val safety: List<String>,
    val regionNotes: Map<String, String>, val seasonNotes: Map<String, String>, val stageNotes: Map<String, String>,
    val sources: List<Pair<String, String>>,
) {
    /** Every chemical entry is shown and given to the LLM; kb_lint checks each dose against the cited source text. */
    fun chemicals(level: String) = actions.getValue(level).chemical

    companion object {
        private fun strs(a: JSONArray?) = if (a == null) emptyList() else List(a.length()) { a.getString(it) }
        private fun map(o: JSONObject?) = o?.keys()?.asSequence()?.associateWith { o.getString(it) }?.filterValues { it.isNotBlank() } ?: emptyMap()

        fun parse(s: String): Card {
            val j = JSONObject(s)
            val act = j.getJSONObject("actions")
            return Card(
                j.getString("id"), j.getString("kind"), j.getString("key"), strs(j.optJSONArray("also_keys")),
                strs(j.getJSONArray("crops")), j.getJSONObject("names").getString("en"),
                j.getJSONObject("names").optString("scientific").ifBlank { null }, j.getString("relevance_india"),
                strs(j.getJSONArray("identify")), j.getString("damage"),
                j.getJSONObject("severity").getString("mode"),
                j.getJSONObject("severity").getJSONObject("thresholds").optString("etl").ifBlank { null },
                listOf("low", "medium", "high").associateWith { lv ->
                    val a = act.getJSONObject(lv)
                    val ch = a.getJSONArray("chemical")
                    Actions(strs(a.getJSONArray("cultural")), strs(a.getJSONArray("biological")), List(ch.length()) {
                        val c = ch.getJSONObject(it)
                        Chemical(c.getString("active"), c.getString("formulation"), c.getString("dose"), c.getString("dose_unit"))
                    })
                },
                strs(j.getJSONArray("safety")), map(j.optJSONObject("region_notes")), map(j.optJSONObject("season_notes")),
                map(j.optJSONObject("stage_notes")),
                j.getJSONArray("sources").let { a -> List(a.length()) { a.getJSONObject(it).let { s -> s.getString("title") to s.getString("url") } } },
            )
        }
    }
}

/** Selection key: model key (group or disease label, routed through also_keys). */
class Kb(cards: List<Card>, val glossary: Map<String, String>) {
    private val byKey = cards.flatMap { c -> (listOf(c.key) + c.alsoKeys).map { it to c } }.toMap()
    val size = cards.size

    fun card(key: String): Card? = byKey[key]

    /** Local-language name from the glossary; never machine translated. */
    fun localName(key: String, fallback: String) = glossary[key] ?: fallback

    companion object {
        const val KCC = "Kisan Call Centre 1800-180-1551"  // verified on mkisan.gov.in, 2026-11-19

        fun season(month: Int) = when (month) { in 6..10 -> "kharif"; 4, 5 -> "zaid"; else -> "rabi" }  // 1-based month
    }
}
