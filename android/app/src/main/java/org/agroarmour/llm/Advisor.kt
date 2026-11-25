package org.agroarmour.llm

import org.agroarmour.kb.Card
import org.agroarmour.kb.Kb

data class AdviceContext(
    val crop: String, val severity: String, val basis: String, val state: String, val stateCode: String?,
    val season: String, val stage: String?,
)

/** The LLM sees only FACTS rendered from the KB card. */
object PromptBuilder {
    const val SYSTEM = "You are a crop advisor for small farmers in India. Use ONLY the FACTS given.\n" +
        "Talk to the farmer as \"you\". Only suggest steps that are written in FACTS.\n" +
        "Never add a chemical, dose, number or product that is not in FACTS.\n" +
        "Write 4 to 6 short, simple sentences. One idea per sentence. No lists, no headings.\n" +
        "If the question is not about this crop problem, say you can only help with crop problems.\n" +
        "If FACTS do not answer the question, say so and tell the farmer to ask the local KVK."
    const val FACTS_BUDGET_TOKENS = 250

    /** Rough Llama-3 token estimate for English (about 4 characters per token); used only for trimming. */
    fun estTokens(s: String) = (s.length + 3) / 4

    fun facts(card: Card, c: AdviceContext): String {
        val lv = c.severity.lowercase()
        val doNow = (card.actions.getValue(lv).cultural + card.actions.getValue(lv).biological).take(3)
        val chem = card.chemicals(lv).firstOrNull()?.sentence() ?: "none listed; ask KVK"
        val local = c.stateCode?.let { card.regionNotes[it] }
        val stage = c.stage?.let { card.stageNotes[it] }
        fun render(identify: List<String>, actions: List<String>, withStage: Boolean, withLocal: Boolean) = buildString {
            appendLine("FACTS")
            appendLine("Problem: ${card.nameEn} on ${c.crop}. Severity: ${c.severity} (${c.basis}).")
            appendLine("Region: ${c.state}. Season: ${c.season}. Crop stage: ${c.stage ?: "not given"}.")
            appendLine("How to recognise: ${identify.joinToString(" ")}")
            if (card.damage.isNotBlank()) appendLine("Damage: ${card.damage}")
            appendLine("Do now: ${actions.joinToString(" ").ifBlank { "Ask the local KVK." }}")
            appendLine("Chemical option: $chem")
            card.safety.firstOrNull()?.let { appendLine("Safety: $it") }
            if (withStage && stage != null) appendLine("Stage note: $stage")
            if (withLocal && local != null) appendLine("Local note: $local")
            append("Help line: ${Kb.KCC}.")
        }
        // Trim order when over budget: local/stage notes, then the second recognition sentence, then actions.
        val tries = sequence {
            yield(render(card.identify.take(2), doNow, true, true))
            yield(render(card.identify.take(2), doNow, false, false))
            yield(render(card.identify.take(1), doNow, false, false))
            yield(render(card.identify.take(1), doNow.take(2), false, false))
            yield(render(card.identify.take(1), doNow.take(1), false, false))
        }
        return tries.firstOrNull { estTokens(it) <= FACTS_BUDGET_TOKENS } ?: tries.last()
    }

    /** The system turn. The app pre-fills it into the KV cache while idle (ExecuTorchLlmEngine.prime). */
    const val SYSTEM_PART = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n$SYSTEM<|eot_id|>"
    // Repeated in the follow-up: the 1B model ignored the system-prompt rule for off-topic questions (eval x01-x03).
    // No KVK referral: the model attached it to off-topic subjects ("ask your local KVK about the cricket match").
    const val OFF_TOPIC_REPLY = "I cannot help with that."  // one plain reply whenever unsure
    const val OFF_TOPIC = "If the QUESTION is not about this crop problem, reply only: $OFF_TOPIC_REPLY"

    /**
     * A sentence in which the model declines (in its own varied words, often with extra chatter such as "I'd be happy to
     * discuss cricket"). The app then shows exactly [OFF_TOPIC_REPLY] instead of the whole answer.
     */
    val REFUSAL = Regex("only help with crop|can ?not help|can't help|unable to help|not able to help|cannot answer|can't answer",
        RegexOption.IGNORE_CASE)

    fun userTurn(user: String) = "<|start_header_id|>user<|end_header_id|>\n\n$user<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    fun chat(user: String) = SYSTEM_PART + userTurn(user)

    fun advisory(facts: String) = chat("$facts\n\nTASK\nExplain to the farmer what this is, how serious it is, and what to do first.")

    /** Follow-up: same FACTS, at most [history] previous Q/A pairs, then the question. */
    fun followUp(facts: String, history: List<Pair<String, String>>, question: String) = chat(buildString {
        appendLine(facts); appendLine()
        history.forEach { (q, a) -> appendLine("Earlier question: $q"); appendLine("Earlier answer: $a") }
        append("QUESTION\n$question\n\n")
        append(OFF_TOPIC)
    })
}

/** Sentence-level filter: numbers and pesticide names must come from FACTS. */
class OutputValidator(private val facts: String) {
    private val factsNorm = norm(facts)
    val actions = mutableMapOf("number" to 0, "chemical" to 0, "format" to 0)

    fun check(sentence: String): String? {
        val s = sentence.replace(SPECIAL, "").replace(Regex("^[\\s*#>-]+|\\*\\*"), "").trim()  // eot token arrives as text
        if (s.isEmpty()) return null
        if ("<|" in s || s.contains("assistant", true) && s.length < 20) { actions.merge("format", 1, Int::plus); return null }
        if (NUM_UNIT.findAll(s).any { norm(it.value) !in factsNorm }) { actions.merge("number", 1, Int::plus); return null }
        val low = s.lowercase()
        if (ACTIVES.any { Regex("\\b${Regex.escape(it)}\\b").containsMatchIn(low) && norm(it) !in factsNorm }) {
            actions.merge("chemical", 1, Int::plus); return null
        }
        return s
    }

    companion object {
        val NUM_UNIT = Regex("\\d+(\\.\\d+)?\\s?(ml|l|g|kg|%|days?|weeks?|months?|hours?|ha|acre|litres?)\\b", RegexOption.IGNORE_CASE)
        val SENTENCE_END = Regex("(?<=[.!?])\\s+")
        fun norm(s: String) = s.lowercase().replace(Regex("\\s+"), "")
        fun accept(kept: List<String>) = kept.size >= 2
        val SPECIAL = Regex("<\\|(eot_id|end_of_text|eom_id)\\|>")

        // Common actives and pesticide words seen in Indian IPM packages and the DPPQS banned list; detection only.
        val ACTIVES = listOf(
            "imidacloprid", "thiamethoxam", "acetamiprid", "clothianidin", "dinotefuran", "buprofezin", "pymetrozine",
            "ethofenprox", "etofenprox", "ethofenoprox", "acephate", "chlorpyrifos", "chlorpyriphos", "quinalphos", "profenofos",
            "monocrotophos", "dimethoate", "malathion", "carbofuran", "carbaryl", "phorate", "fipronil", "cartap",
            "chlorantraniliprole", "flubendiamide", "emamectin", "spinosad", "indoxacarb", "lambda-cyhalothrin",
            "cypermethrin", "deltamethrin", "fenvalerate", "triazophos", "endosulfan", "dichlorvos", "methyl parathion",
            "oxydemeton", "spiromesifen", "diafenthiuron", "fenpyroximate", "propargite", "dicofol", "sulphur", "sulfur",
            "abamectin", "mancozeb", "zineb", "metiram", "propineb", "chlorothalonil", "copper oxychloride",
            "copper hydroxide", "carbendazim", "thiophanate", "tricyclazole", "isoprothiolane", "hexaconazole",
            "propiconazole", "tebuconazole", "difenoconazole", "azoxystrobin", "pyraclostrobin", "trifloxystrobin",
            "kresoxim", "metalaxyl", "cymoxanil", "dimethomorph", "fluopicolide", "cyazofamid", "validamycin",
            "streptomycin", "streptocycline", "kasugamycin", "edifenphos", "ediphenphos", "iprodione", "captan", "thiram",
            "paraquat", "glyphosate", "2,4-d", "atrazine",
        )
    }
}

/** Deterministic advisory from the card. */
object TemplateAdvisor {
    fun render(card: Card, c: AdviceContext): List<String> {
        val lv = c.severity.lowercase()
        val a = card.actions.getValue(lv)
        return buildList {
            add("This looks like ${card.nameEn} on ${c.crop}.")
            add("The problem level is ${c.severity.lowercase()}, ${c.basis}.")
            card.identify.firstOrNull()?.let(::add)
            (a.cultural + a.biological).take(2).forEach(::add)
            card.chemicals(lv).firstOrNull()?.let { add("Chemical option: ${it.sentence()}.") }
            card.safety.firstOrNull()?.let(::add)
            add("For more help, ask your local KVK or call the ${Kb.KCC}.")
        }
    }
}
