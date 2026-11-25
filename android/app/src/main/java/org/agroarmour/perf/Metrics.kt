package org.agroarmour.perf

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONObject

/** Memory tiers, decided once from total RAM. */
enum class Tier(val desc: String) {
    A("≥ 6 GB: 2 earlier turns in a rebuilt follow-up prompt"), B("4-6 GB: 1 earlier turn in a rebuilt follow-up prompt"), C("< 4 GB: template advice only");

    companion object {
        fun of(ctx: Context): Tier {
            val mi = ActivityManager.MemoryInfo().also { (ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(it) }
            val gb = mi.totalMem / 1e9
            return if (gb >= 6.0) A else if (gb >= 4.0) B else C
        }
    }
}

/** Per-run metrics shown on Diagnostics and logged as JSON (tag AgroArmourBench). */
class Metrics {
    private val t0 = System.nanoTime()
    fun sinceMs() = (System.nanoTime() - t0) / 1_000_000
    val stages = linkedMapOf<String, Long>()
    val values = linkedMapOf<String, Any>()

    inline fun <T> time(stage: String, block: () -> T): T {
        val t0 = System.nanoTime()
        try { return block() } finally { stages[stage] = (System.nanoTime() - t0) / 1_000_000 }
    }

    fun json() = JSONObject().apply { put("stages_ms", JSONObject(stages as Map<*, *>)); values.forEach { (k, v) -> put(k, v) } }
}

/** Polls total PSS of this process every 250 ms while running; peak in MB. */
class MemorySampler {
    @Volatile var peakMb = 0; private set

    fun start(scope: CoroutineScope) = scope.launch(Dispatchers.Default) {
        peakMb = 0
        val mi = Debug.MemoryInfo()
        while (isActive) {
            Debug.getMemoryInfo(mi)
            peakMb = maxOf(peakMb, mi.totalPss / 1024)
            delay(250)
        }
    }
}
