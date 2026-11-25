package org.agroarmour

import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.activity.enableEdgeToEdge
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.compose.foundation.layout.fillMaxSize
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import org.agroarmour.ui.App
import org.agroarmour.ui.AgroArmourTheme
import java.io.File

class MainActivity : ComponentActivity() {
    private val vm: AppVm by viewModels()
    // System camera app via TakePicture instead of CameraX; works offline. A fixed file, so the photo
    // still arrives if Android killed the app (models resident) while the camera was open.
    private val captureUri by lazy {
        FileProvider.getUriForFile(this, "$packageName.files", File(cacheDir, "photos/capture.jpg").apply { parentFile!!.mkdirs() })
    }

    private val camera = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok -> if (ok) vm.analyze(captureUri) }
    private val gallery = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> uri?.let { vm.analyze(it) } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Thread sweep (scripts/thread_sweep.sh): --ei threads N on a cold start, read before the view model loads models.
        intent.getIntExtra("threads", 0).takeIf { it > 0 }?.let { AppVm.threadsOverride = it }
        enableEdgeToEdge()
        setContent {
            AgroArmourTheme {
                Surface(Modifier.fillMaxSize()) {
                    BackHandler(vm.screen != Screen.HOME) { if (vm.screen == Screen.RESULT) vm.leaveResult() else vm.screen = Screen.HOME }
                    App(vm, takePhoto = { camera.launch(captureUri) },
                        pickPhoto = { gallery.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) })
                }
            }
        }
        bench(intent.getStringExtra("bench_image"))
    }

    /**
     * Benchmark hook for scripts/bench_device.sh:
     * adb shell am start -n org.agroarmour/.MainActivity --es bench_image <file in Android/data/org.agroarmour/files/bench/> [--es crop rice]
     * Runs vision + advice and logs one JSON line (tag AgroArmourBench), also appended to files/metrics.jsonl.
     */
    private fun bench(name: String?) {
        // --es lang hi|ta|en  [--ez provision true]: switch language (and download its ML Kit pack while online).
        intent.getStringExtra("lang")?.let { vm.setLanguage(it); if (intent.getBooleanExtra("provision", false)) vm.downloadPack() }
        // LLM eval: --es eval_id ID --es eval_key KEY --es eval_sev LOW|MEDIUM|HIGH [--es eval_q "question"]
        intent.getStringExtra("eval_key")?.let { key ->
            val id = intent.getStringExtra("eval_id")
            vm.evalCase(key, intent.getStringExtra("eval_sev") ?: "MEDIUM", intent.getStringExtra("eval_q")) { Log.i("AgroArmourBench", "done $id") }
        }
        // Debug (--ez mt_probe true): masked translation of a fixed sentence; glossary names must survive.
        if (intent.getBooleanExtra("prime_check", false)) vm.primeCheck()
        if (intent.getBooleanExtra("mt_probe", false)) lifecycleScope.launch {
            val s = "You are dealing with a rice planthopper infestation. Spray 30 ml of imidacloprid."
            val t = mapOf("rice planthopper" to "<PLANTHOPPER>", "rice" to "<RICE>")
            Log.i("AgroArmourProbe", "$s => ${runCatching { vm.translator.toLocal(s, t) }.getOrElse { it.message }}")
            val masked = org.agroarmour.i18n.EntityMasker.mask(s, t).first  // raw MT of the tokens, to see how they come back
            Log.i("AgroArmourProbe", "$masked => ${runCatching { vm.translator.toLocal(masked) }.getOrElse { it.message }}")
        }
        if (name == null) return
        intent.getStringExtra("crop")?.let { vm.pickCrop(it.takeUnless { c -> c.isBlank() || c == "none" }) }
        val f = File(getExternalFilesDir(null), "bench/$name")
        Log.i("AgroArmourBench", "start $f")
        vm.analyze(Uri.fromFile(f)) { Log.i("AgroArmourBench", "done $name") }
    }

    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        bench(intent.getStringExtra("bench_image"))
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        Log.i(TAG, "onTrimMemory $level")
        // Models are resident; free them under pressure. Android 14+ delivers only UI_HIDDEN/BACKGROUND, so there the
        // low-memory killer decides, and the fixed capture file keeps the photo.
        if (level == TRIM_MEMORY_RUNNING_CRITICAL || level >= TRIM_MEMORY_COMPLETE) vm.freeModels()
    }
}
