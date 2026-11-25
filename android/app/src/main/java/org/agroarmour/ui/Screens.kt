@file:OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)

package org.agroarmour.ui

import android.content.Intent
import android.content.res.Configuration
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.os.Build
import android.speech.tts.TextToSpeech
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.MutableTransitionState
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.StartOffset
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import org.agroarmour.AppVm
import org.agroarmour.BuildConfig
import org.agroarmour.R
import org.agroarmour.Screen
import org.agroarmour.vision.Detection
import org.agroarmour.vision.LesionMap
import org.agroarmour.vision.Outcome
import java.util.Locale

private val LANG_LABEL = mapOf("en" to "English", "hi" to "हिंदी", "ta" to "தமிழ்")
private val CROPS = listOf("rice", "tomato", "potato", "maize")
private val STAGES = listOf("seedling" to R.string.stage_seedling, "vegetative" to R.string.stage_vegetative,
    "flowering" to R.string.stage_flowering, "maturity" to R.string.stage_maturity)
private val PAD = PaddingValues(horizontal = 16.dp, vertical = 8.dp)

@Composable
private fun res(vm: AppVm): Resources {
    val ctx = LocalContext.current
    return remember(vm.lang) {
        ctx.createConfigurationContext(Configuration(ctx.resources.configuration).apply { setLocale(Locale(vm.lang)) }).resources
    }
}

// ---------------------------------------------------------------- building blocks

@Composable
private fun SectionCard(title: String? = null, icon: ImageVector? = null, modifier: Modifier = Modifier,
                        container: Color = MaterialTheme.colorScheme.surfaceContainerLow, content: @Composable ColumnScope.() -> Unit) =
    Card(modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp), colors = CardDefaults.cardColors(containerColor = container),
        border = if (container == MaterialTheme.colorScheme.surfaceContainerLow)
            androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f)) else null) {
        Column(Modifier.padding(16.dp).animateContentSize(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (title != null) Row(verticalAlignment = Alignment.CenterVertically) {
                icon?.let { Icon(it, null, Modifier.size(20.dp), tint = MaterialTheme.colorScheme.primary); Spacer(Modifier.width(8.dp)) }
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            }
            content()
        }
    }

@Composable
private fun Body(t: String, small: Boolean = false, color: Color = Color.Unspecified) =
    Text(t, style = if (small) MaterialTheme.typography.bodyMedium else MaterialTheme.typography.bodyLarge,
        color = if (small && color == Color.Unspecified) MaterialTheme.colorScheme.onSurfaceVariant else color)

@Composable
private fun Bullet(t: String) = Row {
    Box(Modifier.padding(top = 10.dp, end = 10.dp).size(6.dp).clip(CircleShape).background(MaterialTheme.colorScheme.primary))
    Text(t, style = MaterialTheme.typography.bodyLarge)
}

/** Card text in the chosen language; shows English until the on-device translation is ready. */
@Composable
private fun tr(vm: AppVm, en: String): String = produceState(en, en, vm.lang) { value = vm.localFact(en) }.value

@Composable
private fun Label(t: String) =
    Text(t, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.SemiBold,
        modifier = Modifier.padding(top = 4.dp))

@Composable
private fun Pill(text: String, color: Color, icon: ImageVector? = null, onClick: (() -> Unit)? = null) =
    Surface(shape = CircleShape, color = color.copy(alpha = 0.14f), contentColor = color,
        modifier = if (onClick != null) Modifier.clip(CircleShape).clickable(onClick = onClick) else Modifier) {
        Row(Modifier.padding(horizontal = 12.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            icon?.let { Icon(it, null, Modifier.size(16.dp)); Spacer(Modifier.width(6.dp)) }
            Text(text, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
        }
    }

/** Three dots that rise and fade in turn: the "thinking" indicator for the advisor and chat answers. */
@Composable
fun ThinkingDots(color: Color = MaterialTheme.colorScheme.primary) {
    val t = rememberInfiniteTransition(label = "dots")
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.height(20.dp)) {
        repeat(3) { i ->
            val p by t.animateFloat(0f, 1f, infiniteRepeatable(tween(520, easing = FastOutSlowInEasing), RepeatMode.Reverse,
                StartOffset(i * 170)), label = "dot$i")
            Box(Modifier.size(8.dp).graphicsLayer { translationY = -6.dp.toPx() * p; alpha = 0.35f + 0.65f * p }
                .clip(CircleShape).background(color))
        }
    }
}

/** A text line that fades and slides in once, so streamed sentences arrive smoothly. */
@Composable
private fun Arrive(content: @Composable () -> Unit) {
    val s = remember { MutableTransitionState(false).apply { targetState = true } }
    AnimatedVisibility(s, enter = fadeIn(tween(350)) + slideInVertically(tween(350)) { it / 3 }) { content() }
}

@Composable
private fun Bar(title: String, onBack: (() -> Unit)? = null, backLabel: String = "", actions: @Composable () -> Unit = {}) =
    TopAppBar(
        title = { Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold) },
        navigationIcon = {
            if (onBack != null) IconButton(onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, backLabel) }
            else Box(Modifier.padding(start = 12.dp, end = 4.dp).size(36.dp).clip(CircleShape)
                .background(MaterialTheme.colorScheme.primary), contentAlignment = Alignment.Center) {
                Icon(AppIcons.Leaf, null, Modifier.size(22.dp), tint = MaterialTheme.colorScheme.onPrimary)
            }
        },
        actions = { actions() },
        colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
    )

@Composable
private fun ScrollBody(inner: PaddingValues, content: @Composable ColumnScope.() -> Unit) =
    Column(Modifier.fillMaxSize().padding(inner).consumeWindowInsets(inner).imePadding().verticalScroll(rememberScrollState()).padding(PAD),
        verticalArrangement = Arrangement.spacedBy(12.dp)) {
        content(); Spacer(Modifier.height(16.dp))
    }

// ---------------------------------------------------------------- app

@Composable
fun App(vm: AppVm, takePhoto: () -> Unit, pickPhoto: () -> Unit) {
    val r = res(vm)
    AnimatedContent(vm.screen, transitionSpec = { fadeIn(tween(240)) togetherWith fadeOut(tween(160)) }, label = "screen") { s ->
        when (s) {
            Screen.SETUP -> Setup(vm, r)
            Screen.HOME -> Home(vm, r, takePhoto, pickPhoto)
            Screen.RESULT -> Result(vm, r)
            Screen.DIAG -> Diagnostics(vm, r)
        }
    }
}

// ---------------------------------------------------------------- setup

@Composable
private fun Setup(vm: AppVm, r: Resources) {
    val ctx = LocalContext.current
    Scaffold(
        topBar = { Bar(r.getString(R.string.setup)) },
        bottomBar = {
            Surface(color = MaterialTheme.colorScheme.background) {
                Button({ vm.finishSetup() }, Modifier.navigationBarsPadding().padding(16.dp).fillMaxWidth().heightIn(min = 56.dp),
                    shape = RoundedCornerShape(16.dp)) { Text(r.getString(R.string.done), style = MaterialTheme.typography.titleMedium) }
            }
        },
    ) { inner ->
        ScrollBody(inner) {
            SectionCard(r.getString(R.string.language)) {
                SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                    LANG_LABEL.entries.forEachIndexed { i, (l, label) ->
                        SegmentedButton(vm.lang == l, { vm.setLanguage(l) }, SegmentedButtonDefaults.itemShape(i, LANG_LABEL.size),
                            Modifier.heightIn(min = 52.dp)) { Text(label, style = MaterialTheme.typography.titleMedium) }
                    }
                }
            }
            SectionCard(r.getString(R.string.state), Icons.Default.LocationOn) { StatePicker(vm) }
            SectionCard(r.getString(R.string.readiness), Icons.Default.CheckCircle) {
                Check(vm.visionReady, r.getString(R.string.r_vision))
                Check(vm.llamaReady, r.getString(R.string.r_llama))
                if (vm.lang != "en") {
                    Check(vm.packReady, r.getString(R.string.r_pack)) {
                        if (!vm.packReady) OutlinedButton({ vm.downloadPack() }) { Text(r.getString(R.string.download_pack)) }
                        if (vm.status.isNotEmpty()) Body(vm.status, small = true)
                    }
                    Check(vm.voiceReady, r.getString(R.string.r_voice)) {
                        if (!vm.voiceReady) OutlinedButton({
                            ctx.startActivity(Intent(TextToSpeech.Engine.ACTION_INSTALL_TTS_DATA).setPackage("com.google.android.tts")
                                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                        }) { Text(r.getString(R.string.install_voice)) }
                    }
                    FilledTonalIconButton({ vm.refreshReadiness() }, Modifier.align(Alignment.End)) {
                        Icon(Icons.Default.Refresh, r.getString(R.string.readiness))
                    }
                }
            }
        }
    }
}

@Composable
private fun StatePicker(vm: AppVm) {
    var open by remember { mutableStateOf(false) }
    val current = vm.regions.firstOrNull { it.code == vm.stateCode }?.name ?: vm.stateCode
    ExposedDropdownMenuBox(open, { open = it }) {
        OutlinedTextField(current, {}, readOnly = true, singleLine = true, shape = RoundedCornerShape(14.dp),
            textStyle = MaterialTheme.typography.bodyLarge,
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(open) },
            modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryNotEditable).fillMaxWidth())
        ExposedDropdownMenu(open, { open = false }) {
            vm.regions.forEach { s ->
                DropdownMenuItem(
                    text = { Text(s.name, style = MaterialTheme.typography.bodyLarge,
                        fontWeight = if (s.code == vm.stateCode) FontWeight.SemiBold else FontWeight.Normal) },
                    onClick = { vm.setState(s.code); open = false },
                    trailingIcon = if (s.code == vm.stateCode) ({ Icon(Icons.Default.CheckCircle, null, tint = MaterialTheme.colorScheme.primary) }) else null,
                    contentPadding = ExposedDropdownMenuDefaults.ItemContentPadding,
                )
            }
        }
    }
}

/** One readiness row: spinner while unknown, a green tick when ready, an amber warning plus the fix when not. */
@Composable
private fun Check(ok: Boolean?, label: String, fix: @Composable ColumnScope.() -> Unit = {}) {
    val dark = isSystemInDarkTheme()
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().heightIn(min = 40.dp)) {
        Box(Modifier.size(28.dp), contentAlignment = Alignment.Center) {
            when (ok) {
                null -> CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.5.dp)
                true -> Icon(Icons.Default.CheckCircle, "ready", tint = SevColor.of("LOW", dark))
                false -> Icon(Icons.Default.Warning, "not ready", tint = SevColor.of("MEDIUM", dark))
            }
        }
        Spacer(Modifier.width(12.dp))
        Text(label, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
    }
    if (ok != true) Column(Modifier.padding(start = 40.dp), verticalArrangement = Arrangement.spacedBy(4.dp), content = fix)
}

// ---------------------------------------------------------------- home

@Composable
private fun Home(vm: AppVm, r: Resources, takePhoto: () -> Unit, pickPhoto: () -> Unit) {
    val dark = isSystemInDarkTheme()
    Scaffold(topBar = {
        Bar(r.getString(R.string.app_name)) {
            IconButton({ vm.screen = Screen.SETUP }) { Icon(Icons.Default.Settings, r.getString(R.string.setup)) }
            IconButton({ vm.screen = Screen.DIAG }) { Icon(Icons.Default.Info, r.getString(R.string.diagnostics)) }
        }
    }) { inner ->
        ScrollBody(inner) {
            if (vm.offlineReady) Pill(r.getString(R.string.ready_offline), SevColor.of("LOW", dark), Icons.Default.CheckCircle)
            else Pill(r.getString(R.string.not_ready), SevColor.of("MEDIUM", dark), Icons.Default.Warning) { vm.screen = Screen.SETUP }

            // Capture first: it is the one thing the farmer came to do.
            Card(Modifier.fillMaxWidth(), shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
                Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Row(verticalAlignment = Alignment.Top) {
                        Icon(AppIcons.Leaf, null, Modifier.size(22.dp).padding(top = 2.dp), tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(10.dp))
                        Text(r.getString(R.string.tip), style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onPrimaryContainer)
                    }
                    Button(takePhoto, Modifier.fillMaxWidth().heightIn(min = 72.dp), shape = RoundedCornerShape(18.dp)) {
                        Icon(AppIcons.Camera, null, Modifier.size(28.dp)); Spacer(Modifier.width(12.dp))
                        Text(r.getString(R.string.take_photo), style = MaterialTheme.typography.titleLarge)
                    }
                    OutlinedButton(pickPhoto, Modifier.fillMaxWidth().heightIn(min = 56.dp), shape = RoundedCornerShape(18.dp),
                        colors = ButtonDefaults.outlinedButtonColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLowest)) {
                        Icon(AppIcons.Image, null); Spacer(Modifier.width(10.dp))
                        Text(r.getString(R.string.gallery), style = MaterialTheme.typography.titleMedium)
                    }
                }
            }

            SectionCard(r.getString(R.string.crop), AppIcons.Leaf) {
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    for (c in CROPS) BigChip(vm.crop == c, vm.localName(c)) { vm.pickCrop(c) }
                    BigChip(vm.crop == null, r.getString(R.string.dont_know)) { vm.pickCrop(null) }
                }
            }
            SectionCard(r.getString(R.string.stage)) {
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    for ((s, id) in STAGES) BigChip(vm.stage == s, r.getString(id)) { vm.stage = if (vm.stage == s) null else s }
                }
            }
        }
    }
}

@Composable
private fun BigChip(selected: Boolean, label: String, onClick: () -> Unit) =
    FilterChip(selected, onClick, { Text(label, style = MaterialTheme.typography.titleSmall) },
        Modifier.heightIn(min = 44.dp), shape = RoundedCornerShape(12.dp),
        leadingIcon = if (selected) ({ Icon(Icons.Default.CheckCircle, null, Modifier.size(FilterChipDefaults.IconSize)) }) else null)

// ---------------------------------------------------------------- result

private fun annotate(src: Bitmap, dets: List<Detection>, lesion: LesionMap?): Bitmap {
    val b = src.copy(Bitmap.Config.ARGB_8888, true)
    val p = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = maxOf(3f, b.width / 200f); color = 0xFFFFC107.toInt() }
    val c = Canvas(b)
    lesion?.let { l ->  // lesion pixels tinted red, mapped back from the letterboxed model input
        val ov = Bitmap.createBitmap(IntArray(l.size * l.size) { if (l.cls[it].toInt() == 2) 0x99FF2020.toInt() else 0 }, l.size, l.size, Bitmap.Config.ARGB_8888)
        c.drawBitmap(ov, android.graphics.Rect(l.lb.padLeft, l.lb.padTop, l.lb.padLeft + l.lb.newW, l.lb.padTop + l.lb.newH),
            android.graphics.Rect(0, 0, b.width, b.height), Paint(Paint.FILTER_BITMAP_FLAG))
    }
    val rad = b.width / 80f
    dets.take(20).forEach { c.drawRoundRect(it.x1.toFloat(), it.y1.toFloat(), it.x2.toFloat(), it.y2.toFloat(), rad, rad, p) }
    return b
}

/** The photo, with a scanning sweep over it while the vision models run. */
@Composable
private fun Photo(vm: AppVm) {
    val ph = vm.photo ?: return
    val img = remember(ph, vm.dets, vm.lesionMap) { annotate(ph, vm.dets.filter { it.score >= 0.25f }, vm.lesionMap).asImageBitmap() }
    val scan = MaterialTheme.colorScheme.primaryContainer
    Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(Color.Black)) {
        Image(img, null, Modifier.fillMaxWidth().heightIn(max = 340.dp), contentScale = ContentScale.Fit)
        if (vm.busy) {
            val y by rememberInfiniteTransition(label = "scan").animateFloat(0f, 1f,
                infiniteRepeatable(tween(1400, easing = LinearEasing), RepeatMode.Reverse), label = "y")
            Canvas(Modifier.matchParentSize()) {
                drawRect(Color.Black.copy(alpha = 0.25f))
                val cy = size.height * y; val band = size.height * 0.18f
                drawRect(Brush.verticalGradient(listOf(Color.Transparent, scan.copy(alpha = 0.45f), Color.Transparent),
                    startY = cy - band, endY = cy + band), topLeft = Offset(0f, cy - band),
                    size = androidx.compose.ui.geometry.Size(size.width, band * 2))
                drawLine(scan, Offset(0f, cy), Offset(size.width, cy), strokeWidth = 3.dp.toPx())
            }
        }
    }
}

@Composable
private fun Message(icon: ImageVector, tint: Color, text: String) = SectionCard {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(icon, null, Modifier.size(32.dp), tint = tint); Spacer(Modifier.width(14.dp))
        Text(text, style = MaterialTheme.typography.bodyLarge)
    }
}

@Composable
private fun Result(vm: AppVm, r: Resources) {
    val dark = isSystemInDarkTheme()
    val f = vm.current().takeIf { !vm.busy && vm.outcome is Outcome.Findings }
    Scaffold(topBar = { Bar(f?.let { vm.localName(it.key) } ?: r.getString(R.string.app_name), { vm.leaveResult() }, r.getString(R.string.back)) }) { inner ->
        ScrollBody(inner) {
            Photo(vm)
            if (vm.busy) {
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(4.dp)) {
                    ThinkingDots(); Spacer(Modifier.width(12.dp)); Body(r.getString(R.string.checking))
                }
                return@ScrollBody
            }
            when (val o = vm.outcome) {
                Outcome.NotPlant -> { Message(Icons.Default.Warning, SevColor.of("MEDIUM", dark), r.getString(R.string.not_plant)); return@ScrollBody }
                Outcome.NotSure, null -> { Message(Icons.Default.Info, MaterialTheme.colorScheme.primary, r.getString(R.string.not_sure)); return@ScrollBody }
                is Outcome.Healthy -> {
                    Message(Icons.Default.CheckCircle, SevColor.of("LOW", dark), "${vm.localName(o.label)}\n${r.getString(R.string.healthy)}")
                    return@ScrollBody
                }
                is Outcome.Findings -> Unit
            }
            if (vm.findings.size > 1) ScrollableTabRow(vm.selected, edgePadding = 0.dp, containerColor = Color.Transparent) {
                vm.findings.forEachIndexed { i, fd -> Tab(vm.selected == i, { vm.select(i) }, text = { Text(vm.localName(fd.key)) }) }
            }
            Findings(vm, r)
        }
    }
}

@Composable
private fun ColumnScope.Findings(vm: AppVm, r: Resources) {
    val dark = isSystemInDarkTheme()
    val f = vm.current() ?: return
    val card = vm.card()
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(vm.localName(f.key), style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        if (vm.lang != "en") Body(card?.nameEn ?: f.key, small = true)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Pill("${r.getString(R.string.confidence)}: ${r.getString(bandRes(f.band.name))} · ${"%.2f".format(f.conf)}",
                MaterialTheme.colorScheme.secondary)
            f.pest?.let { Pill("IP102 #${it.fineClass} · ${it.count}×", MaterialTheme.colorScheme.secondary) }
        }
    }

    val lv = vm.level()
    val sev = SevColor.of(lv.name, dark)
    SectionCard {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(r.getString(R.string.severity), style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
            Pill(r.getString(bandRes(lv.name)), sev)
        }
        SeverityMeter(lv.ordinal, sev)
        Body(r.getString(if (vm.incidence == null) R.string.from_photo else R.string.from_check), small = true)
        HorizontalDivider(Modifier.padding(vertical = 4.dp), color = MaterialTheme.colorScheme.outlineVariant)
        Body(r.getString(R.string.incidence_q))
        val opts = listOf("0-1" to 0, "2-4" to 3, "5-10" to 5, r.getString(R.string.skip) to null)
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            opts.forEachIndexed { i, (label, n) ->
                SegmentedButton(vm.incidence == n, { vm.setIncidenceAnswer(n) }, SegmentedButtonDefaults.itemShape(i, opts.size),
                    Modifier.heightIn(min = 48.dp), icon = {}) { Text(label, maxLines = 1) }
            }
        }
    }
    if (card == null) { HelpLine(r); return }

    // Facts panel: card text translated on device with doses and names masked; chemical lines keep the English original below.
    val a = card.actions.getValue(lv.name.lowercase())
    SectionCard(r.getString(R.string.facts), Icons.Default.Info) {
        if (card.relevance == "low") Body(r.getString(R.string.relevance_low), color = SevColor.of("MEDIUM", dark))
        card.identify.forEach { Body(tr(vm, it), small = true) }
        Label(r.getString(R.string.do_now))
        (a.cultural + a.biological).forEach { Bullet(tr(vm, it)) }
        Label(r.getString(R.string.chemical))
        val chem = card.chemicals(lv.name.lowercase())
        if (chem.isEmpty()) Body(r.getString(R.string.chemical_hidden), small = true) else chem.forEach { c -> Bullet(tr(vm, c.sentence())); if (vm.lang != "en") Body(c.sentence(), small = true) }
        card.safety.firstOrNull()?.let {
            Surface(shape = RoundedCornerShape(14.dp), color = MaterialTheme.colorScheme.tertiaryContainer,
                modifier = Modifier.fillMaxWidth().padding(top = 4.dp)) {
                Row(Modifier.padding(12.dp)) {
                    Icon(Icons.Default.Warning, null, Modifier.size(20.dp), tint = MaterialTheme.colorScheme.onTertiaryContainer)
                    Spacer(Modifier.width(10.dp))
                    Column {
                        Text(r.getString(R.string.safety), style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onTertiaryContainer)
                        Text(tr(vm, it), style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onTertiaryContainer)
                    }
                }
            }
        }
        card.regionNotes[vm.stateCode]?.let {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.LocationOn, null, Modifier.size(18.dp), tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(6.dp)); Label(r.getString(R.string.local_note))
            }
            Body(tr(vm, it))
        }
    }
    HelpLine(r)
    Advisor(vm, r)
    Chat(vm, r)
    Sources(card.sources, "Card ${card.id}", r)
}

/** Three segments filling up to the level, animated when the farmer's field check changes it. */
@Composable
private fun SeverityMeter(level: Int, color: Color) = Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
    repeat(3) { i ->
        val fill by animateFloatAsState(if (i <= level) 1f else 0f, tween(450), label = "seg$i")
        Box(Modifier.weight(1f).height(10.dp).clip(CircleShape).background(MaterialTheme.colorScheme.surfaceVariant)) {
            Box(Modifier.fillMaxWidth(fill).height(10.dp).clip(CircleShape).background(color))
        }
    }
}

@Composable
private fun HelpLine(r: Resources) = Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(horizontal = 4.dp)) {
    Icon(Icons.Default.Call, null, Modifier.size(18.dp), tint = MaterialTheme.colorScheme.primary)
    Spacer(Modifier.width(8.dp)); Body(r.getString(R.string.help_line), small = true)
}

@Composable
private fun Advisor(vm: AppVm, r: Resources) = SectionCard(container = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.55f)) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(32.dp).clip(CircleShape).background(MaterialTheme.colorScheme.primary), contentAlignment = Alignment.Center) {
            Icon(AppIcons.Leaf, null, Modifier.size(18.dp), tint = MaterialTheme.colorScheme.onPrimary)
        }
        Spacer(Modifier.width(10.dp))
        Text(r.getString(R.string.advisor), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
        FilledTonalIconButton({ vm.tts.speak((if (vm.lang == "en") vm.adviceEn else vm.adviceLocal).joinToString(" "), vm.lang, flush = true) }) {
            Icon(AppIcons.VolumeUp, r.getString(R.string.speak))
        }
        IconButton({ vm.tts.stop() }) { Icon(AppIcons.Stop, r.getString(R.string.stop)) }
    }
    if (vm.adviceMode == "template") Body(r.getString(R.string.basic_mode), small = true)
    if (vm.adviceEn.isEmpty()) Row(verticalAlignment = Alignment.CenterVertically) {
        ThinkingDots(); Spacer(Modifier.width(12.dp)); Body(r.getString(R.string.advisor_loading), small = true)
    }
    (if (vm.lang == "en") vm.adviceEn else vm.adviceLocal).forEach { s -> Arrive { Body(s) } }
    if (vm.lang != "en" && vm.adviceEn.isNotEmpty()) Body("EN: " + vm.adviceEn.joinToString(" "), small = true)
}

@Composable
private fun Chat(vm: AppVm, r: Resources) = SectionCard(r.getString(R.string.ask)) {
    vm.chat.forEach { t ->
        Bubble(mine = true) {
            Text(t.qLocal, style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onPrimary)
            // interpretation, so a helper can spot a mistranslation
            if (vm.lang != "en") Text(t.qEn, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.8f))
        }
        Bubble(mine = false) {
            if (t.aEn.isEmpty()) ThinkingDots(MaterialTheme.colorScheme.onSurfaceVariant)
            else Arrive { Text(t.aLocal.ifEmpty { t.aEn }, style = MaterialTheme.typography.bodyLarge) }
        }
    }
    var q by rememberSaveable { mutableStateOf("") }
    val send = { if (q.isNotBlank()) { vm.ask(q); q = "" } }
    Row(verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(q, { q = it }, Modifier.weight(1f), textStyle = MaterialTheme.typography.bodyLarge, shape = RoundedCornerShape(24.dp),
            placeholder = { Text(r.getString(R.string.ask)) }, maxLines = 4)
        Spacer(Modifier.width(8.dp))
        FilledTonalIconButton(send, Modifier.size(52.dp), enabled = q.isNotBlank()) {
            Icon(Icons.AutoMirrored.Filled.Send, r.getString(R.string.send))
        }
    }
}

@Composable
private fun Bubble(mine: Boolean, content: @Composable ColumnScope.() -> Unit) =
    Box(Modifier.fillMaxWidth(), contentAlignment = if (mine) Alignment.CenterEnd else Alignment.CenterStart) {
        Surface(color = if (mine) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceContainerHighest,
            shape = RoundedCornerShape(topStart = 18.dp, topEnd = 18.dp, bottomStart = if (mine) 18.dp else 4.dp, bottomEnd = if (mine) 4.dp else 18.dp),
            modifier = Modifier.widthIn(max = 300.dp)) {
            Column(Modifier.padding(horizontal = 14.dp, vertical = 10.dp).animateContentSize(), content = content)
        }
    }

@Composable
private fun Sources(sources: List<Pair<String, String>>, review: String, r: Resources) {
    var open by rememberSaveable { mutableStateOf(false) }
    SectionCard {
        Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).clickable { open = !open }.heightIn(min = 40.dp),
            verticalAlignment = Alignment.CenterVertically) {
            Text(r.getString(R.string.sources), style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
            Icon(if (open) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown, null)
        }
        AnimatedVisibility(open, enter = expandVertically() + fadeIn()) {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                sources.forEach { (t, u) -> Body("$t — $u", small = true) }
                Body(review, small = true)
            }
        }
    }
}

private fun bandRes(name: String) = when (name) { "HIGH" -> R.string.band_HIGH; "MEDIUM" -> R.string.band_MEDIUM; else -> R.string.band_LOW }

// ---------------------------------------------------------------- diagnostics

@Composable
private fun Diagnostics(vm: AppVm, r: Resources) {
    val ctx = LocalContext.current
    val am = ctx.getSystemService(android.app.ActivityManager::class.java)
    val mi = android.app.ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
    Scaffold(topBar = { Bar(r.getString(R.string.diagnostics), { vm.screen = Screen.HOME }, r.getString(R.string.back)) }) { inner ->
        ScrollBody(inner) {
            SectionCard("Device") {
                Kv("Device", "${Build.MANUFACTURER} ${Build.MODEL}")
                Kv("SoC", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE)
                Kv("ABI · cores", "${Build.SUPPORTED_ABIS.first()} · ${Runtime.getRuntime().availableProcessors()}")
                Kv("RAM · Android", "${"%.1f".format(mi.totalMem / 1e9)} GB · ${Build.VERSION.RELEASE}")
                Kv("Memory tier", "${vm.tier.name}: ${vm.tier.desc}")
            }
            SectionCard("Runtime") {
                Kv("ExecuTorch", "${BuildConfig.EXECUTORCH_VERSION} (XNNPACK, CPU)")
                Kv("Threads", "${vm.threads} (all models)")
                Kv("Llama sha256", "${BuildConfig.LLAMA_SHA256.take(16)}…")
                Kv("Verified", "${vm.llamaReady}")
                Kv("KB cards", "${vm.kb.size}")
            }
            SectionCard("Last run") {
                if (vm.metrics.stages.isEmpty() && vm.metrics.values.isEmpty()) Body("-", small = true)
                vm.metrics.stages.forEach { (k, v) -> Kv(k, "$v ms") }
                vm.metrics.values.forEach { (k, v) -> Kv(k, "$v") }
            }
            SectionCard("Validator (session)") { vm.validatorTotals.forEach { (k, v) -> Kv(k, "$v") } }
        }
    }
}

@Composable
private fun Kv(k: String, v: String) = Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
    Text(k, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(0.4f))
    Text(v, style = MaterialTheme.typography.bodyMedium, fontFamily = FontFamily.Monospace, textAlign = TextAlign.End, modifier = Modifier.weight(0.6f))
}
