package org.agroarmour.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material.icons.materialIcon
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.addPathNodes
import androidx.compose.ui.unit.sp

// Leaf green + warm soil tones; no dynamic colour so the app looks the same on every phone and reads well in sunlight.
private val Light = lightColorScheme(
    primary = Color(0xFF2E6B3F), onPrimary = Color.White,
    primaryContainer = Color(0xFFCDE9CF), onPrimaryContainer = Color(0xFF0B2A15),
    secondary = Color(0xFF52634F), onSecondary = Color.White,
    secondaryContainer = Color(0xFFE2EBDB), onSecondaryContainer = Color(0xFF18241A),
    tertiary = Color(0xFF8A5A1F), onTertiary = Color.White,
    tertiaryContainer = Color(0xFFFBE2C2), onTertiaryContainer = Color(0xFF2E1800),
    background = Color(0xFFF7F6F0), onBackground = Color(0xFF1B1C18),
    surface = Color(0xFFF7F6F0), onSurface = Color(0xFF1B1C18),
    surfaceVariant = Color(0xFFE3E5DA), onSurfaceVariant = Color(0xFF44483F),
    surfaceContainerLowest = Color.White, surfaceContainerLow = Color(0xFFFFFFFF),
    surfaceContainer = Color(0xFFF0F0E8), surfaceContainerHigh = Color(0xFFEAEBE2), surfaceContainerHighest = Color(0xFFE4E5DC),
    outline = Color(0xFF74796D), outlineVariant = Color(0xFFC6C9BC),
    error = Color(0xFFB3261E),
)
private val Dark = darkColorScheme(
    primary = Color(0xFF9AD5A3), onPrimary = Color(0xFF00391A),
    primaryContainer = Color(0xFF15522A), onPrimaryContainer = Color(0xFFCDE9CF),
    secondary = Color(0xFFB9CCB4), onSecondary = Color(0xFF253423),
    secondaryContainer = Color(0xFF3B4B38), onSecondaryContainer = Color(0xFFD5E8CF),
    tertiary = Color(0xFFF0BD7E), onTertiary = Color(0xFF4A2800),
    tertiaryContainer = Color(0xFF6A430A), onTertiaryContainer = Color(0xFFFBE2C2),
    background = Color(0xFF121411), onBackground = Color(0xFFE2E3DB),
    surface = Color(0xFF121411), onSurface = Color(0xFFE2E3DB),
    surfaceVariant = Color(0xFF43483F), onSurfaceVariant = Color(0xFFC3C8BB),
    surfaceContainerLow = Color(0xFF1A1C19), surfaceContainer = Color(0xFF1E201C),
    surfaceContainerHigh = Color(0xFF282B26), surfaceContainerHighest = Color(0xFF333630),
    outline = Color(0xFF8D9286), outlineVariant = Color(0xFF43483F),
)

// Larger body text than the M3 default: many users read with difficulty (nothing under 14 sp).
private val Type = Typography().let { t ->
    t.copy(
        bodyLarge = t.bodyLarge.copy(fontSize = 17.sp, lineHeight = 25.sp),
        bodyMedium = t.bodyMedium.copy(fontSize = 15.sp, lineHeight = 21.sp),
        bodySmall = t.bodySmall.copy(fontSize = 14.sp, lineHeight = 19.sp),
        labelLarge = t.labelLarge.copy(fontSize = 16.sp),
        labelMedium = t.labelMedium.copy(fontSize = 14.sp),
    )
}

@Composable
fun AgroArmourTheme(content: @Composable () -> Unit) =
    MaterialTheme(colorScheme = if (isSystemInDarkTheme()) Dark else Light, typography = Type, content = content)

/** Severity colours; always shown together with the level word, never colour alone. */
object SevColor {
    fun of(level: String, dark: Boolean) = when (level) {
        "HIGH" -> if (dark) Color(0xFFFFB4A9) else Color(0xFFB3261E)
        "MEDIUM" -> if (dark) Color(0xFFF0BD7E) else Color(0xFF9A5B00)
        else -> if (dark) Color(0xFF9AD5A3) else Color(0xFF2E6B3F)
    }
}

// Material icons that are not in material-icons-core (paths from Material Icons, Apache-2.0).
private fun icon(name: String, vararg paths: String): ImageVector = materialIcon(name) {
    paths.forEach { p -> addPath(addPathNodes(p), fill = SolidColor(Color.Black)) }; this
}
object AppIcons {
    val Camera = icon("PhotoCamera",
        "M12,8.8a3.2,3.2 0,1 0,0 6.4a3.2,3.2 0,1 0,0 -6.4z",
        "M9,2L7.17,4H4c-1.1,0 -2,0.9 -2,2v12c0,1.1 0.9,2 2,2h16c1.1,0 2,-0.9 2,-2V6c0,-1.1 -0.9,-2 -2,-2h-3.17L15,2H9zM12,17c-2.76,0 -5,-2.24 -5,-5s2.24,-5 5,-5 5,2.24 5,5 -2.24,5 -5,5z")
    val Image = icon("Image", "M21,19V5c0,-1.1 -0.9,-2 -2,-2H5c-1.1,0 -2,0.9 -2,2v14c0,1.1 0.9,2 2,2h14c1.1,0 2,-0.9 2,-2zM8.5,13.5l2.5,3.01L14.5,12l4.5,6H5l3.5,-4.5z")
    val Leaf = icon("Eco", "M6.05,8.05c-2.73,2.73 -2.73,7.15 -0.02,9.88c1.47,-3.4 4.09,-6.24 7.36,-7.93c-2.77,2.34 -4.71,5.61 -5.39,9.32c2.6,1.23 5.8,0.78 7.95,-1.37C19.43,14.47 20,4 20,4S9.53,4.57 6.05,8.05z")
    val VolumeUp = icon("VolumeUp", "M3,9v6h4l5,5V4L7,9H3zM16.5,12c0,-1.77 -1.02,-3.29 -2.5,-4.03v8.05c1.48,-0.73 2.5,-2.25 2.5,-4.02zM14,3.23v2.06c2.89,0.86 5,3.54 5,6.71s-2.11,5.85 -5,6.71v2.06c4.01,-0.91 7,-4.49 7,-8.77s-2.99,-7.86 -7,-8.77z")
    val Stop = icon("Stop", "M6,6h12v12H6z")
}
