plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Must equal the Python-side executorch version (VERSIONS.lock).
val executorchVersion = "1.5.1"

android {
    namespace = "org.agroarmour"
    compileSdk = 35
    defaultConfig {
        applicationId = "org.agroarmour"
        minSdk = 26          // AAR declares minSdk 23 (verified 2026-11-16)
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += "arm64-v8a" }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("String", "EXECUTORCH_VERSION", "\"$executorchVersion\"")
        // Community SpinQuant export; the app checks the pushed files against these.
        buildConfigField("String", "LLAMA_SHA256", "\"8715cdba9e91f6bede00cc5f2d6b12397b95225da5630c4972a8da03001cda3b\"")
        buildConfigField("String", "TOKENIZER_SHA256", "\"82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55\"")
    }
    buildTypes {
        release { isMinifyEnabled = false }
    }
    testOptions {
        unitTests.all { it.testLogging { events("passed", "failed", "skipped"); showStandardStreams = true } }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true; buildConfig = true }
    androidResources { noCompress += "pte" }  // .pte must stay uncompressed so it can be mmapped from assets copy
    packaging { jniLibs { useLegacyPackaging = false } }
    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("generated/kbAssets"))
}

// Repo-root kb/ (cards, glossary, regions) is the single source; copied into assets/kb at build time.
val copyKb by tasks.registering(Copy::class) {
    from(rootProject.file("../kb")) { exclude("schema.json") }
    into(layout.buildDirectory.dir("generated/kbAssets/kb"))
}
tasks.named("preBuild") { dependsOn(copyKb) }

dependencies {
    implementation("org.pytorch:executorch-android:$executorchVersion")
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("com.google.mlkit:translate:17.0.3")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")  // android.jar's org.json is a stub on the JVM
    testImplementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}
