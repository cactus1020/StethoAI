package com.stethoai.digitalstethoscope.stetho_ai

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.annotation.NonNull
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.log10
import kotlin.math.sqrt

class MainActivity : FlutterActivity() {
    private val CHANNEL = "com.stethoai.digitalstethoscope/audio"
    private val PERMISSION_REQUEST_CODE = 2001
    private var pendingPermissionResult: MethodChannel.Result? = null

    // Audio recording state
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingThread: Thread? = null
    private val sampleRate = 16000
    private var pcmFile: File? = null
    private var wavFile: File? = null

    // Live metrics
    @Volatile private var currentRms = 0.0
    @Volatile private var currentPeak = 0.0
    @Volatile private var currentDb = -80.0
    private val liveWaveform = ArrayList<Double>()
    private val waveformLock = Any()
    @Volatile private var totalSamplesRecorded: Long = 0L

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "checkPermission" -> {
                    val granted = ContextCompat.checkSelfPermission(
                        this,
                        Manifest.permission.RECORD_AUDIO
                    ) == PackageManager.PERMISSION_GRANTED
                    result.success(granted)
                }

                "requestPermission" -> {
                    val granted = ContextCompat.checkSelfPermission(
                        this,
                        Manifest.permission.RECORD_AUDIO
                    ) == PackageManager.PERMISSION_GRANTED

                    if (granted) {
                        result.success(true)
                    } else {
                        pendingPermissionResult = result
                        ActivityCompat.requestPermissions(
                            this,
                            arrayOf(Manifest.permission.RECORD_AUDIO),
                            PERMISSION_REQUEST_CODE
                        )
                    }
                }

                "startRecording" -> {
                    val granted = ContextCompat.checkSelfPermission(
                        this,
                        Manifest.permission.RECORD_AUDIO
                    ) == PackageManager.PERMISSION_GRANTED

                    if (!granted) {
                        result.error("PERMISSION_DENIED", "Microphone permission is not granted", null)
                        return@setMethodCallHandler
                    }

                    try {
                        startAudioCapture()
                        result.success(true)
                    } catch (e: Exception) {
                        result.error("RECORDING_FAILED", e.message, null)
                    }
                }

                "stopRecording" -> {
                    try {
                        val path = stopAudioCapture()
                        result.success(path)
                    } catch (e: Exception) {
                        result.error("STOP_FAILED", e.message, null)
                    }
                }

                "getLiveLevel" -> {
                    val samplesCopy = ArrayList<Double>()
                    synchronized(waveformLock) {
                        samplesCopy.addAll(liveWaveform)
                    }
                    val data = mapOf(
                        "isRecording" to isRecording,
                        "rms" to currentRms,
                        "peak" to currentPeak,
                        "db" to currentDb,
                        "totalSamples" to totalSamplesRecorded,
                        "waveform" to samplesCopy
                    )
                    result.success(data)
                }

                "isRecording" -> {
                    result.success(isRecording)
                }

                else -> {
                    result.notImplemented()
                }
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            val granted = grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED
            pendingPermissionResult?.success(granted)
            pendingPermissionResult = null
        }
    }

    private fun startAudioCapture() {
        if (isRecording) return

        val minBufSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        val bufSize = minBufSize.coerceAtLeast(4096)

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            throw SecurityException("Microphone permission not granted")
        }

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufSize
        )

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            throw IllegalStateException("AudioRecord initialization failed")
        }

        val cacheDir = externalCacheDir ?: cacheDir
        val timestamp = System.currentTimeMillis()
        pcmFile = File(cacheDir, "temp_auscultation_$timestamp.pcm")
        wavFile = File(cacheDir, "auscultation_$timestamp.wav")

        totalSamplesRecorded = 0L
        synchronized(waveformLock) {
            liveWaveform.clear()
            for (i in 0 until 64) liveWaveform.add(0.0)
        }

        audioRecord?.startRecording()
        isRecording = true

        recordingThread = Thread {
            val shortBuf = ShortArray(1024)
            var fos: FileOutputStream? = null
            try {
                fos = FileOutputStream(pcmFile)
                val byteBuf = ByteBuffer.allocate(shortBuf.size * 2).order(ByteOrder.LITTLE_ENDIAN)

                while (isRecording) {
                    val read = audioRecord?.read(shortBuf, 0, shortBuf.size) ?: 0
                    if (read > 0) {
                        totalSamplesRecorded += read

                        // Write to PCM file
                        byteBuf.clear()
                        for (i in 0 until read) {
                            byteBuf.putShort(shortBuf[i])
                        }
                        fos.write(byteBuf.array(), 0, read * 2)

                        // Calculate live metrics (RMS, Peak, dB)
                        var sumSq = 0.0
                        var peak = 0
                        for (i in 0 until read) {
                            val s = shortBuf[i].toDouble()
                            sumSq += s * s
                            val absVal = Math.abs(shortBuf[i].toInt())
                            if (absVal > peak) peak = absVal
                        }

                        val rmsVal = sqrt(sumSq / read)
                        currentRms = (rmsVal / 32768.0).coerceIn(0.0, 1.0)
                        currentPeak = (peak.toDouble() / 32768.0).coerceIn(0.0, 1.0)
                        currentDb = if (rmsVal > 1.0) 20 * log10(rmsVal / 32768.0) else -80.0

                        // Downsample buffer to populate live waveform for oscilloscope
                        synchronized(waveformLock) {
                            val step = (read / 32).coerceAtLeast(1)
                            var idx = 0
                            while (idx < read) {
                                val norm = (shortBuf[idx] / 32768.0).coerceIn(-1.0, 1.0)
                                if (liveWaveform.size >= 64) {
                                    liveWaveform.removeAt(0)
                                }
                                liveWaveform.add(norm)
                                idx += step
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                try {
                    fos?.flush()
                    fos?.close()
                } catch (e: IOException) {
                    e.printStackTrace()
                }
            }
        }
        recordingThread?.start()
    }

    private fun stopAudioCapture(): String? {
        if (!isRecording) return wavFile?.absolutePath

        isRecording = false
        try {
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
        } catch (e: Exception) {
            e.printStackTrace()
        }

        try {
            recordingThread?.join(1000)
            recordingThread = null
        } catch (e: InterruptedException) {
            e.printStackTrace()
        }

        // Convert PCM to standard WAV format
        val pcm = pcmFile
        val wav = wavFile
        if (pcm != null && pcm.exists() && wav != null) {
            convertPcmToWav(pcm, wav, sampleRate, 1, 16)
            pcm.delete()
            return wav.absolutePath
        }

        return null
    }

    private fun convertPcmToWav(
        pcmFile: File,
        wavFile: File,
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int
    ) {
        val pcmSize = pcmFile.length()
        val totalDataLen = pcmSize + 36
        val byteRate = (sampleRate * channels * bitsPerSample / 8).toLong()

        val header = ByteArray(44)
        // RIFF chunk descriptor
        header[0] = 'R'.code.toByte()
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte()
        header[7] = ((totalDataLen shr 24) and 0xff).toByte()
        // WAVE header
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        // 'fmt ' chunk
        header[12] = 'f'.code.toByte()
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16 // Subchunk1Size for PCM
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1 // AudioFormat 1 = PCM
        header[21] = 0
        header[22] = channels.toByte()
        header[23] = 0
        header[24] = (sampleRate and 0xff).toByte()
        header[25] = ((sampleRate shr 8) and 0xff).toByte()
        header[26] = ((sampleRate shr 16) and 0xff).toByte()
        header[27] = ((sampleRate shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = ((channels * bitsPerSample) / 8).toByte() // Block align
        header[33] = 0
        header[34] = bitsPerSample.toByte()
        header[35] = 0
        // 'data' chunk
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (pcmSize and 0xff).toByte()
        header[41] = ((pcmSize shr 8) and 0xff).toByte()
        header[42] = ((pcmSize shr 16) and 0xff).toByte()
        header[43] = ((pcmSize shr 24) and 0xff).toByte()

        FileInputStream(pcmFile).use { fis ->
            FileOutputStream(wavFile).use { fos ->
                fos.write(header, 0, 44)
                val buffer = ByteArray(4096)
                var bytesRead: Int
                while (fis.read(buffer).also { bytesRead = it } != -1) {
                    fos.write(buffer, 0, bytesRead)
                }
            }
        }
    }

    override fun onDestroy() {
        stopAudioCapture()
        super.onDestroy()
    }
}
