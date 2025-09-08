package com.davieir4.segpet

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class SegmentationProcessor(private val context: Context) {

    private var interpreter: Interpreter? = null
    private val inputSize = 128 // Ajuste conforme seu modelo
    private val numClasses = 3 // Pet, Margin, Background

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            val modelBuffer = loadModelFile("segmentation_model.tflite")
            interpreter = Interpreter(modelBuffer)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun processImage(bitmap: Bitmap): Bitmap? {
        if (interpreter == null) return null

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

        val outputBuffer = Array(1) { Array(inputSize) { Array(inputSize) { FloatArray(numClasses) } } }

        interpreter?.run(inputBuffer, outputBuffer)

        return convertOutputToBitmap(outputBuffer[0])
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(inputSize * inputSize)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val value = intValues[pixel++]
                byteBuffer.putFloat(((value shr 16) and 0xFF) / 255.0f) // R
                byteBuffer.putFloat(((value shr 8) and 0xFF) / 255.0f)  // G
                byteBuffer.putFloat((value and 0xFF) / 255.0f)          // B
            }
        }
        return byteBuffer
    }

    private fun convertOutputToBitmap(output: Array<Array<FloatArray>>): Bitmap {
        val bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)

        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val classScores = output[i][j]
                
                var classId = 0
                var maxScore = -Float.MAX_VALUE
                for (k in 0 until numClasses) {
                    if (classScores[k] > maxScore) {
                        maxScore = classScores[k]
                        classId = k
                    }
                }

                val color = getColorForClass(classId)
                bitmap.setPixel(j, i, color)
            }
        }
        return bitmap
    }

    private fun getColorForClass(classId: Int): Int {
        return when (classId) {
            0 -> Color.RED          // Pet
            1 -> Color.BLUE         // Margin
            2 -> Color.TRANSPARENT  // Background
            else -> Color.GREEN     // Cor default para classes inesperadas (improvável com argmax)
        }
    }

    fun close() {
        interpreter?.close()
    }
}