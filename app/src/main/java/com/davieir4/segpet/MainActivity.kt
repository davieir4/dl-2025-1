package com.davieir4.segpet


import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity() {

    private lateinit var btnSelectImage: Button
    private lateinit var btnSegment: Button
    private lateinit var imgOriginal: ImageView
    private lateinit var imgSegmented: ImageView
    private lateinit var txtStatus: TextView

    private var originalBitmap: Bitmap? = null
    private var segmentationProcessor: SegmentationProcessor? = null

    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                loadImageFromUri(uri)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        setupListeners()
        initializeProcessor()
        checkPermissions()
    }

    private fun initViews() {
        btnSelectImage = findViewById(R.id.btnSelectImage)
        btnSegment = findViewById(R.id.btnSegment)
        imgOriginal = findViewById(R.id.imgOriginal)
        imgSegmented = findViewById(R.id.imgSegmented)
        txtStatus = findViewById(R.id.txtStatus)
    }

    private fun setupListeners() {
        btnSelectImage.setOnClickListener {
            openImagePicker()
        }

        btnSegment.setOnClickListener {
            performSegmentation()
        }
    }

    private fun initializeProcessor() {
        segmentationProcessor = SegmentationProcessor(this)
    }

    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                PERMISSION_REQUEST_CODE)
        }
    }

    private fun openImagePicker() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        imagePickerLauncher.launch(intent)
    }

    private fun loadImageFromUri(uri: Uri) {
        try {
            val inputStream = contentResolver.openInputStream(uri)
            originalBitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            imgOriginal.setImageBitmap(originalBitmap)
            btnSegment.isEnabled = true
            txtStatus.text = "Imagem carregada. Clique em 'Fazer Segmentação'"

        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Erro ao carregar imagem", Toast.LENGTH_SHORT).show()
        }
    }

    private fun performSegmentation() {
        originalBitmap?.let { bitmap ->
            txtStatus.text = "Processando segmentação..."
            btnSegment.isEnabled = false

            // Executar em background thread
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    val segmentedBitmap = segmentationProcessor?.processImage(bitmap)

                    // Criar imagem combinada (original + máscara)
                    val combinedBitmap = combineImages(bitmap, segmentedBitmap)

                    // Voltar para a UI thread
                    withContext(Dispatchers.Main) {
                        imgSegmented.setImageBitmap(combinedBitmap)
                        txtStatus.text = "Segmentação concluída!"
                        btnSegment.isEnabled = true
                    }

                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        e.printStackTrace()
                        txtStatus.text = "Erro na segmentação"
                        btnSegment.isEnabled = true
                        Toast.makeText(this@MainActivity, "Erro: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                }
            }
        }
    }

    private fun combineImages(original: Bitmap, mask: Bitmap?): Bitmap {
        if (mask == null) return original

        val result = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        val paint = Paint()

        // Desenhar imagem original
        canvas.drawBitmap(original, 0f, 0f, paint)

        // Desenhar máscara com transparência
        paint.alpha = 128 // 50% de transparência
        val scaledMask = Bitmap.createScaledBitmap(mask, original.width, original.height, false)
        canvas.drawBitmap(scaledMask, 0f, 0f, paint)

        return result
    }

    override fun onDestroy() {
        super.onDestroy()
        segmentationProcessor?.close()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permissão concedida", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Permissão necessária para acessar imagens", Toast.LENGTH_LONG).show()
            }
        }
    }

    companion object {
        private const val PERMISSION_REQUEST_CODE = 100
    }
}