package com.app.mycamapp

//import jdk.nashorn.internal.objects.NativeRegExp.source
import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.database.Cursor
import android.net.Uri
import android.os.*
import android.os.StrictMode.VmPolicy
import android.provider.MediaStore
import android.provider.Settings
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import com.app.mycamapp.databinding.ActivityUploadBinding
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import kotlin.Int as Int1


class upload : AppCompatActivity() {
    companion object {

        private const val STORAGE_CODE_PERMISSIONS = 100
        private const val PICKFILE_RESULT_CODE = 110
    }
    lateinit var age:String
    lateinit var gender:String
    var g: kotlin.Int =-1
    var a: kotlin.Int =-1
    lateinit var viewBinding: ActivityUploadBinding
    lateinit var t: TextView
    lateinit var b: Button
    lateinit var c: Button
    lateinit var selectbtn:Button

    lateinit var handler:Handler
    lateinit var filepath:String
    lateinit var filedir:String
    lateinit var filename:String
    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_upload)
        viewBinding= ActivityUploadBinding.inflate(layoutInflater)
        age=intent.getStringExtra("age").toString()
        gender=intent.getStringExtra("gender").toString()
        b=findViewById(R.id.uploadbutton)
        c=findViewById(R.id.selectbutton)
        t=findViewById(R.id.hello)
        selectbtn=findViewById(R.id.selectbutton)
        val builder = VmPolicy.Builder()
        StrictMode.setVmPolicy(builder.build())
       b.isVisible=false
        b.setOnClickListener {
           if(filedir.isNotEmpty() && filename.isNotEmpty())
           {getPythonStarted()}

            handler= Handler()
            handler.postDelayed({
                val intent = Intent(this,WaitActivity::class.java)
                startActivity(intent)
                finish()
            },20000)



        }
        c.setOnClickListener{
            val chooseFile: Intent
            val intent: Intent
            chooseFile = Intent(Intent.ACTION_GET_CONTENT)
            chooseFile.addCategory(Intent.CATEGORY_OPENABLE)
            val file = File(
                Environment.getExternalStorageDirectory().absolutePath, "Movies/CameraX-Video"

            )
            chooseFile.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            chooseFile.setDataAndType (  Uri.fromFile(file), "video/mp4")

            intent = Intent.createChooser(chooseFile, "Choose a file")
            startActivityForResult(intent, PICKFILE_RESULT_CODE)

        }



    }

//    private fun getRealPathFromUri(cntx:Context ,uri: Uri): String? {
//        var cursor: Cursor? = null
//        return try {
//            val arr = arrayOf(MediaStore.Images.Media.DATA)
//            cursor = cntx.getContentResolver().query(uri, arr, null, null, null)
//            val column_index: Int1 = cursor!!.getColumnIndexOrThrow(MediaStore.Images.Media.DATA)
//            cursor.moveToFirst()
//            cursor.getString(column_index)
//        } finally {
//            cursor?.close()
//        }
//    }
   override fun onActivityResult(requestCode: kotlin.Int, resultCode: kotlin.Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICKFILE_RESULT_CODE && resultCode == RESULT_OK) {
            val content_describer: Uri? = data?.data
            val src = content_describer!!.path

            val uriPathHelper = URIPathHelper()
             filepath = uriPathHelper.getPath(this, content_describer).toString()
            filedir = filepath.substring(0,filepath.lastIndexOf("/"))
             filename = filepath.substring(filepath.lastIndexOf("/")+1)

            selectbtn.text=filename.toString()
            b.isVisible=true


        }
    }
    private fun getPythonStarted(){
        if(gender=="Male")
        {
            g=1
        }
        else
        {
            g=0
        }
        a=age.toInt()
        if(!Python.isStarted())
        {
            Python.start(AndroidPlatform(this))
        }
        val python=Python.getInstance()
        val pythonfile=python.getModule("generate_dataset")
        val abc=pythonfile.callAttr("main",filedir,"/storage/emulated/0/Download/Output/ppg_feats.csv",a,g,filename)
        t.text = abc.toString()
    }


    private fun requestPermission(){
        if(Build.VERSION.SDK_INT>= Build.VERSION_CODES.R){
            //ANDROID is 11 or above
            try{
                val intent = Intent()
                intent.action= Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION
                val uri = Uri.fromParts("package",this.packageName,null)
                intent.data=uri
                storageActivityResultLauncher.launch(intent)
            }
            catch (e:Exception){
                val intent = Intent()
                intent.action = Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION
                storageActivityResultLauncher.launch(intent)
            }
        }
        else{
            //Android below 11
            ActivityCompat.requestPermissions(this, arrayOf(
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE),
                STORAGE_CODE_PERMISSIONS
            )
        }
    }

    private  val storageActivityResultLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()){
        if(Build.VERSION.SDK_INT>= Build.VERSION_CODES.R){
            if (Environment.isExternalStorageManager()){
                getPythonStarted()
            }
        }

        else{
            //below 11

        }

    }

    private fun checkPermission():Boolean{
        return if(Build.VERSION.SDK_INT>= Build.VERSION_CODES.R){
            Environment.isExternalStorageManager()
        }
        else{
            val write = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            val read = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
            write == PackageManager.PERMISSION_GRANTED && read == PackageManager.PERMISSION_GRANTED
        }
    }




    override fun onRequestPermissionsResult(
        requestCode: Int1, permissions: Array<String>, grantResults:
        IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if(requestCode== STORAGE_CODE_PERMISSIONS){
            if(grantResults.isNotEmpty()){
                val write = grantResults[0] == PackageManager.PERMISSION_GRANTED
                val read = grantResults[1] == PackageManager.PERMISSION_GRANTED
                if(read && write){
                    getPythonStarted()
                }
                else{
                    Toast.makeText(this,
                        "Permissions not granted by the user.",
                        Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

}