package com.app.mycamapp

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.provider.Settings
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.app.mycamapp.databinding.ActivityUploadBinding
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform


class upload : AppCompatActivity() {
    companion object {

        private const val STORAGE_CODE_PERMISSIONS = 100
    }
    lateinit var viewBinding: ActivityUploadBinding
    lateinit var t: TextView
    lateinit var b: Button
  lateinit var handler:Handler
    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_upload)
        viewBinding= ActivityUploadBinding.inflate(layoutInflater)
        b=findViewById(R.id.uploadbutton)
        t=findViewById(R.id.hello)


        b.setOnClickListener {

            getPythonStarted()
            handler= Handler()
            handler.postDelayed({
                val intent = Intent(this,WaitActivity::class.java)
                startActivity(intent)
                finish()
            },5000)



        }

    }

    private fun getPythonStarted(){
        if(!Python.isStarted())
        {
            Python.start(AndroidPlatform(this))
        }
        val python=Python.getInstance()
        val pythonfile=python.getModule("generate_dataset")
        val abc=pythonfile.callAttr("main","/storage/emulated/0/Download","/storage/emulated/0/Download/Output/ppg_feats.csv")
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
        requestCode: Int, permissions: Array<String>, grantResults:
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