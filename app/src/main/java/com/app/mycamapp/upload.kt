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
import android.view.Menu
import android.view.MenuItem
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import com.app.mycamapp.databinding.ActivityUploadBinding
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.database.DatabaseReference
import com.google.firebase.database.FirebaseDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import kotlin.Int as Int1
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.time.format.FormatStyle
import java.util.*


class upload : AppCompatActivity() {
    companion object {

        private const val STORAGE_CODE_PERMISSIONS = 100
        private const val PICKFILE_RESULT_CODE = 110
    }
    private lateinit var database : DatabaseReference
    lateinit var age: String
    lateinit var gender: String
    lateinit var email: String
    lateinit var abc:String
    var g: kotlin.Int = -1
    var a: kotlin.Int = -1
    lateinit var viewBinding: ActivityUploadBinding
   // lateinit var t: TextView
    lateinit var b: Button
    lateinit var c: Button
    lateinit var selectbtn: Button
    private lateinit var user:FirebaseAuth

    lateinit var handler: Handler
    lateinit var filepath: String
    lateinit var filedir: String
    lateinit var filename: String
    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_upload)
        viewBinding = ActivityUploadBinding.inflate(layoutInflater)
        email = intent.getStringExtra("Email").toString()
        age = intent.getStringExtra("age").toString()
        gender = intent.getStringExtra("gender").toString()
        user=FirebaseAuth.getInstance()
        b = findViewById(R.id.uploadbutton)
        c = findViewById(R.id.selectbutton)
        //t = findViewById(R.id.hello)
        selectbtn = findViewById(R.id.selectbutton)
        val builder = VmPolicy.Builder()
        StrictMode.setVmPolicy(builder.build())
        b.isVisible = false
        b.setOnClickListener {
            if (filedir.isNotEmpty() && filename.isNotEmpty()) {

                GlobalScope.launch {
                    var intent = Intent(this@upload, WaitActivity::class.java)
                    /* intent.putExtra("age", age)
                     intent.putExtra("gender", gender)
                     intent.putExtra("filename", filename)
                     intent.putExtra("filedir", filedir)*/

                    startActivity(intent)


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
                        Python.start(AndroidPlatform(this@upload))
                    }
                    val python=Python.getInstance()
                    val pythonfile=python.getModule("generate_dataset")
                    abc=pythonfile.callAttr("main",filedir,"/storage/emulated/0/Download/Output/ppg_feats.csv",a,g,filename).toString()
                    val delim = ":"
                    val arr = abc.split(delim).toTypedArray()
                    var gl=arr[1].toString()
                    var hb=arr[0].toString()
                    gl=gl.substring(2,5)
                    hb=hb.substring(2,6)
                    val re = Regex("[^0-9.]")
                    gl = re.replace(gl, "") // works
                    hb = re.replace(hb, "")
                    if (email.trim().length > 0 &&  hb.trim().length > 0) {
                        updateData(email, gl, hb)
                    }
                    intent = Intent(this@upload, result::class.java)
                    intent.putExtra("Email",email)
                    intent.putExtra("gl",gl)
                    intent.putExtra("hb", hb)
                    startActivity(intent)
                    finish()

                }

                }




            }
            c.setOnClickListener {
                if(checkPermission()==false){
                    requestPermission()
                }
                val chooseFile: Intent
                val intent: Intent
                chooseFile = Intent(Intent.ACTION_GET_CONTENT)
                chooseFile.addCategory(Intent.CATEGORY_OPENABLE)
                val file = File(
                    Environment.getExternalStorageDirectory().absolutePath, "Movies/CameraX-Video"

                )
                chooseFile.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                chooseFile.setDataAndType(Uri.fromFile(file), "video/mp4")

                intent = Intent.createChooser(chooseFile, "Choose a file")
                startActivityForResult(intent, PICKFILE_RESULT_CODE)

            }


        }

    fun recordShow(){
        val intent = Intent(this,record::class.java)
        startActivity(intent)
        finish()
    }
    fun fragmentShow(){
        val dialog = BottomSheetDialog(this)

        // on below line we are inflating a layout file which we have created.
        val view = layoutInflater.inflate(R.layout.bottom_sheet_dialog, null)

        // on below line we are creating a variable for our button
        // which we are using to dismiss our dialog.
        val btnnext = view.findViewById<Button>(R.id.nextbtn)

        // on below line we are adding on click listener
        // for our dismissing the dialog button.
        btnnext.setOnClickListener {
            // on below line we are calling a dismiss
            // method to close our dialog.
            dialog.dismiss()
            val dialog = BottomSheetDialog(this)

            // on below line we are inflating a layout file which we have created.
            val view = layoutInflater.inflate(R.layout.bottom_sheet_dialog2, null)

            // on below line we are creating a variable for our button
            // which we are using to dismiss our dialog.
            val btndone = view.findViewById<Button>(R.id.donebtn)

            // on below line we are adding on click listener
            // for our dismissing the dialog button.
            btndone.setOnClickListener {
                // on below line we are calling a dismiss
                // method to close our dialog.
                dialog.dismiss()
            }
            // below line is use to set cancelable to avoid
            // closing of dialog box when clicking on the screen.
            dialog.setCancelable(false)

            // on below line we are setting
            // content view to our view.
            dialog.setContentView(view)

            // on below line we are calling
            // a show method to display a dialog.
            dialog.show()
        }
        // below line is use to set cancelable to avoid
        // closing of dialog box when clicking on the screen.
        dialog.setCancelable(false)

        // on below line we are setting
        // content view to our view.
        dialog.setContentView(view)

        // on below line we are calling
        // a show method to display a dialog.
        dialog.show()
    }
    fun signOut(){
        user= FirebaseAuth.getInstance()
        user.signOut()
        val intent = Intent(this,StartupActivity::class.java)
        startActivity(intent)
        finish()
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.menu,menu);
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        val id = item.itemId;

        if (id == R.id.logout_action){

            signOut()
            return true
        }
        if(id==R.id.settings_action){
            // on below line we are creating a new bottom sheet dialog.


            fragmentShow()
            return true
        }
        if(id==R.id.record_action){

            recordShow()
            return true

        }

        return super.onOptionsItemSelected(item)
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
        override fun onActivityResult(
            requestCode: kotlin.Int,
            resultCode: kotlin.Int,
            data: Intent?
        ) {
            super.onActivityResult(requestCode, resultCode, data)
            if (requestCode == PICKFILE_RESULT_CODE && resultCode == RESULT_OK) {
                val content_describer: Uri? = data?.data
                val src = content_describer!!.path

                val uriPathHelper = URIPathHelper()
                filepath = uriPathHelper.getPath(this, content_describer).toString()
                filedir = filepath.substring(0, filepath.lastIndexOf("/"))
                filename = filepath.substring(filepath.lastIndexOf("/") + 1)

                selectbtn.text = filename.toString()
                b.isVisible = true


            }
        }

        private fun getPythonStarted() {
            if (gender == "Male") {
                g = 1
            } else {
                g = 0
            }
            a = age.toInt()
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(this))
            }
            val python = Python.getInstance()
            val pythonfile = python.getModule("generate_dataset")
            val abc = pythonfile.callAttr(
                "main",
                filedir,
                "/storage/emulated/0/Download/Output/ppg_feats.csv",
                a,
                g,
                filename
            )

            //return abc
            //t.text = abc.toString()
        }
    private fun updateData(email: String, gl: String,hb:String) {
        val formatter = SimpleDateFormat("yyyy-MM-dd")
        val date = Date()
        val curdate = formatter.format(date)

        database = FirebaseDatabase.getInstance().getReference("user_record")

        val userid=database.push().key!!

        val user = mapOf<String,String>(
            "Email" to email,
            "GLucose" to gl,
            "Hemo" to hb,
            "Date" to curdate
        )

        database.child(userid).setValue(user).addOnSuccessListener {
            Toast.makeText(this,"Successfuly Updated", Toast.LENGTH_SHORT).show()
            //intent change
            //implement code here


        }.addOnFailureListener{

            Toast.makeText(this,"Failed to Update", Toast.LENGTH_SHORT).show()

        }}

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

        private val storageActivityResultLauncher =
            registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    if (Environment.isExternalStorageManager()) {
                        getPythonStarted()
                    }
                } else {
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
            IntArray
        ) {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults)

            if (requestCode == STORAGE_CODE_PERMISSIONS) {
                if (grantResults.isNotEmpty()) {
                    val write = grantResults[0] == PackageManager.PERMISSION_GRANTED
                    val read = grantResults[1] == PackageManager.PERMISSION_GRANTED
                    if (read && write) {
                        getPythonStarted()
                    } else {
                        Toast.makeText(
                            this,
                            "Permissions not granted by the user.",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }
        }


    }
