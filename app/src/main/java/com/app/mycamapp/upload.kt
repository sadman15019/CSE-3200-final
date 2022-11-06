package com.app.mycamapp

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.widget.Button
import android.widget.TextView
import androidx.fragment.app.Fragment
import com.app.mycamapp.databinding.ActivityUploadBinding
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform


class upload : AppCompatActivity() {
    lateinit var viewBinding: ActivityUploadBinding
    lateinit var t: TextView
    lateinit var b: Button
  lateinit var handler:Handler
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_upload)
        viewBinding= ActivityUploadBinding.inflate(layoutInflater)
        if(!Python.isStarted())
        {
            Python.start(AndroidPlatform(this))
        }
        val python=Python.getInstance()
        val pythonfile=python.getModule("generate_dataset")
        val abc=pythonfile.callAttr("main","/storage/emulated/0/Movies/CameraX-Video/")
        b=findViewById(R.id.uploadbutton)
        t=findViewById(R.id.hello)
        b.setOnClickListener {
            t.text = abc.toString()

            handler= Handler()
            handler.postDelayed({
                val intent = Intent(this,WaitActivity::class.java)
                startActivity(intent)
                finish()
            },5000)



        }

    }


}