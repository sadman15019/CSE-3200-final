package com.app.mycamapp

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.EditText
import android.widget.TextView

class result : AppCompatActivity() {
    lateinit var result:TextView
    lateinit var gl_result:TextView
    lateinit var c:String
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_result)
        c=intent.getStringExtra("abc").toString()
        result=findViewById(R.id.resulttext)
        gl_result=findViewById(R.id.GL)
        result.text=c
        val delim = ":"
        val arr = c.split(delim).toTypedArray()
        gl_result.text=arr[1]

    }
}