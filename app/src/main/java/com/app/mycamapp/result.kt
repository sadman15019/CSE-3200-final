package com.app.mycamapp

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.EditText
import android.widget.TextView

class result : AppCompatActivity() {
    lateinit var result:TextView
    lateinit var c:String
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_result)
        c=intent.getStringExtra("abc").toString()
        result=findViewById(R.id.resulttext)
        result.text=c
    }
}