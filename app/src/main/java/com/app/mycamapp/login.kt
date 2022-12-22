package com.app.mycamapp

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button

class login : AppCompatActivity() {
    lateinit var signinbtn:Button
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)
        signinbtn=findViewById(R.id.signinbtn)
        signinbtn.setOnClickListener {
            val intent= Intent(this,DataUpdate::class.java)
            startActivity(intent);
            finish()

        }
    }
}