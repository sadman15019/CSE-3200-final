package com.app.mycamapp

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import com.google.firebase.auth.FirebaseAuth

class StartupActivity : AppCompatActivity() {
    lateinit var sign:Button
    lateinit var log:Button
    lateinit var email:String
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_startup)
        sign=findViewById(R.id.email_button1)
        log=findViewById(R.id.google_button)
        val user = FirebaseAuth.getInstance().currentUser
        if (user != null) {
            email = user.email.toString()
            val intent = Intent(this,DataUpdate::class.java)
            startActivity(intent)
        } else {

        }

        sign.setOnClickListener{

            val intent = Intent(this,signup::class.java)
            startActivity(intent)





        }

        log.setOnClickListener{

            val intent = Intent(this,login::class.java)
            startActivity(intent)





        }

    }
}