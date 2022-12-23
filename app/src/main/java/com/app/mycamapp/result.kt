package com.app.mycamapp

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import com.google.firebase.database.DataSnapshot
import com.google.firebase.database.DatabaseError
import com.google.firebase.database.DatabaseReference
import com.google.firebase.database.FirebaseDatabase
import com.google.firebase.database.ValueEventListener

class result : AppCompatActivity() {
    lateinit var gl_result:TextView
    lateinit var hb_result:TextView
    lateinit var ref:DatabaseReference
    lateinit var yo:TextView
    lateinit var gl:String
    lateinit var hb:String
    lateinit var c:String
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_result)
        gl=intent.getStringExtra("gl").toString()
        hb=intent.getStringExtra("hb").toString()
        gl_result=findViewById(R.id.GL)
        hb_result=findViewById(R.id.Hb)
        yo=findViewById(R.id.hbtext)
        /*val delim = ":"
        val arr = c.split(delim).toTypedArray()
        gl=arr[1].toString()
        hb=arr[0].toString()
        gl=gl.substring(2,5)
        hb=hb.substring(2,6)*/
        gl_result.text=gl
        hb_result.text=hb
        yo.text="Hb"


    }
}