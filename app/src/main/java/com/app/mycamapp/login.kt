package com.app.mycamapp

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import com.app.mycamapp.databinding.ActivityLoginBinding
import com.app.mycamapp.databinding.ActivitySignupBinding
import com.google.firebase.database.*

class login : AppCompatActivity() {
    private lateinit var binding: ActivityLoginBinding
    lateinit var signinbtn:Button
    private lateinit var database : DatabaseReference
    private lateinit var ref : DatabaseReference
    var maxid:Int=1
    var x:Int=1
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLoginBinding.inflate(layoutInflater)
        setContentView(binding.root)
        signinbtn=findViewById(R.id.signinbtn)
        signinbtn.setOnClickListener {
            database = FirebaseDatabase.getInstance().getReference("account")
           // ref = FirebaseDatabase.getInstance().getReference().child("account")
            val email=binding.loginemail.text.toString()
            val pass=binding.loginpass.text.toString()
            getdata(email,pass)
            if(x==1)
            {
                val intent= Intent(this,DataUpdate::class.java)
                intent.putExtra("Email",email)
                startActivity(intent);
                finish()
            }
            else
            {
                Toast.makeText(this,"Invalid email or password", Toast.LENGTH_SHORT).show()
            }
        }
        binding.gotosignup.setOnClickListener {
            val intent= Intent(this,signup::class.java)
            startActivity(intent);
            finish()
        }
    }
    private fun getdata(e:String,p:String)
    {
        database.addValueEventListener(object :ValueEventListener{
            override fun onDataChange(snapshot: DataSnapshot) {
              for(ds in snapshot.children)
              {
                  val id=ds.key
                  val email=ds.child("Email").value.toString()
                  val pass=ds.child("Password").value.toString()
                  if(email==e && pass==p)
                  {
                      x=1
                  }
              }
            }

            override fun onCancelled(error: DatabaseError) {

            }
        })
    }
}