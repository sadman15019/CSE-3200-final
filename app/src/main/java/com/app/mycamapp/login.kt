package com.app.mycamapp

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import com.app.mycamapp.databinding.ActivityLoginBinding
import com.app.mycamapp.databinding.ActivitySignupBinding
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.ktx.auth
import com.google.firebase.database.*
import com.google.firebase.ktx.Firebase

class login : AppCompatActivity() {
    private lateinit var binding: ActivityLoginBinding
    lateinit var signinbtn:Button
    private lateinit var database : DatabaseReference
    private lateinit var ref : DatabaseReference
    private lateinit var auth: FirebaseAuth
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLoginBinding.inflate(layoutInflater)
        setContentView(binding.root)
        auth = Firebase.auth
        signinbtn = findViewById(R.id.signinbtn)
        signinbtn.setOnClickListener {
                database = FirebaseDatabase.getInstance().getReference("account")
                // ref = FirebaseDatabase.getInstance().getReference().child("account")
                val email = binding.loginemail.text.toString()
                val pass = binding.loginpass.text.toString()
                if (email.trim().length <= 0) {
                    Toast.makeText(this, "email is required ", Toast.LENGTH_SHORT).show()
                } else if (pass.trim().length <= 0) {
                    Toast.makeText(this, "password is required", Toast.LENGTH_SHORT).show()
                }
                else {
                    auth.signInWithEmailAndPassword(email, pass)
                        .addOnCompleteListener(this) { task ->
                            if (task.isSuccessful) {
                                val user = auth.currentUser
                                Toast.makeText(this, "Login successful", Toast.LENGTH_SHORT).show()
                                val intent = Intent(this, DataUpdate::class.java)
                                intent.putExtra("Email", email)
                                startActivity(intent);
                                finish()
                            } else {
                                // If sign in fails, display a message to the user.

                                Toast.makeText(this, "Authentication failed.", Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
                }
            binding.gotosignup.setOnClickListener {
                val intent = Intent(this, signup::class.java)
                startActivity(intent);
                finish()
            }
        }
  /*  private fun getdata(e:String,p:String)
    {
            database.addValueEventListener(object :ValueEventListener{
            override fun onDataChange(snapshot: DataSnapshot) {
              for(ds in snapshot.children)
              {
                  val id=ds.key
                  val email=ds.child("Email").value.toString()
                  val pass=ds.child("Password").value.toString()
                  if(email==e)
                  {
                      x1=1
                  }
                  Log.d("sdf",pass)
                  Log.d("asda",p)
                  if(pass==p)
                  {
                      x2=1
                  }
              }
            }

            override fun onCancelled(error: DatabaseError) {

            }
        })
    }*/
}