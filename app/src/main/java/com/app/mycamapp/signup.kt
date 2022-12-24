package com.app.mycamapp

import android.content.ContentValues.TAG
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import com.app.mycamapp.databinding.ActivitySignupBinding
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.ktx.auth
import com.google.firebase.database.*
import com.google.firebase.ktx.Firebase

class signup : AppCompatActivity() {
    private lateinit var binding:ActivitySignupBinding
    private lateinit var database : DatabaseReference
    private lateinit var ref : DatabaseReference
    private lateinit var auth: FirebaseAuth
    var maxid:Int=1
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySignupBinding.inflate(layoutInflater)
        setContentView(binding.root)
        auth = Firebase.auth
        binding.signupbtn.setOnClickListener {
            val email=binding.signupemail.text.toString()
            val pass = binding.signuppass.text.toString()
            if(email.trim().length<=0)
            {
                Toast.makeText(this,"An email is required ", Toast.LENGTH_SHORT).show()
            }
            else if(pass.trim().length<=0)
            {
                Toast.makeText(this,"password is required", Toast.LENGTH_SHORT).show()
            }
            else {
                auth.createUserWithEmailAndPassword(email, pass).addOnCompleteListener{
                        if (it.isSuccessful) {

                            Toast.makeText(this, "User created", Toast.LENGTH_SHORT).show()
                        } else {


                            Toast.makeText(this, "Authentication failed", Toast.LENGTH_SHORT).show()
                        }
                    }

               updateData(email, pass)
               val intent = Intent(this, login::class.java)
               startActivity(intent);
                finish()
            }
        }
    }
    private fun updateData(email: String, pass: String) {

        database = FirebaseDatabase.getInstance().getReference("account")
       /* ref = FirebaseDatabase.getInstance().getReference().child("account")
        ref.addValueEventListener(object:ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                maxid=(snapshot.getChildrenCount().toInt())
                maxid++
            }

            override fun onCancelled(error: DatabaseError) {

            }

        })*/

        val userid=database.push().key!!

        val user = mapOf<String,String>(
            "Email" to email,
            "Password" to pass
             )

            database.child(userid).setValue(user).addOnSuccessListener {
            binding.signupemail.text.clear()
            binding.signuppass.text.clear()
            Toast.makeText(this,"Successfuly Updated", Toast.LENGTH_SHORT).show()
            //intent change
            //implement code here


        }.addOnFailureListener{

            Toast.makeText(this,"Failed to Update", Toast.LENGTH_SHORT).show()

        }}
}