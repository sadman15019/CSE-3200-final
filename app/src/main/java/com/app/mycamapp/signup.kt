package com.app.mycamapp

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import com.app.mycamapp.databinding.ActivitySignupBinding
import com.google.firebase.database.*

class signup : AppCompatActivity() {
    private lateinit var binding:ActivitySignupBinding
    private lateinit var database : DatabaseReference
    private lateinit var ref : DatabaseReference
    var maxid:Int=1
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySignupBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.signupbtn.setOnClickListener {
            val email=binding.signupemail.text.toString()
            val name = binding.signuppass.text.toString()
            updateData(email,name)
        }
    }
    private fun updateData(email: String, pass: String) {

        database = FirebaseDatabase.getInstance().getReference("account")
        ref = FirebaseDatabase.getInstance().getReference().child("account")
        ref.addValueEventListener(object:ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                maxid=(snapshot.getChildrenCount().toInt())
                maxid++
            }

            override fun onCancelled(error: DatabaseError) {

            }

        })

        val user = mapOf<String,String>(
            "Email" to email,
            "Password" to pass
        )

            database.child(maxid.toString()).updateChildren(user).addOnSuccessListener {
            binding.signupemail.text.clear()
            binding.signuppass.text.clear()
            Toast.makeText(this,"Successfuly Updated", Toast.LENGTH_SHORT).show()
            //intent change
            //implement code here


        }.addOnFailureListener{

            Toast.makeText(this,"Failed to Update", Toast.LENGTH_SHORT).show()

        }}
}