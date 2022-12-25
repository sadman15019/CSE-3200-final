package com.app.mycamapp

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import android.widget.Toast
import com.app.mycamapp.databinding.ActivityDataUpdateBinding
import com.google.firebase.database.*

class DataUpdate : AppCompatActivity() {

    private lateinit var binding: ActivityDataUpdateBinding
    private lateinit var database : DatabaseReference
    private lateinit var sp: Spinner
    private lateinit var sp2: Spinner
    private lateinit var sp3: Spinner
    private lateinit var gender: String
    private lateinit var feet: String
    private lateinit var inch: String
    private var a: Double=1.0
    private var b: Double=1.0
    private lateinit var height: String
    private lateinit var email: String
    private lateinit var ref : DatabaseReference
    var x:Int=1
    private var tmp:String=""


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityDataUpdateBinding.inflate(layoutInflater)
        setContentView(binding.root)
        email=intent.getStringExtra("Email").toString()
        sp=binding.spinner
        sp2=binding.feet
        sp3=binding.inch
        val options= arrayOf("Male","Female")
        val options1= arrayOf("1","2","3","4","5","6","7")
        val options2= arrayOf("1","2","3","4","5","6","7","8","9","10","11","12")
        sp.adapter=ArrayAdapter<String>(this,android.R.layout.simple_dropdown_item_1line,options)
        sp.onItemSelectedListener=object : AdapterView.OnItemSelectedListener{
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
               gender=options.get(position)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
            //  println("nothing")
            }

        }
        sp2.adapter=ArrayAdapter<String>(this,android.R.layout.simple_dropdown_item_1line,options1)
        sp2.onItemSelectedListener=object : AdapterView.OnItemSelectedListener{
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                feet=options1.get(position)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                //  println("nothing")
            }

        }
        sp3.adapter=ArrayAdapter<String>(this,android.R.layout.simple_dropdown_item_1line,options2)
        sp3.onItemSelectedListener=object : AdapterView.OnItemSelectedListener{
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                inch=options2.get(position)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                //  println("nothing")
            }

        }

        binding.updateBtn.setOnClickListener {

            val nid= binding.NID.text.toString()
            val name = binding.Name.text.toString()
            val age = binding.Age.text.toString()
            val weight= binding.Weight.text.toString()
            a= feet.toDouble()
            a*=0.3048
            b= inch.toDouble()
            b*=0.0254
            a+=b
            height=a.toString()

            updateData(email,nid,name,gender,age,weight)
           val intent = Intent(this,instruction::class.java)
            intent.putExtra("Email",email)
            intent.putExtra("nid",nid)
            intent.putExtra("name",name)
            intent.putExtra("gender",gender)
            intent.putExtra("age",age)
            startActivity(intent)
            finish()

        }

    }
    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.menu,menu);
        return true
    }
    private fun updateData(e:String, nid: String, name: String, gender: String, age: String,weight:String) {
        database = FirebaseDatabase.getInstance().getReference("user_info")
        ref = FirebaseDatabase.getInstance().getReference().child("user_info")

        database.addValueEventListener(object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                for (ds in snapshot.children) {
                    val id = ds.key
                    val email = ds.child("Email").value.toString()
                    Log.d("sdfdf",id.toString())
                    Log.d("sdfdf",email)
                    if (email == e)  //if already exists a record for that person
                    {
                        tmp = id.toString()
                        x = 0
                    }
                }
            }

            override fun onCancelled(error: DatabaseError) {

            }
        })
        if (x == 1)  //create new node
        {
            val userid=database.push().key!!

            val user = mapOf<String, String>(
                "Email" to email,
                "Nid" to nid,
                "Name" to name,
                "Gender" to gender,
                "Age" to age,
                "Height" to height,
                "Weight" to weight
            )
            database.child(userid).setValue(user).addOnSuccessListener {
                binding.NID.text.clear()
                binding.Name.text.clear()
                binding.Weight.text.clear()
                binding.Age.text.clear()
                Toast.makeText(this, "Successfuly Updated", Toast.LENGTH_SHORT).show()
                //intent change
                //implement code here


            }.addOnFailureListener {

                Toast.makeText(this, "Failed to Update", Toast.LENGTH_SHORT).show()

            }

        }
        else  //update existing node
        {
            val user = mapOf<String, String>(
                "Email" to email,
                "Nid" to nid,
                "Name" to name,
                "Gender" to gender,
                "Age" to age,
                "Height" to height,
                "Weight" to weight
            )

            database.child(tmp).updateChildren(user).addOnSuccessListener {
                binding.NID.text.clear()
                binding.Name.text.clear()
                binding.Weight.text.clear()
                binding.Age.text.clear()
                Toast.makeText(this, "Successfuly Updated", Toast.LENGTH_SHORT).show()
                //intent change
                //implement code here


            }.addOnFailureListener {

                Toast.makeText(this, "Failed to Update", Toast.LENGTH_SHORT).show()

            }
        }
    }
}
