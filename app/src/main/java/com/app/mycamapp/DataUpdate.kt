package com.app.mycamapp

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.app.mycamapp.databinding.ActivityDataUpdateBinding
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.firebase.auth.FirebaseAuth
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
    private lateinit var user:FirebaseAuth


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityDataUpdateBinding.inflate(layoutInflater)
        setContentView(binding.root)
       // email=intent.getStringExtra("Email").toString()
        user=FirebaseAuth.getInstance()
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
            email = getemail()
            if (email == "") {
                val intent = Intent(this, login::class.java)
                startActivity(intent)
                finish()
            }
            else {
                val name = binding.Name.text.toString()
                val age = binding.Age.text.toString()
                val weight = binding.Weight.text.toString()
                a = feet.toDouble()
                a *= 0.3048
                b = inch.toDouble()
                b *= 0.0254
                a += b
                height = a.toString()

                 if (age.trim().length <= 0) {
                    Toast.makeText(this, "age is required", Toast.LENGTH_SHORT).show()
                }
                 else {

                    updateData(email, name, gender, age, weight)
                    val intent = Intent(this, instruction::class.java)
                    intent.putExtra("Email", email)
                    intent.putExtra("name", name)
                    intent.putExtra("gender", gender)
                    intent.putExtra("age", age)
                    startActivity(intent)


                }
            }
        }
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.menu,menu);
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        val id = item.itemId;
        if (id == R.id.logout_action){

             signOut()
            return true
        }
        if(id==R.id.settings_action){
            // on below line we are creating a new bottom sheet dialog.
          fragmentShow()
            return true
        }
        if(id==R.id.record_action){

            recordShow()
            return true

        }

        return super.onOptionsItemSelected(item)
    }
    fun recordShow(){
        val intent = Intent(this,record::class.java)
        startActivity(intent)
        finish()
    }
    fun fragmentShow(){
        val dialog = BottomSheetDialog(this)

        // on below line we are inflating a layout file which we have created.
        val view = layoutInflater.inflate(R.layout.bottom_sheet_dialog, null)

        // on below line we are creating a variable for our button
        // which we are using to dismiss our dialog.
        val btnnext = view.findViewById<Button>(R.id.nextbtn)

        // on below line we are adding on click listener
        // for our dismissing the dialog button.
        btnnext.setOnClickListener {
            // on below line we are calling a dismiss
            // method to close our dialog.
            dialog.dismiss()
            val dialog = BottomSheetDialog(this)

            // on below line we are inflating a layout file which we have created.
            val view = layoutInflater.inflate(R.layout.bottom_sheet_dialog2, null)

            // on below line we are creating a variable for our button
            // which we are using to dismiss our dialog.
            val btndone = view.findViewById<Button>(R.id.donebtn)

            // on below line we are adding on click listener
            // for our dismissing the dialog button.
            btndone.setOnClickListener {
                // on below line we are calling a dismiss
                // method to close our dialog.
                dialog.dismiss()
            }
            // below line is use to set cancelable to avoid
            // closing of dialog box when clicking on the screen.
            dialog.setCancelable(false)

            // on below line we are setting
            // content view to our view.
            dialog.setContentView(view)

            // on below line we are calling
            // a show method to display a dialog.
            dialog.show()
        }
        // below line is use to set cancelable to avoid
        // closing of dialog box when clicking on the screen.
        dialog.setCancelable(false)

        // on below line we are setting
        // content view to our view.
        dialog.setContentView(view)

        // on below line we are calling
        // a show method to display a dialog.
        dialog.show()
    }
 fun signOut(){
     user=FirebaseAuth.getInstance()
     user.signOut()
     val intent = Intent(this,StartupActivity::class.java)
     startActivity(intent)
     finish()
 }
    private fun updateData(e:String, name: String, gender: String, age: String,weight:String) {

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
                "Name" to name,
                "Gender" to gender,
                "Age" to age,
                "Height" to height,
                "Weight" to weight
            )
            database.child(userid).setValue(user).addOnSuccessListener {
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
                "Name" to name,
                "Gender" to gender,
                "Age" to age,
                "Height" to height,
                "Weight" to weight
            )

            database.child(tmp).updateChildren(user).addOnSuccessListener {
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
    fun getemail():String
    {
        val curuser = FirebaseAuth.getInstance().currentUser
        var abc:String
        if (curuser != null) {
            abc=curuser.email.toString()
        } else {
           abc=""
        }
        return abc
    }
}
